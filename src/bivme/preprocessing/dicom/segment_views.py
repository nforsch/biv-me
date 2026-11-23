import os
import sys
import shutil
import subprocess
from pathlib import Path
import torch
import nibabel as nib
import numpy as np
import pandas as pd

# Set nnUNet environment variables so it doesn't scream at you with warnings
os.environ['nnUNet_raw'] = '.'
os.environ['nnUNet_preprocessed'] = '.'
os.environ['nnUNet_results'] = '.'

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from bivme.preprocessing.dicom.src.utils import write_nifti
from bivme.preprocessing.dicom.src.utils import write_sliceinfofile


# view -> (3D dataset, 2D dataset); the 2D models segment each frame of the same 2D+time inputs independently
DATASETS = {'SAX': ("Dataset260_SAX_3D", "Dataset270_SAX_2D"), '2ch': ("Dataset261_2ch_3D", "Dataset271_2ch_2D"),
            '3ch': ("Dataset262_3ch_3D", "Dataset272_3ch_2D"), '4ch': ("Dataset263_4ch_3D", "Dataset273_4ch_2D"),
            'RVOT': ("Dataset264_RVOT_3D", "Dataset274_RVOT_2D")}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # supported by nnunetv2>=2.8; ~3.5x faster than CPU on M4 Pro
    return torch.device("cpu")


def init_nnUNetv2(model_folder, my_logger):
    device = get_device()
    my_logger.info(f'nnU-Net device: {device}')

    predictor = nnUNetPredictor(tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )
    return predictor


def summarise_probabilities(view_output_folder, view, uncertain_below=0.6):
    """Per-slice confidence from nnU-Net softmax files; deletes the .npz/.pkl afterwards (they are large)."""
    rows = []
    for f in sorted(os.listdir(view_output_folder)):
        if not f.endswith('.npz'):
            continue
        path = os.path.join(view_output_folder, f)
        probs = np.load(path)['probabilities']  # (classes, ...) softmax
        max_p = probs.max(axis=0)
        fg = probs.argmax(axis=0) > 0
        fg_p = max_p[fg]
        rows.append({'View': view, 'Slice': f[:-4],
                     'Mean confidence': float(fg_p.mean()) if fg_p.size else float('nan'),
                     'Uncertain fraction': float((fg_p < uncertain_below).mean()) if fg_p.size else float('nan'),
                     'Foreground voxels': int(fg_p.size)})
        os.remove(path)
        pkl = path[:-4] + '.pkl'
        if os.path.exists(pkl):
            os.remove(pkl)
    return rows


def predict_view_cli(view_input_folder, view_output_folder, model_folder, pp_workers, export_workers, my_logger, save_probabilities=False):
    """2D models run through the nnU-Net CLI. The 2D models only have fold_0 (the CLI default is folds 0-4)."""
    device = get_device()
    my_logger.info(f'nnU-Net (CLI, 2D) device: {device}')
    cmd = [str(Path(sys.executable).parent / 'nnUNetv2_predict_from_modelfolder'),
           '-i', view_input_folder, '-o', view_output_folder, '-m', model_folder, '-f', '0',
           '-device', device.type, '-npp', str(pp_workers), '-nps', str(export_workers), '--disable_progress_bar']
    if save_probabilities:
        cmd.append('--save_probabilities')
    subprocess.run(cmd, check=True)  # a failed prediction must fail the case, not leave empty contours


def predict_view(input_folder, output_folder, model, view, dataset, workers, my_logger, save_probabilities=False, seg_model='3d'):
    # Split workers between preprocessing and segmentation export
    # Will use only 1 worker for export, and the rest for preprocessing as it is more intensive
    pp_workers = max(workers-1, 1)  # At least 1 worker for preprocessing
    export_workers = max(workers - pp_workers, 1)  # At least 1 worker for export

    # Define the trained model to use (Specified by the Task)
    plans = 'nnUNetTrainer__nnUNetPlans__2d' if seg_model == '2d' else 'nnUNetTrainer__nnUNetPlans__3d_fullres'
    model_folder_name = os.path.join(model, "Segmentation", dataset, plans)
    
    view_input_folder = os.path.join(input_folder, view, 'resized-cropped-normalised')
    view_output_folder = os.path.join(output_folder, view, 'raw')
        
    if len(os.listdir(view_input_folder)) > 0 and seg_model == '2d':
        predict_view_cli(view_input_folder, view_output_folder, model_folder_name, pp_workers, export_workers, my_logger, save_probabilities)
        my_logger.info(f'Done with {view}')
    elif len(os.listdir(view_input_folder)) > 0:
        # Initialize nnUNet model
        predictor = init_nnUNetv2(model_folder_name, my_logger)

        # Make predictions
        predictor.predict_from_files(
            view_input_folder,
            view_output_folder,
            save_probabilities=save_probabilities, overwrite=True, num_processes_preprocessing=pp_workers, 
            num_processes_segmentation_export=export_workers,
            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
        )

        my_logger.info(f'Done with {view}')


def reassemble_full_segmentation(input_folder, output_folder, view, slice_info_df, my_logger):
    original_view_input_folder = os.path.join(input_folder, view, 'resized')
    view_output_folder = os.path.join(output_folder, view, 'raw')
    new_view_output_folder = os.path.join(output_folder, view, 'uncropped')

    os.makedirs(new_view_output_folder, exist_ok=True)

    img_seg_dict = {}
    original_images = os.listdir(original_view_input_folder)
    segmentations = [f for f in os.listdir(view_output_folder) if f.endswith('.nii.gz')]

    for img_file in original_images:
        img_id = img_file.split('_')[2]
        # Look for corresponding segmentation file
        seg_file = None
        for seg in segmentations:
            if seg.split('_')[2].replace('.nii.gz','') == img_id:
                seg_file = seg
                break

        img_seg_dict[img_id] = {'img_file': img_file, 'seg_file': seg_file}

    for img_id in img_seg_dict.keys():
        if img_seg_dict[img_id]['seg_file'] is None:
            my_logger.warning(f'No segmentation found for slice {img_id}, skipping...')
            continue

        # Read original image to get dimensions
        img_nii = nib.load(os.path.join(original_view_input_folder, img_seg_dict[img_id]['img_file']))
        img_data = img_nii.get_fdata()
        img_shape = img_data.shape

        # Read segmentation
        seg_nii = nib.load(os.path.join(view_output_folder, img_seg_dict[img_id]['seg_file']))
        seg_data = seg_nii.get_fdata()
        seg_shape = seg_data.shape

        # Create full segmentation array
        full_seg = np.zeros(img_shape, dtype=seg_data.dtype)
        start_x = (img_shape[0] - seg_shape[0]) // 2
        start_y = (img_shape[1] - seg_shape[1]) // 2

        # Pad segmentation back to original size
        full_seg[start_x:start_x+seg_shape[0], start_y:start_y+seg_shape[1], :] = seg_data
        # Save uncropped segmentation
        full_seg_nii = nib.Nifti1Image(full_seg, img_nii.affine, img_nii.header)
        nib.save(full_seg_nii, os.path.join(new_view_output_folder, f'{view}_3d_{img_id}.nii.gz'))


def segment_views(dst, model, slice_info_df, workers, my_logger, save_confidence=False, seg_model='3d'):
    # define I/O parameters for nnUnet segmentation
    input_folder = os.path.join(dst, 'images')
    output_folder = os.path.join(dst, 'segmentations')

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    else:
        shutil.rmtree(input_folder)
        os.makedirs(input_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        
    if seg_model not in ('3d', '2d'):
        raise ValueError(f'Unknown segmentation model "{seg_model}", use "3d" or "2d".')
    views = list(DATASETS)
    confidence_rows = []

    for view in views:
        if len(slice_info_df[slice_info_df['View'] == view]) == 0:
            my_logger.info(f'No {view} images found, skipping...')
            continue

        os.makedirs(os.path.join(input_folder, view), exist_ok=True)
        os.makedirs(os.path.join(output_folder, view), exist_ok=True)
        
        my_logger.info(f'Writing {view} images to nifti files...')

        view_rows = slice_info_df[slice_info_df['View'] == view]
        for j, row in view_rows.iterrows():
            slice_id = row['Slice ID']
            pixel_array = row['Img']
            pixel_spacing = row['Pixel Spacing']
            rescale_factor = write_nifti(slice_id, pixel_array, pixel_spacing, input_folder, view)

            if rescale_factor != 1:
                # Update pixel spacing
                idx = slice_info_df.index[slice_info_df['Slice ID'] == slice_id].tolist()[0]
                # Use idx to update the original slice_info_df
                slice_info_df.at[idx, 'Pixel Spacing'] = [pixel_spacing[0]*rescale_factor, pixel_spacing[1]*rescale_factor]

        my_logger.info(f'Segmenting {view} images...')
        
        dataset = DATASETS[view][seg_model == '2d']

        predict_view(input_folder, output_folder, model, view, dataset, workers, my_logger, save_probabilities=save_confidence, seg_model=seg_model) # generate segmentations for each view
        if save_confidence:
            confidence_rows += summarise_probabilities(os.path.join(output_folder, view, 'raw'), view)

        reassemble_full_segmentation(input_folder, output_folder, view, slice_info_df, my_logger) # convert segmentations back to original dimensions by padding (i.e. 'uncropping')

    if save_confidence:
        pd.DataFrame(confidence_rows).to_csv(os.path.join(output_folder, 'segmentation_confidence.csv'), index=False)

    # Write updated slice info file with new pixel spacings
    write_sliceinfofile(dst, slice_info_df)