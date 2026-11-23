"""Full biv-me pipeline (preprocessing + fitting) on MAD_OUS_sorted, with 3D or 2D nnU-Net segmentation.

    .venv/bin/python scripts/run_MAD_OUS.py 3d            # native nnU-Net 3D models (Python API)
    .venv/bin/python scripts/run_MAD_OUS.py 2d            # nnU-Net 2D models via nnUNetv2_predict_from_modelfolder
    .venv/bin/python scripts/run_MAD_OUS.py 2d -case 102  # one case

Run either first: view predictions land in <out>/states and the second run reuses them.
biv-me reads <out>/source/<ID> -> <data>/<ID>/cine. Per-run folders: <out>/{processing,guidepoints,fitted-models}_<seg>. Summaries in <out>:
biv-me_MAD_OUS_<seg>.md/.csv per run and biv-me_MAD_OUS_compare.md with both runs side by side.
Re-running resumes: OK/WARN cases are kept, FAIL cases are re-run.
"""
import argparse
from pathlib import Path

import tomli

from bivme.batch import run_batch

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / 'RAW_DATA/ProCardio/MAD/data_raw/MAD_OUS_sorted'
OUT = Path.home() / 'RAW_DATA/ProCardio/MAD/data_processed/biv-me'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('seg', choices=['3d', '2d'])
    parser.add_argument('-case', default=None, help='Only run this case')
    parser.add_argument('-data', type=Path, default=DATA)
    parser.add_argument('-out', type=Path, default=OUT)
    parser.add_argument('-workers', type=int, default=None, help='Override [multiprocessing] workers')
    args = parser.parse_args()

    # MAD_OUS has no SeriesDescription tags, so biv-me cannot filter non-cines by name: give it only <ID>/cine
    source = args.out / 'source'
    source.mkdir(parents=True, exist_ok=True)
    for cine in sorted(args.data.glob('*/cine')):
        link = source / cine.parent.name
        if not link.exists():
            link.symlink_to(cine, target_is_directory=True)

    with open(REPO / 'src/bivme/configs/config.toml', 'rb') as fp:
        config = tomli.load(fp)
    config['input_pp'].update(source=str(source), batch_ID='MAD_OUS', processing=str(args.out / 'processing'),
                              states=str(args.out / 'states'))
    config['output_pp']['output_directory'] = str(args.out / 'guidepoints')
    config['output_fitting']['output_directory'] = str(args.out / 'fitted-models')
    config['view-selection']['option'] = 'default'  # first run classifies; the batch switches to 'load' once states exist
    if args.workers:
        config['multiprocessing']['workers'] = args.workers

    run_batch(config, args.seg, args.case, summary_dir=args.out)


if __name__ == '__main__':  # required: nnU-Net workers re-import this module on macOS (spawn)
    main()
