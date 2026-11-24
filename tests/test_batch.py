import os
import numpy as np
import pandas as pd

from bivme.batch import view_qc, guidepoint_qc, fitting_qc, segmentation_qc, assess, write_compare, BATCH_DEFAULTS
from bivme.preprocessing.dicom.segment_views import summarise_probabilities

DEMO = os.path.join(os.path.dirname(__file__), '..', 'demo')


def test_qc_and_status(tmp_path):
    views = view_qc(os.path.join(DEMO, 'states', 'example', 'patient1', 'analyst1', 'view_predictions.csv'), 0.66)
    assert views['view_series_n'] > 0 and 'SAX' in views['view_conf_min']

    gps = guidepoint_qc(os.path.join(DEMO, 'guidepoints', 'example', 'patient1'))
    assert gps['gp_frames'] > 0 and gps['valves_missing'] == ''

    res = tmp_path / 'residuals.txt'
    res.write_text('Frame,Residual\n0,1.0\n1,1.0\n2,5.0\n3,-1\n')
    fit = fitting_qc(res, 1.2)
    assert fit == {'fit_frames_failed': 1, 'fit_residual_mean': 2.333, 'fit_residual_max': 5.0, 'fit_frames_high': 1}

    row = assess({**views, **gps, **fit}, BATCH_DEFAULTS)
    assert row['status'] == 'WARN' and 'high-residual' in row['notes']
    assert assess({**views, **gps}, BATCH_DEFAULTS)['status'] == 'FAIL'  # no fit -> FAIL


def test_segmentation_confidence(tmp_path):
    probs = np.zeros((2, 1, 2, 2), dtype=np.float32)  # 2 classes, 4 voxels
    probs[1] = [[[0.9, 0.55], [0.1, 0.2]]]  # two foreground voxels: 0.9 and 0.55
    probs[0] = 1 - probs[1]
    np.savez_compressed(tmp_path / 'SAX_3d_1.npz', probabilities=probs)
    rows = summarise_probabilities(str(tmp_path), 'SAX')
    assert len(rows) == 1 and rows[0]['Foreground voxels'] == 2
    assert np.isclose(rows[0]['Mean confidence'], 0.725) and rows[0]['Uncertain fraction'] == 0.5
    assert not list(tmp_path.glob('*.npz'))  # deleted after use

    pd.DataFrame(rows).to_csv(tmp_path / 'seg.csv', index=False)
    assert segmentation_qc(tmp_path / 'seg.csv', 0.85)['seg_low_views'] == 'SAX'


def test_compare_3d_vs_2d(tmp_path):
    pd.DataFrame({'patient': ['007', '12'], 'status': ['OK', 'WARN'], 'fit_residual_mean': [1.1, 2.0]}).to_csv(tmp_path / 'biv-me_b_3d.csv', index=False)
    write_compare(tmp_path, 'b')  # only 3D done so far
    assert 'Status 2D' not in (tmp_path / 'biv-me_b_compare.md').read_text()

    pd.DataFrame({'patient': ['007'], 'status': ['OK'], 'notes': ['']}).to_csv(tmp_path / 'biv-me_b_2d.csv', index=False)
    write_compare(tmp_path, 'b')
    md = (tmp_path / 'biv-me_b_compare.md').read_text()
    assert '| Patient | Status 3D | Status 2D |' in md and 'OK in both: 1' in md
    assert '| 007 | OK | OK | 1.1 |' in md and '| 12 | WARN |  | 2.0 |' in md  # leading zero kept, 2D pending


def test_tightened_rules():
    cfg = {**BATCH_DEFAULTS, 'ignore_missing': ['RVOT', 'PULMONARY_VALVE']}
    base = {'view_series_n': 14, 'gp_frames': 30, 'fit_residual_mean': 1.8, 'crashed': False}
    # missing RVOT/PV ignored; 0.43 view conf no longer flagged, 0.05 is
    row = assess({**base, 'views_missing': 'RVOT', 'valves_missing': 'PULMONARY_VALVE', 'view_low_conf': 'SAX#23 0.43'}, cfg)
    assert row['status'] == 'OK', row['notes']
    row = assess({**base, 'view_low_conf': 'SAX#23 0.43, 4ch#18 0.05'}, cfg)
    assert row['status'] == 'WARN' and row['notes'] == 'low view confidence (<0.3): 4ch#18 0.05'
    assert assess({**base, 'views_missing': 'SAX RVOT'}, cfg)['notes'] == 'no SAX view'
    # rows read back from a summary CSV: NaN means empty; crashed rows keep their exception note
    loaded = {**base, 'views_missing': np.nan, 'seg_low_views': np.nan, 'status': 'WARN', 'notes': 'old'}
    assert assess(loaded, cfg)['status'] == 'OK'
    crashed = {'patient': '5', 'status': 'FAIL', 'crashed': np.True_, 'notes': 'RuntimeError: boom'}
    assert assess(crashed, cfg)['notes'] == 'RuntimeError: boom'
