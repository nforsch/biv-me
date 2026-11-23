"""Unattended biv-me batch run: preprocessing + fitting per patient, with a QC summary
(view-classification confidence, segmentation confidence, guide points, fit residuals)
rewritten after every patient.

Usage (from src/bivme, like main.py):
    python -m bivme.batch -config configs/config.toml [-seg 3d|2d] [-summary <dir>]

-seg picks the nnU-Net segmentation models. Processing, guide point and fitting folders get a _3d/_2d suffix
so both runs coexist; the states folder (view predictions) is shared, so whichever run goes second reuses the
first run's view classification. biv-me_<batch_ID>_compare.md shows both runs' status side by side.
"""
import os, sys, time, argparse, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tomli
from loguru import logger

from bivme.main import run_preprocessing, run_fitting, validate_config_preprocessing, validate_config_fitting, validate_config_misc

VIEWS = ['SAX', '2ch', '3ch', '4ch', 'RVOT']
SEGS = ['3d', '2d']
VALVES = ['MITRAL_VALVE', 'TRICUSPID_VALVE', 'AORTA_VALVE', 'PULMONARY_VALVE']
BATCH_DEFAULTS = {'segmentation_confidence': True,
                  'view_confidence_min': 0.66, 'segmentation_confidence_min': 0.85}


def _fresh(path, since):
    """True if path exists and was written during this run (stale files from earlier runs are ignored)."""
    return os.path.exists(path) and os.path.getmtime(path) >= since


def view_qc(csv_path, conf_min):
    df = pd.read_csv(csv_path)
    df = df[df['Predicted View'].isin(VIEWS)]
    per_view = df.groupby('Predicted View')['Confidence'].min()
    low = df[df['Confidence'] < conf_min]
    return {
        'views_missing': ' '.join(v for v in VIEWS if v not in per_view.index),
        'view_conf_min': ', '.join(f'{v} {per_view[v]:.2f}' for v in VIEWS if v in per_view.index),
        'view_series_n': len(df),
        'view_low_conf': ', '.join(f'{r["Predicted View"]}#{r["Series Number"]} {r["Confidence"]:.2f}' for _, r in low.iterrows()),
    }


def segmentation_qc(csv_path, conf_min):
    df = pd.read_csv(csv_path)
    per_view = df.groupby('View').agg(mean=('Mean confidence', 'mean'), worst=('Mean confidence', 'min'),
                                      uncertain=('Uncertain fraction', 'mean'))
    per_view = per_view.reindex([v for v in VIEWS if v in per_view.index])
    return {
        'seg_conf_mean': ', '.join(f'{v} {r["mean"]:.2f}' for v, r in per_view.iterrows()),
        'seg_conf_worst_slice': ', '.join(f'{v} {r["worst"]:.2f}' for v, r in per_view.iterrows()),
        'seg_uncertain_frac': ', '.join(f'{v} {r["uncertain"]:.3f}' for v, r in per_view.iterrows()),
        'seg_low_views': ' '.join(v for v, r in per_view.iterrows() if r['worst'] < conf_min),
    }


def guidepoint_qc(gp_dir):
    files = sorted(Path(gp_dir).glob('GPFile_*.txt'))
    if not files:
        return {'gp_frames': 0, 'valves_missing': ''}
    labels = set(pd.read_csv(files[0], sep='\t')['contour type'])
    return {'gp_frames': len(files), 'valves_missing': ' '.join(v for v in VALVES if v not in labels)}


def fitting_qc(residuals_path, factor):
    res = pd.read_csv(residuals_path)['Residual'].to_numpy(dtype=float)
    ok = res[res >= 0]
    med = np.median(ok) if ok.size else np.nan
    return {
        'fit_frames_failed': int((res < 0).sum()),
        'fit_residual_mean': round(float(ok.mean()), 3) if ok.size else np.nan,
        'fit_residual_max': round(float(ok.max()), 3) if ok.size else np.nan,
        'fit_frames_high': int((ok > factor * med).sum()) if ok.size else 0,
    }


def assess(row, cfg):
    """Fill status/notes from the QC columns. FAIL = no usable model; WARN = model exists but check it."""
    if row.get('status') == 'FAIL':
        return row
    fails, warns = [], []
    if not row.get('view_series_n'):
        fails.append('view selection failed')
    if not row.get('gp_frames'):
        fails.append('no guide points')
    if 'fit_residual_mean' not in row or pd.isna(row['fit_residual_mean']):
        fails.append('fitting failed')
    if row.get('views_missing'):
        warns.append(f'missing views: {row["views_missing"]}')
    if row.get('view_low_conf'):
        warns.append(f'low view confidence (<{cfg["view_confidence_min"]})')
    if row.get('seg_low_views'):
        warns.append(f'low segmentation confidence (<{cfg["segmentation_confidence_min"]}): {row["seg_low_views"]}')
    if row.get('valves_missing'):
        warns.append(f'missing valve points: {row["valves_missing"]}')
    if row.get('fit_frames_failed'):
        warns.append(f'{row["fit_frames_failed"]} frame(s) failed to fit')
    if row.get('fit_frames_high'):
        warns.append(f'{row["fit_frames_high"]} high-residual frame(s)')
    row['status'] = 'FAIL' if fails else ('WARN' if warns else 'OK')
    row['notes'] = '; '.join(fails + warns)
    return row


def run_case(case, config, cfg):
    t0 = time.time()
    row = {'patient': case, 'started': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
    batch_id = config['input_pp']['batch_ID']
    states_csv = os.path.join(config['input_pp']['states'], batch_id, case, config['input_pp']['analyst_id'], 'view_predictions.csv')
    # Reuse view predictions from the other segmentation run, so both approaches segment the same views
    config['view-selection']['option'] = 'load' if os.path.exists(states_csv) else cfg['view_option']
    row['views_loaded'] = config['view-selection']['option'] == 'load'
    # per-case copy, rewritten on every run ('load' leaves the states copy untouched, so it would look stale)
    view_csv = os.path.join(config['input_pp']['processing'], batch_id, case, 'view-classification', 'view_predictions.csv')
    seg_csv = os.path.join(config['input_pp']['processing'], batch_id, case, 'segmentations', 'segmentation_confidence.csv')
    gp_dir = os.path.join(config['output_pp']['output_directory'], batch_id, case)
    residuals = os.path.join(config['output_fitting']['output_directory'], case, 'residuals.txt')

    try:
        run_preprocessing(case, config, logger)
        _reset_logging()
        row['preprocessing_min'] = round((time.time() - t0) / 60, 1)
        if _fresh(view_csv, t0):
            row.update(view_qc(view_csv, cfg['view_confidence_min']))
        if _fresh(seg_csv, t0):
            row.update(segmentation_qc(seg_csv, cfg['segmentation_confidence_min']))
        row.update(guidepoint_qc(gp_dir))
        if row['gp_frames']:
            t1 = time.time()
            run_fitting(case, config, logger)
            _reset_logging()
            row['fitting_min'] = round((time.time() - t1) / 60, 1)
            if _fresh(residuals, t1):
                row.update(fitting_qc(residuals, config['refitting']['threshold_factor']))
    except Exception as e:  # one bad patient must not stop the batch
        _reset_logging()
        logger.exception(f'{case} failed')
        row.update(status='FAIL', notes=f'{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ""}')
    row['total_min'] = round((time.time() - t0) / 60, 1)
    return assess(row, cfg)


def _reset_logging():
    # the pipeline adds per-case file sinks (and removes stderr when show_detailed_logging=false); an exception can leak them
    logger.remove()
    logger.add(sys.stderr)


MD_COLUMNS = [('patient', 'Patient'), ('status', 'Status'), ('notes', 'Notes'), ('view_conf_min', 'View conf (min)'),
              ('seg_conf_mean', 'Seg conf (mean)'), ('fit_residual_mean', 'Residual mean'), ('fit_residual_max', 'Residual max'),
              ('total_min', 'Minutes')]


def write_summary(rows, csv_path, md_path, title):
    df = pd.DataFrame(rows)
    tmp = csv_path.with_suffix('.csv.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)

    counts = df['status'].value_counts()
    lines = [f'# {title}', '',
             f'Updated {datetime.datetime.now():%Y-%m-%d %H:%M} — {len(df)} done: '
             + ', '.join(f'{counts.get(s, 0)} {s}' for s in ['OK', 'WARN', 'FAIL']), '',
             '| ' + ' | '.join(h for _, h in MD_COLUMNS) + ' |',
             '|' + '---|' * len(MD_COLUMNS)]
    for r in rows:
        cells = ['' if pd.isna(r.get(k, '')) else str(r.get(k, '')) for k, _ in MD_COLUMNS]
        lines.append('| ' + ' | '.join(c.replace('|', '/') for c in cells) + ' |')
    lines += ['', f'Full metrics: `{csv_path}`']
    tmp = md_path.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines) + '\n')
    os.replace(tmp, md_path)


def write_compare(summary_dir, batch_id):
    """Side-by-side status of the 3D and 2D runs, from whichever per-run CSVs exist."""
    runs = {seg: pd.read_csv(summary_dir / f'biv-me_{batch_id}_{seg}.csv', dtype={'patient': str}).set_index('patient')
            for seg in SEGS if (summary_dir / f'biv-me_{batch_id}_{seg}.csv').exists()}
    if not runs:
        return
    cols = [('status', 'Status'), ('fit_residual_mean', 'Residual mean'), ('seg_conf_mean', 'Seg conf (mean)'), ('notes', 'Notes')]
    df = pd.concat({seg: r.reindex(columns=[k for k, _ in cols]) for seg, r in runs.items()}, axis=1).sort_index()
    header = ['Patient'] + [f'{h} {seg.upper()}' for k, h in cols for seg in runs]
    lines = [f'# biv-me batch {batch_id}: 3D vs 2D segmentation', '', f'Updated {datetime.datetime.now():%Y-%m-%d %H:%M}', '']
    for seg in runs:
        counts = df[(seg, 'status')].value_counts()
        lines.append(f'- {seg.upper()}: {df[(seg, "status")].notna().sum()} done: ' + ', '.join(f'{counts.get(s, 0)} {s}' for s in ['OK', 'WARN', 'FAIL']))
    if len(runs) == 2:
        lines.append(f'- OK in both: {int(((df[("3d", "status")] == "OK") & (df[("2d", "status")] == "OK")).sum())}')
    lines += ['', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for patient, r in df.iterrows():
        cells = [str(patient)] + ['' if pd.isna(r[(seg, k)]) else str(r[(seg, k)]) for k, _ in cols for seg in runs]
        lines.append('| ' + ' | '.join(c.replace('|', '/') for c in cells) + ' |')
    md_path = summary_dir / f'biv-me_{batch_id}_compare.md'
    tmp = md_path.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines) + '\n')
    os.replace(tmp, md_path)


def main():
    parser = argparse.ArgumentParser(description='Run biv-me preprocessing + fitting on every case, with a QC summary')
    parser.add_argument('-config', '--config_file', default='configs/config.toml')
    parser.add_argument('-seg', '--segmentation', default='3d', choices=SEGS, help='nnU-Net segmentation models (default: 3d)')
    parser.add_argument('-case', '--case_name', default=None, help='Only run this case')
    parser.add_argument('-summary', '--summary_dir', default=None,
                        help='Folder for biv-me_<batch_ID>_<seg>.md/.csv (default: output_fitting output_directory)')
    args = parser.parse_args()

    with open(args.config_file, 'rb') as fp:
        config = tomli.load(fp)
    run_batch(config, args.segmentation, args.case_name, args.summary_dir)


def run_batch(config, seg='3d', case_name=None, summary_dir=None):
    cfg = {**BATCH_DEFAULTS, **config.get('batch', {}), 'seg': seg, 'view_option': config['view-selection']['option']}
    config['batch'] = cfg
    # Per-approach folders (preprocessing deletes these per case); states stays shared for view reuse
    summary_dir = Path(summary_dir or config['output_fitting']['output_directory']).expanduser()
    for section, key in [('input_pp', 'processing'), ('output_pp', 'output_directory'), ('output_fitting', 'output_directory')]:
        config[section][key] = str(Path(config[section][key]).expanduser()) + f'_{seg}'

    validate_config_misc(config, logger)
    validate_config_preprocessing(config, logger)
    if config['view-selection']['correct_mode'] != 'automatic':
        logger.warning(f'correct_mode "{config["view-selection"]["correct_mode"]}" would open the correction GUI; using "automatic" for the batch.')
        config['view-selection']['correct_mode'] = 'automatic'
    # Failed cases are re-run, so their partial fitting output must be replaced
    config['output_fitting']['overwrite'] = True
    config['input_fitting']['gp_directory'] = os.path.join(config['output_pp']['output_directory'], config['input_pp']['batch_ID'])
    os.makedirs(config['input_fitting']['gp_directory'], exist_ok=True)
    validate_config_fitting(config, logger)

    src = config['input_pp']['source']
    cases = sorted(c for c in os.listdir(src) if os.path.isdir(os.path.join(src, c)))
    if case_name:
        cases = [c for c in cases if c == case_name]

    batch_id = config['input_pp']['batch_ID']
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f'biv-me_{batch_id}_{seg}.csv'
    md_path = summary_dir / f'biv-me_{batch_id}_{seg}.md'

    # Resume: skip finished cases (OK/WARN), re-run failed ones. FAIL rows stay until their rerun replaces them,
    # so a rerun cut short still lists every case attempted so far.
    rows = {r['patient']: r for r in pd.read_csv(csv_path, dtype={'patient': str}).to_dict('records')} if csv_path.exists() else {}
    done = {p for p, r in rows.items() if r['status'] in ('OK', 'WARN')}
    todo = [c for c in cases if c not in done]
    logger.info(f'{len(cases)} case(s), {len(done)} already done, {len(todo)} to run. Summary: {md_path}')

    for i, case in enumerate(todo):
        logger.info(f'=== Case {i + 1}/{len(todo)} ({seg}): {case}')
        row = run_case(case, config, cfg)
        rows[case] = row  # replaces an earlier FAIL row in place
        write_summary(list(rows.values()), csv_path, md_path, f'biv-me batch {batch_id} ({seg.upper()} segmentation)')
        write_compare(summary_dir, batch_id)
        logger.info(f'{case}: {row["status"]} {row.get("notes", "")}')


if __name__ == '__main__':
    main()
