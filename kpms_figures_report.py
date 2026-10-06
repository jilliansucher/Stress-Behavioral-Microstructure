"""
Syllable figures + statistics report from an already-fitted keypoint-MoSeq model.

This does NOT refit or search parameters. It loads the model you point it at,
rebuilds the moseq dataframes, and produces:

    syllable_velocity_cm_s.pdf
    syllable_angular_velocity.pdf
    syllable_velocity_by_condition.pdf          (new: t-tests, one '*')
    syllable_angular_velocity_by_condition.pdf  (new: t-tests, one '*')
    syllable_frequency.pdf
    syllable_regressions.pdf
    syllable_regressions_grouped_by_condition.pdf

Outputs
-------
    <ANALYSIS_ROOT>/figures/                 individual PDFs, overwritten each run
    <ANALYSIS_ROOT>/kpms_report_<stamp>.pdf  one report: every figure + all stats

Formatting matches the behavioral figures: grayscale with Control darker than
CNSDS, no legends, a single asterisk for p < 0.05.

Velocity and angular velocity both come straight from kpms's moseq_df.
angular_velocity is signed - negative is one turn direction - and nothing here
takes an absolute value, so if a bar is positive that is what the data says.
"""

import os
import io
import re
import sys
import datetime
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from scipy.stats import ttest_ind, pearsonr
import statsmodels.api as sm

import keypoint_moseq as kpms


# =============================================================================
#                                 CONFIGURATION
# =============================================================================

PROJECT_DIR = r'D:\Barrier_testing_day1_videos\KPMS'

# The fitted model to analyse: a SUBFOLDER of PROJECT_DIR containing
# results.h5. Set to None to auto-detect the most recently modified one.
#
# NOTE this is a grid point from the parameter search (K=6, kappa=1e3), not
# necessarily the model the search selected as best. If there are other
# search-* folders, change this to the one you want - the chosen K/kappa are
# recorded in param_search_output/chosen_model.json. K=6 means the model has
# six syllables, 0 through 5.
MODEL_NAME = 'search-K6-kappa1e+03-seed0'

# Where the outputs go.
ANALYSIS_ROOT = (r'C:\Users\Jillian.Sucher\Documents'
                 r'\Stress_microstructure_testing_day_1\analysis_output')
FIGURE_DIR = os.path.join(ANALYSIS_ROOT, 'figures')

# Supplementary table holding HR_ratio per subject (for the regressions).
ANALYSIS_DF_CSV_PATH = 'analysis_df.csv'

# --- Syllable selection ---------------------------------------------------
# Frequency floor for treating a syllable as real.
MIN_SYLLABLE_FREQUENCY = 0.005          # 0.5% of frames

# Syllable 0 is kept (kpms's reindexing makes it the most frequent, not noise).
INCLUDE_SYLLABLE_ZERO = True

# Syllables to drop regardless of how frequent they are.
EXCLUDE_SYLLABLES = [5]

# --- Syllable location plot ------------------------------------------------
# Needs pts.csv (with the Maze_Center columns) and one .avi to use as a
# background frame. If either is missing the plot is skipped with a message.
PTS_CSV_PATH = r'D:\Barrier_testing_day1_videos\pts.csv'
VIDEO_DIR = r'D:\Barrier_testing_day1_videos'
MAZE_LIKELIHOOD_THRESHOLD = 0.7
# The location plot keeps the original colour scheme and the full frame
# extent, unlike the grayscale bar figures.
LOCATION_CONDITION_COLORS = {'Control': 'orange', 'CNSDS': 'blue'}

# Sign convention for the pooled angular-velocity legend. kpms differences the
# heading, so one sign is a left turn and the other a right turn; swap these
# two strings if the convention is the other way round for your setup.
NEGATIVE_TURN_LABEL = 'Left turn (negative)'
POSITIVE_TURN_LABEL = 'Right turn (positive)'

# Pixels -> cm, same factor the notebook used.
PX_TO_CM = 38.735 / 885

# --- Figure style ---------------------------------------------------------
CONTROL_GRAY = '0.20'                   # Control is the darker condition
CNSDS_GRAY = '0.65'
CONDITION_PALETTE = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
CONDITIONS = ['Control', 'CNSDS']

SIG_STAR_FONTSIZE = 24
SIG_NS_FONTSIZE = 13

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['figure.max_open_warning'] = 0
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=[CONTROL_GRAY, CNSDS_GRAY, '0.40', '0.80'])
sns.set_palette([CONTROL_GRAY, CNSDS_GRAY, '0.40', '0.80'])


# =============================================================================
#                          OUTPUT / REPORT PLUMBING
# =============================================================================

# Nothing below runs at import time. Everything that can fail - creating
# folders, opening the report, redirecting stdout - happens inside start_run(),
# which main() calls under try/finally. Doing it at module scope meant a failure
# could leave stdout redirected and the report file empty, with no visible error.

RUN_ID = None
REPORT_PATH = None
_report = None
_log_buffer = None
_real_stdout = sys.stdout
_counts = {'figures': 0, 'text': 0}
_mark = [0]
_names_used = {}
_finalised = [False]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def console(*args, **kwargs):
    """Progress messages: terminal only, kept out of the report."""
    kwargs['file'] = _real_stdout
    print(*args, **kwargs)
    _real_stdout.flush()


def start_run():
    """Create the output folders and open the report."""
    global RUN_ID, REPORT_PATH, _report, _log_buffer

    console('Checking output locations...')
    os.makedirs(ANALYSIS_ROOT, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    probe = os.path.join(FIGURE_DIR, '_write_test.tmp')
    with open(probe, 'w') as fh:
        fh.write('ok')
    os.remove(probe)

    RUN_ID = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    suffix = 2
    while os.path.exists(os.path.join(ANALYSIS_ROOT,
                                      f'kpms_report_{RUN_ID}.pdf')):
        RUN_ID = f'{RUN_ID}_{suffix}'
        suffix += 1
    REPORT_PATH = os.path.join(ANALYSIS_ROOT, f'kpms_report_{RUN_ID}.pdf')

    console(f'  figures -> {FIGURE_DIR}')
    console(f'  report  -> {REPORT_PATH}')

    _report = PdfPages(REPORT_PATH)
    _log_buffer = io.StringIO()
    sys.stdout = _Tee(_real_stdout, _log_buffer)


def _emit_text_pages(title):
    """Flush statistics printed since the last page into the report."""
    if _report is None or _log_buffer is None:
        return
    text = _log_buffer.getvalue()[_mark[0]:]
    _mark[0] = _log_buffer.tell()
    if not text.strip():
        return
    lines = []
    for raw in text.splitlines():
        lines.extend(textwrap.wrap(raw, 108, drop_whitespace=False) or [''])
    for start in range(0, len(lines), 60):
        page = plt.figure(figsize=(8.5, 11))
        page.text(0.06, 0.96, title, fontsize=11, weight='bold', va='top')
        page.text(0.06, 0.92, '\n'.join(lines[start:start + 60]),
                  fontsize=7, family='monospace', va='top')
        _report.savefig(page)
        plt.close(page)
        _counts['text'] += 1


def save_figure(fig, stem):
    """Write the bare figure to FIGURE_DIR, then add it to the report with the
    statistics printed just before it."""
    seen = _names_used.get(stem, 0) + 1
    _names_used[stem] = seen
    name = stem if seen == 1 else f'{stem}_{seen}'
    single_path = os.path.join(FIGURE_DIR, f'{name}.pdf')

    try:
        fig.savefig(single_path, format='pdf', bbox_inches='tight')
        size = os.path.getsize(single_path)
        console(f'  saved {name}.pdf  ({size:,} bytes)')
    except Exception as exc:
        console(f'  !! FAILED to write {single_path}: '
                f'{type(exc).__name__}: {exc}')

    if _report is not None:
        _emit_text_pages(f'Output preceding: {name}.pdf')
        fig.suptitle(f'{name}.pdf', fontsize=8, color='0.45',
                     x=0.01, ha='left', y=0.995)
        _report.savefig(fig, bbox_inches='tight')
        _counts['figures'] += 1
    plt.close(fig)


def finalise():
    """Close the report. Called from a finally block so the PDF is always
    complete, even if the run fails.

    This must NOT rely on atexit: inside Spyder / IPython / Jupyter the
    interpreter stays alive after the script ends, atexit never fires, and the
    report is left at 0 bytes - which opens as a corrupted PDF.
    """
    if _finalised[0]:
        return
    _finalised[0] = True
    try:
        if _report is not None:
            _emit_text_pages('Final output')
            info = _report.infodict()
            info['Title'] = f'KPMS syllable report - run {RUN_ID}'
            info['CreationDate'] = datetime.datetime.now()
            _report.close()
    except Exception as exc:
        console(f'(problem closing the report: {type(exc).__name__}: {exc})')
    finally:
        sys.stdout = _real_stdout
        size = (os.path.getsize(REPORT_PATH)
                if REPORT_PATH and os.path.exists(REPORT_PATH) else 0)
        console(f'\nReport : {REPORT_PATH}  ({size:,} bytes)')
        console(f'Figures: {FIGURE_DIR}')
        console(f'Pages  : {_counts["figures"]} figure, {_counts["text"]} text')
        if _counts['figures'] == 0:
            console('\nNO FIGURES WERE PRODUCED - see the error above.')


# =============================================================================
#                             DATA PREPARATION
# =============================================================================

def find_model_dirs(project_dir):
    """Every subdirectory of project_dir that looks like a fitted kpms model,
    i.e. contains results.h5 (what compute_moseq_df reads) or checkpoint.h5.

    Returns a list of (name, has_results, has_checkpoint, mtime).
    """
    found = []
    if not os.path.isdir(project_dir):
        return found
    for entry in sorted(os.listdir(project_dir)):
        full = os.path.join(project_dir, entry)
        if not os.path.isdir(full):
            continue
        has_results = os.path.exists(os.path.join(full, 'results.h5'))
        has_ckpt = os.path.exists(os.path.join(full, 'checkpoint.h5'))
        if has_results or has_ckpt:
            found.append((entry, has_results, has_ckpt,
                          os.path.getmtime(full)))
    return found


def resolve_model_name(project_dir, model_name=None):
    """Return the model to analyse.

    A kpms model is a SUBDIRECTORY of the project folder containing
    results.h5. The project-level files (config.yml, index.csv, pca.p) are not
    models. If nothing is found this explains exactly what is missing rather
    than failing with an opaque error.
    """
    if model_name:
        full = os.path.join(project_dir, model_name)
        if not os.path.isdir(full):
            raise RuntimeError(
                f'MODEL_NAME is set to {model_name!r}, but '
                f'{full} is not a directory.')
        if not os.path.exists(os.path.join(full, 'results.h5')):
            console(f'WARNING: {full} has no results.h5. If this fails, run '
                    f'kpms.extract_results() for this model first.')
        return model_name

    candidates = find_model_dirs(project_dir)
    if not candidates:
        subdirs = [e for e in sorted(os.listdir(project_dir))
                   if os.path.isdir(os.path.join(project_dir, e))] \
            if os.path.isdir(project_dir) else []
        raise RuntimeError(
            'No fitted model found in\n'
            f'    {project_dir}\n\n'
            'A kpms model is a SUBFOLDER of the project directory containing '
            'results.h5.\n'
            'The project-level files (config.yml, index.csv, pca.p, '
            'pca_scree.pdf) are\nnot models - they only mean the project was '
            'set up and PCA was run.\n\n'
            f'Subfolders present: {subdirs if subdirs else "(none)"}\n\n'
            'Fix by one of:\n'
            '  * set MODEL_NAME at the top of this script to the right '
            'subfolder name;\n'
            '  * point PROJECT_DIR at the folder that actually holds the '
            'model subfolders;\n'
            '  * if a model was fitted but never exported, run\n'
            '        kpms.extract_results(model, metadata, PROJECT_DIR, '
            'MODEL_NAME)\n'
            '    to create results.h5.')

    console(f'Models found in {project_dir}:')
    for name, has_results, has_ckpt, _ in candidates:
        console(f'    {name}   results.h5={has_results}  '
                f'checkpoint.h5={has_ckpt}')

    usable = [c for c in candidates if c[1]]
    if not usable:
        raise RuntimeError(
            'Model folders exist but none contains results.h5, which is what '
            'compute_moseq_df reads.\nRun kpms.extract_results(...) for the '
            'model you want, then re-run this script.')

    usable.sort(key=lambda c: c[3])
    chosen = usable[-1][0]
    console(f'Using most recent: {chosen}')
    return chosen


def extract_metadata_from_name(name_col):
    """Pull condition / barrier / day / subject / trial out of the recording
    name. Same parsing as the parameter-search script."""
    norm = name_col.str.replace(
        r'Barrier_Testing_(\d+cm)_Day_(\d)', r'\1_barrier_day\2', regex=True)

    condition = (norm.str.extract(r'(cnsds|control)', flags=re.IGNORECASE)[0]
                 .str.lower()
                 .map({'control': 'Control', 'cnsds': 'CNSDS'}))
    barrier = norm.str.extract(r'(10|15|20)\s*cm?', flags=re.IGNORECASE)[0]
    day = norm.str.extract(r'day[_\s]?(1|2|3)', flags=re.IGNORECASE)[0]
    subject = norm.str.extract(r'(wt\d+)', flags=re.IGNORECASE)[0].str.lower()
    trial = pd.to_numeric(
        norm.str.extract(r'(\d+)[^\d]*DLC', flags=re.IGNORECASE)[0],
        errors='coerce')

    return pd.DataFrame({'name': norm, 'condition': condition,
                         'barrier': barrier, 'day': day,
                         'subject': subject, 'trial': trial})


def _filter_angle(angles, size=9, method='median'):
    """Median/gaussian filter on angles via the (cos, sin) representation.
    Same as keypoint_moseq.util.filter_angle, reimplemented so this script
    does not depend on kpms internals."""
    from scipy.ndimage import median_filter, gaussian_filter1d
    if method == 'median':
        f = lambda x: median_filter(x, size)
    else:
        f = lambda x: gaussian_filter1d(x, size, axis=0)
    return np.arctan2(f(np.sin(angles)), f(np.cos(angles)))


def compute_moseq_df_without_index(project_dir, model_name, fps=30,
                                   smooth_heading=True):
    """Rebuild the same dataframe kpms.compute_moseq_df returns, without
    touching index.csv.

    kpms looks up every recording name in index.csv to attach a 'group'
    column, and raises IndexError if a recording is missing from that file.
    We derive condition from the recording name instead and never read
    'group', so skipping the lookup loses nothing.

    The velocity and angular-velocity formulas below are copied from
    keypoint_moseq/analysis.py so the numbers match exactly.
    """
    results = kpms.load_results(project_dir, model_name)

    names, centroids, headings, velocities, ang_vels = [], [], [], [], []
    syllables, frame_indices = [], []

    for key, value in results.items():
        centroid = np.asarray(value['centroid'])
        heading = np.asarray(value['heading'])
        syllable = np.asarray(value['syllable'])
        n = centroid.shape[0]

        names.append([str(key)] * n)
        centroids.append(centroid)
        velocities.append(np.concatenate((
            [0],
            np.sqrt(np.square(np.diff(centroid, axis=0)).sum(axis=1)) * fps)))

        rec_heading = _filter_angle(heading) if smooth_heading else heading
        headings.append(rec_heading)

        smoothed = _filter_angle(rec_heading, size=3, method='gaussian')
        ang_vels.append(np.concatenate(([0], np.diff(smoothed) * fps)))

        syllables.append(syllable)
        frame_indices.append(np.arange(n))

    centroid_all = np.concatenate(centroids)
    columns = (['centroid_x', 'centroid_y'] if centroid_all.shape[1] == 2
               else ['centroid_x', 'centroid_y', 'centroid_z'])

    df = pd.DataFrame(np.concatenate(names), columns=['name'])
    df = pd.concat([df, pd.DataFrame(centroid_all, columns=columns)], axis=1)
    df['heading'] = np.concatenate(headings)
    df['angular_velocity'] = np.concatenate(ang_vels)
    df['velocity_px_s'] = np.concatenate(velocities)
    df['syllable'] = np.concatenate(syllables)
    df['frame_index'] = np.concatenate(frame_indices)
    df['group'] = 'default'
    return df


def report_index_csv_mismatch(project_dir, model_name):
    """Explain which recordings are missing from index.csv, since that is what
    makes kpms.compute_moseq_df raise IndexError."""
    index_path = os.path.join(project_dir, 'index.csv')
    if not os.path.exists(index_path):
        print(f'  index.csv not present at {index_path}')
        return
    try:
        index_data = pd.read_csv(index_path, index_col=False)
        results = kpms.load_results(project_dir, model_name)
        listed = set(index_data['name'].astype(str))
        actual = set(str(k) for k in results.keys())
        missing = sorted(actual - listed)
        extra = sorted(listed - actual)
        print(f'  index.csv lists {len(listed)} recordings; '
              f'the model has {len(actual)}.')
        if missing:
            print(f'  {len(missing)} recording(s) in the model but NOT in '
                  f'index.csv - this is what breaks kpms:')
            for name in missing[:10]:
                print(f'      {name}')
            if len(missing) > 10:
                print(f'      ... and {len(missing) - 10} more')
        if extra:
            print(f'  {len(extra)} row(s) in index.csv with no matching '
                  f'recording (harmless).')
        print('  To repair it properly, re-run kpms.generate_index(...) or the '
              'Assign Groups widget.')
    except Exception as exc:
        print(f'  (could not compare index.csv: {type(exc).__name__}: {exc})')


def build_dataframes(project_dir, model_name):
    """moseq_df (per frame), analysis_df (per frame + metadata),
    new_analysis_df (per subject x syllable x barrier usage ratio)."""
    console('Computing moseq dataframe...')
    try:
        moseq_df = kpms.compute_moseq_df(project_dir, model_name,
                                         smooth_heading=True)
    except (IndexError, KeyError) as exc:
        # kpms raises here when a recording is missing from index.csv. The
        # only thing that lookup provides is a 'group' column we never use.
        print(f'\nkpms.compute_moseq_df failed ({type(exc).__name__}: {exc}).')
        print('This is the index.csv group lookup, not the model data.')
        report_index_csv_mismatch(project_dir, model_name)
        print('Rebuilding the dataframe directly from the model results '
              'instead - same formulas, no index.csv.\n')
        console('  index.csv lookup failed; using the direct path instead.')
        moseq_df = compute_moseq_df_without_index(project_dir, model_name)

    # Keep the original recording name: extract_metadata_from_name rewrites
    # 'name' into a normalised form, but pts.csv keys off the original, so the
    # location plot needs both.
    moseq_df['name_original'] = moseq_df['name'].values

    meta = extract_metadata_from_name(moseq_df['name'])
    for column in ['name', 'condition', 'barrier', 'day', 'subject', 'trial']:
        moseq_df[column] = meta[column].values

    # Velocity in cm/s, straight from kpms's pixel velocity.
    moseq_df['velocity_cm_s'] = moseq_df['velocity_px_s'] * PX_TO_CM

    analysis_df = (
        moseq_df[['subject', 'trial', 'syllable',
                  'condition', 'barrier', 'day']]
        .dropna(subset=['subject', 'condition', 'barrier', 'day'])
        .copy()
    )

    syll_counts = (analysis_df
                   .groupby(['subject', 'barrier', 'day', 'syllable'])
                   .size().reset_index(name='count'))
    total_counts = (analysis_df
                    .groupby(['subject', 'barrier', 'day'])
                    .size().reset_index(name='total'))
    merged = pd.merge(syll_counts, total_counts,
                      on=['subject', 'barrier', 'day'])
    merged['ratio'] = merged['count'] / merged['total']

    avg = (merged.groupby(['subject', 'barrier', 'syllable'])['ratio']
           .mean().reset_index())
    subj_cond = analysis_df[['subject', 'condition']].drop_duplicates()

    new_analysis_df = pd.merge(avg, subj_cond, on='subject')
    new_analysis_df = new_analysis_df[
        ['syllable', 'subject', 'condition', 'barrier', 'ratio']]
    new_analysis_df['barrier'] = new_analysis_df['barrier'].astype(str) + 'cm'

    return moseq_df, analysis_df, new_analysis_df


def select_syllables(moseq_df):
    """Frequent syllables, keeping 0 and dropping anything in
    EXCLUDE_SYLLABLES."""
    counts = moseq_df['syllable'].value_counts(normalize=True)
    selected = sorted(
        int(s) for s, frac in counts.items()
        if frac >= MIN_SYLLABLE_FREQUENCY
        and int(s) not in EXCLUDE_SYLLABLES
        and (INCLUDE_SYLLABLE_ZERO or int(s) != 0)
    )
    dropped = sorted(set(int(s) for s in counts.index) - set(selected))
    print(f'Syllables plotted ({len(selected)}): {selected}')
    print(f'Syllables excluded: {dropped} '
          f'(below {MIN_SYLLABLE_FREQUENCY:.1%} of frames, '
          f'or in EXCLUDE_SYLLABLES={EXCLUDE_SYLLABLES})')
    return selected


# =============================================================================
#                                   FIGURES
# =============================================================================

def _bar_axis(ax, labels):
    ax.set_xlabel('Syllable')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    sns.despine(ax=ax)


def plot_syllable_locations(moseq_df, syllables, stem):
    """Per-syllable scatter of the mouse centroid over a representative video
    frame. Ported from plot_syllable_locations in the parameter-search script,
    Conditions keep their original colours (orange / blue) and the full frame
    extent is shown; only the axis labels differ from the original.

    Needs pts.csv (for the Maze_Center columns) and one .avi for the
    background. Skips with a message if either is missing.
    """
    try:
        import cv2
    except ImportError:
        print('(skipping syllable-location plot: cv2 not installed)')
        return

    if not os.path.exists(PTS_CSV_PATH):
        print(f'(skipping syllable-location plot: {PTS_CSV_PATH} not found)')
        return

    pts = pd.read_csv(PTS_CSV_PATH)
    pts['frame_id'] = pts.groupby('name').cumcount()

    md = moseq_df.copy()
    # pts.csv stores the ORIGINAL recording name; moseq_df['name'] has been
    # normalised by then, so join on the original or every row comes back NaN.
    key = 'name_original' if 'name_original' in md.columns else 'name'
    md = md.rename(columns={key: 'name_join'})
    md['frame_id'] = md.groupby('name_join').cumcount()

    pts = pts.merge(
        md[['name_join', 'frame_id', 'syllable', 'centroid_x', 'centroid_y',
            'condition']],
        left_on=['name', 'frame_id'], right_on=['name_join', 'frame_id'],
        how='left', suffixes=('', '_moseq'))

    matched = int(pts['centroid_x'].notna().sum())
    if matched == 0:
        print('(skipping syllable-location plot: no pts.csv rows matched the '
              'model recordings. Check that the "name" column in pts.csv uses '
              'the same recording names as the model.)')
        print(f'    pts.csv names, first 2 : '
              f'{list(pd.unique(pts["name"]))[:2]}')
        print(f'    model names, first 2   : '
              f'{list(pd.unique(md["name_join"]))[:2]}')
        return

    if 'condition' not in pts.columns:
        print('(skipping syllable-location plot: no condition column)')
        return

    if 'Maze_Center likelihood' in pts.columns:
        bad = pts['Maze_Center likelihood'] < MAZE_LIKELIHOOD_THRESHOLD
        pts.loc[bad, ['Maze_Center x', 'Maze_Center y']] = np.nan
    if 'Maze_Center x' not in pts.columns:
        print('(skipping syllable-location plot: no Maze_Center columns '
              'in pts.csv)')
        return
    mcx = pts['Maze_Center x'].mean(skipna=True)
    mcy = pts['Maze_Center y'].mean(skipna=True)

    pts['nx'] = pts['centroid_x'] - mcx
    pts['ny'] = -(pts['centroid_y'] - mcy)      # image coords are y-down

    avi_files = ([f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
                 if os.path.isdir(VIDEO_DIR) else [])
    if not avi_files:
        print(f'(skipping syllable-location plot: no .avi in {VIDEO_DIR})')
        return
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, avi_files[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print('(skipping syllable-location plot: could not read a frame)')
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fh, fw = frame.shape[:2]
    extent = [-mcx, fw - mcx, -(fh - mcy), mcy]

    print(f'\nSyllable locations: maze centre at '
          f'({mcx:.1f}, {mcy:.1f}) px; {len(pts.dropna(subset=["nx"])):,} '
          f'frames with a centroid.')

    n = len(syllables)
    ncols = min(5, max(1, n))
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.9 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for ax, syl in zip(axes.flatten(), syllables):
        sub = pts[pts['syllable'] == syl]
        ax.imshow(frame, extent=extent, alpha=0.35, zorder=0, aspect='auto')
        for cond in CONDITIONS:
            csub = sub[sub['condition'].astype(str).str.lower()
                       == cond.lower()]
            if not csub.empty:
                ax.scatter(csub['nx'], csub['ny'], s=0.6, alpha=0.45,
                           color=LOCATION_CONDITION_COLORS[cond],
                           edgecolors='none', zorder=2)
        ax.set_title(f'Syllable {syl}', fontsize=12)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel('X')
        ax.tick_params(labelsize=8)

    axes.flatten()[0].set_ylabel('Y')

    # One legend for the whole figure rather than one per panel. Proxy handles
    # are used because the points are drawn at s=0.6, which is far too small to
    # read in a legend key.
    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=7,
                          color=LOCATION_CONDITION_COLORS[cond], label=cond)
               for cond in CONDITIONS]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 1.02))

    for ax in axes.flatten()[n:]:
        fig.delaxes(ax)

    sns.despine(fig=fig)
    plt.tight_layout()
    save_figure(fig, stem)


def plot_metric_per_syllable(moseq_df, syllables, column, ylabel, stem,
                             signed_legend=False):
    """Pooled across conditions: mean of `column` per syllable."""
    df = moseq_df[moseq_df['syllable'].isin(syllables)]
    stats = (df.groupby('syllable')[column]
             .agg(['mean', 'sem', 'median', 'std', 'size'])
             .reindex(syllables))

    print(f'\n{"=" * 74}\n{ylabel.upper()} PER SYLLABLE (all conditions pooled)'
          f'\n{"=" * 74}')
    print(f'{"syllable":>9} {"mean":>10} {"sem":>10} {"median":>10} '
          f'{"std":>10} {"n frames":>10}')
    for syl, row in stats.iterrows():
        print(f'{syl:>9} {row["mean"]:>10.4f} {row["sem"]:>10.4f} '
              f'{row["median"]:>10.4f} {row["std"]:>10.4f} '
              f'{int(row["size"]):>10}')

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x_pos = np.arange(len(stats))
    values = stats['mean'].values

    if signed_legend:
        # Shade each bar by the direction of the turn and label the two, so
        # the sign convention is readable off the figure itself.
        colors = [CNSDS_GRAY if v < 0 else CONTROL_GRAY for v in values]
        ax.bar(x_pos, values, yerr=stats['sem'].values,
               color=colors, edgecolor='black', linewidth=0.6,
               ecolor='black', capsize=3, width=0.7)
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=CONTROL_GRAY,
                          edgecolor='black', linewidth=0.6,
                          label=POSITIVE_TURN_LABEL),
            plt.Rectangle((0, 0), 1, 1, facecolor=CNSDS_GRAY,
                          edgecolor='black', linewidth=0.6,
                          label=NEGATIVE_TURN_LABEL),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=9,
                  loc='best')
    else:
        ax.bar(x_pos, values, yerr=stats['sem'].values,
               color=CONTROL_GRAY, ecolor='black', capsize=3, width=0.7)

    ax.set_ylabel(ylabel)
    ax.axhline(0, color='black', linewidth=0.8)
    _bar_axis(ax, stats.index)
    plt.tight_layout()
    save_figure(fig, stem)


def plot_metric_by_condition(moseq_df, syllables, column, ylabel, stem):
    """Control vs CNSDS per syllable, Welch t-test, one '*' for p < 0.05.

    Each subject contributes one value per syllable, so the t-test compares
    animals rather than frames. Testing frames would treat tens of thousands
    of correlated samples as independent and make everything significant.
    """
    df = moseq_df[moseq_df['syllable'].isin(syllables)].dropna(
        subset=['subject', 'condition'])

    per_subject = (df.groupby(['condition', 'subject', 'syllable'])[column]
                   .mean().reset_index())

    print(f'\n{"=" * 74}\n{ylabel.upper()} BY CONDITION'
          f'\n{"=" * 74}')
    print('Welch t-test on per-subject means; * marks p < 0.05 (uncorrected).')
    print(f'{"syllable":>9} {"Control":>10} {"CNSDS":>10} {"t":>8} '
          f'{"p":>10} {"n_ctrl":>7} {"n_cnsds":>8}  sig')

    means = {c: [] for c in CONDITIONS}
    sems = {c: [] for c in CONDITIONS}
    pvals = []

    for syl in syllables:
        sub = per_subject[per_subject['syllable'] == syl]
        vals = {}
        for cond in CONDITIONS:
            v = sub.loc[sub['condition'] == cond, column]
            vals[cond] = v
            means[cond].append(v.mean() if len(v) else np.nan)
            sems[cond].append(v.sem() if len(v) > 1 else 0.0)

        if len(vals['Control']) > 1 and len(vals['CNSDS']) > 1:
            t, p = ttest_ind(vals['Control'], vals['CNSDS'], equal_var=False)
        else:
            t, p = np.nan, np.nan
        pvals.append(p)

        print(f'{syl:>9} {means["Control"][-1]:>10.4f} '
              f'{means["CNSDS"][-1]:>10.4f} '
              f'{t:>8.3f} {p:>10.4g} '
              f'{len(vals["Control"]):>7} {len(vals["CNSDS"]):>8}  '
              f'{"*" if (p == p and p < 0.05) else ""}')

    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(syllables))
    width = 0.38

    for i, cond in enumerate(CONDITIONS):
        offset = -width / 2 if i == 0 else width / 2
        ax.bar(x + offset, means[cond], width=width,
               yerr=sems[cond], capsize=3,
               facecolor=CONDITION_PALETTE[cond], edgecolor='black',
               linewidth=0.6, ecolor='black')

    # Significance stars, placed above whichever bar is taller.
    span = np.nanmax([np.nanmax(np.abs(means[c])) for c in CONDITIONS])
    pad = (span if span == span and span > 0 else 1.0) * 0.10
    for i, p in enumerate(pvals):
        if p == p and p < 0.05:
            top = max(means['Control'][i] + sems['Control'][i],
                      means['CNSDS'][i] + sems['CNSDS'][i])
            ax.text(x[i], top + pad * 0.25, '*', ha='center', va='bottom',
                    fontsize=SIG_STAR_FONTSIZE)

    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=CONDITION_PALETTE[cond],
                             edgecolor='black', linewidth=0.6, label=cond)
               for cond in CONDITIONS]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc='best')

    ax.set_ylabel(ylabel)
    ax.axhline(0, color='black', linewidth=0.8)
    _bar_axis(ax, syllables)
    plt.tight_layout()
    save_figure(fig, stem)


def plot_syllable_frequency_by_condition(new_analysis_df, syllables, stem):
    """Per-barrier usage ratio, Control vs CNSDS, with Bonferroni-corrected
    Welch t-tests and a single asterisk."""
    barriers = sorted(new_analysis_df['barrier'].dropna().unique())
    if not barriers:
        print('(no barriers available; skipping frequency figure)')
        return

    fig, axes = plt.subplots(1, len(barriers), figsize=(13, 3.6),
                             sharey=True, squeeze=False)
    axes = axes[0]

    print(f'\n{"=" * 74}\nSYLLABLE FREQUENCY BY CONDITION'
          f'\n{"=" * 74}')
    print('Welch t-test per barrier x syllable, Bonferroni-corrected within '
          'each barrier.')
    print(f'{"barrier":>8} {"syllable":>9} {"Control":>9} {"CNSDS":>9} '
          f'{"t":>8} {"p":>10} {"p_bonf":>10} {"n_c":>5} {"n_s":>5}  sig')

    for ax, barrier in zip(axes, barriers):
        sub = new_analysis_df[new_analysis_df['barrier'] == barrier]
        present = [s for s in syllables if s in set(sub['syllable'])]
        ntests = max(len(present), 1)

        for i, syl in enumerate(present):
            ssub = sub[sub['syllable'] == syl]
            means, sems, vals = {}, {}, {}
            for j, cond in enumerate(CONDITIONS):
                v = ssub.loc[ssub['condition'] == cond, 'ratio']
                vals[cond] = v
                means[cond] = v.mean() if len(v) else 0.0
                sems[cond] = v.sem() if len(v) > 1 else 0.0
                offset = -0.2 if j == 0 else 0.2
                jitter = np.random.normal(0, 0.03, size=len(v))
                ax.scatter(i + offset + jitter, v, s=8, alpha=0.65,
                           color=CONDITION_PALETTE[cond], zorder=3,
                           edgecolors='none')
                ax.bar(i + offset, means[cond], yerr=sems[cond], width=0.35,
                       facecolor='white', edgecolor=CONDITION_PALETTE[cond],
                       linewidth=1.4, capsize=3, ecolor='black', zorder=2)

            if len(vals['Control']) > 1 and len(vals['CNSDS']) > 1:
                t, p = ttest_ind(vals['Control'], vals['CNSDS'],
                                 equal_var=False)
                p_bonf = min(p * ntests, 1.0)
            else:
                t, p, p_bonf = np.nan, np.nan, np.nan

            print(f'{str(barrier):>8} {syl:>9} {means["Control"]:>9.4f} '
                  f'{means["CNSDS"]:>9.4f} {t:>8.3f} {p:>10.4g} '
                  f'{p_bonf:>10.4g} {len(vals["Control"]):>5} '
                  f'{len(vals["CNSDS"]):>5}  '
                  f'{"*" if (p_bonf == p_bonf and p_bonf < 0.05) else ""}')

            if p_bonf == p_bonf and p_bonf < 0.05:
                top = max(means['Control'] + sems['Control'],
                          means['CNSDS'] + sems['CNSDS'])
                ax.text(i, top * 1.05, '*', ha='center', va='bottom',
                        fontsize=SIG_STAR_FONTSIZE)

        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels(present, rotation=45)
        ax.set_xlabel('Syllable')
        ax.set_title(str(barrier), fontsize=12)

    axes[0].set_ylabel('Fraction of total syllables')
    sns.despine(fig=fig)
    plt.tight_layout()
    save_figure(fig, stem)


def run_regressions(new_analysis_df, syllables, stem_all, stem_split):
    """R^2 of syllable usage against HR_ratio, pooled and per condition."""
    if not os.path.exists(ANALYSIS_DF_CSV_PATH):
        print(f'\n(skipping HR_ratio regressions: {ANALYSIS_DF_CSV_PATH} '
              f'not found)')
        return
    ext = pd.read_csv(ANALYSIS_DF_CSV_PATH)
    if 'HR_ratio' not in ext.columns:
        print(f'\n(skipping HR_ratio regressions: no HR_ratio column in '
              f'{ANALYSIS_DF_CSV_PATH})')
        return

    ext = ext[['subject', 'HR_ratio']].copy()
    ext['subject'] = ext['subject'].astype(str).str.lower()

    filtered = new_analysis_df[new_analysis_df['syllable'].isin(syllables)]

    def fit_one(df_syll):
        m = df_syll.merge(ext, on='subject', how='inner').dropna()
        if len(m) < 3 or m['ratio'].nunique() < 2:
            return None
        X = sm.add_constant(m['ratio'])
        ols = sm.OLS(m['HR_ratio'], X).fit()
        r, rp = pearsonr(m['ratio'], m['HR_ratio'])
        return {'r2': ols.rsquared,
                'pval': ols.pvalues.get('ratio', np.nan),
                'pearson_r': r, 'pearson_p': rp, 'n_subjects': len(m)}

    # ---- pooled -----------------------------------------------------------
    rows = []
    for syl in syllables:
        info = fit_one(filtered.loc[filtered['syllable'] == syl,
                                    ['subject', 'ratio']])
        if info:
            info['syllable'] = syl
            rows.append(info)
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        print('\n(no syllable had enough subjects for a regression)')
        return

    print(f'\n{"=" * 74}\nSYLLABLE USAGE vs HR_RATIO (all subjects)'
          f'\n{"=" * 74}')
    print(f'{"syllable":>9} {"R2":>8} {"p":>10} {"pearson r":>11} '
          f'{"n":>5}  sig')
    for _, row in rdf.iterrows():
        print(f'{int(row["syllable"]):>9} {row["r2"]:>8.4f} '
              f'{row["pval"]:>10.4g} {row["pearson_r"]:>11.3f} '
              f'{int(row["n_subjects"]):>5}  '
              f'{"*" if row["pval"] < 0.05 else ""}')

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(rdf))
    ax.bar(x, rdf['r2'], color=CONTROL_GRAY, width=0.7)
    ymax = rdf['r2'].max() if len(rdf) else 0.1
    for i, p in enumerate(rdf['pval']):
        if p < 0.05:
            ax.text(x[i], rdf['r2'].iloc[i] + ymax * 0.03, '*',
                    ha='center', va='bottom', fontsize=SIG_STAR_FONTSIZE)
    ax.set_ylabel('R$^2$ against HR ratio')
    _bar_axis(ax, rdf['syllable'].astype(int).tolist())
    plt.tight_layout()
    save_figure(fig, stem_all)

    # ---- split by condition ----------------------------------------------
    rows = []
    for cond in CONDITIONS:
        cdf = filtered[filtered['condition'] == cond]
        for syl in syllables:
            info = fit_one(cdf.loc[cdf['syllable'] == syl,
                                   ['subject', 'ratio']])
            if info:
                info['syllable'] = syl
                info['condition'] = cond
                rows.append(info)
    rdf2 = pd.DataFrame(rows)
    if rdf2.empty:
        print('\n(not enough subjects per condition for split regressions)')
        return

    print(f'\n{"=" * 74}\nSYLLABLE USAGE vs HR_RATIO (split by condition)'
          f'\n{"=" * 74}')
    print(f'{"condition":>10} {"syllable":>9} {"R2":>8} {"p":>10} '
          f'{"pearson r":>11} {"n":>5}  sig')
    for _, row in rdf2.iterrows():
        print(f'{row["condition"]:>10} {int(row["syllable"]):>9} '
              f'{row["r2"]:>8.4f} {row["pval"]:>10.4g} '
              f'{row["pearson_r"]:>11.3f} {int(row["n_subjects"]):>5}  '
              f'{"*" if row["pval"] < 0.05 else ""}')

    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(syllables))
    width = 0.38
    ymax2 = rdf2['r2'].max()
    for i, cond in enumerate(CONDITIONS):
        sub = rdf2[rdf2['condition'] == cond]
        r2v = [float(sub.loc[sub['syllable'] == s, 'r2'].iloc[0])
               if (sub['syllable'] == s).any() else 0.0 for s in syllables]
        pv = [float(sub.loc[sub['syllable'] == s, 'pval'].iloc[0])
              if (sub['syllable'] == s).any() else 1.0 for s in syllables]
        offset = -width / 2 if i == 0 else width / 2
        ax.bar(x + offset, r2v, width=width,
               facecolor=CONDITION_PALETTE[cond], edgecolor='black',
               linewidth=0.6)
        for j, p in enumerate(pv):
            if p < 0.05:
                ax.text(x[j] + offset, r2v[j] + ymax2 * 0.03, '*',
                        ha='center', va='bottom', fontsize=SIG_STAR_FONTSIZE)

    ax.set_ylabel('R$^2$ against HR ratio')
    _bar_axis(ax, syllables)
    plt.tight_layout()
    save_figure(fig, stem_split)


# =============================================================================
#                                     MAIN
# =============================================================================

def main():
    start_run()

    model_name = resolve_model_name(PROJECT_DIR, MODEL_NAME)

    print('=' * 74)
    print('KPMS SYLLABLE REPORT')
    print('=' * 74)
    print(f'project : {PROJECT_DIR}')
    print(f'model   : {model_name}')
    print(f'run     : {RUN_ID}')

    moseq_df, analysis_df, new_analysis_df = build_dataframes(
        PROJECT_DIR, model_name)

    print(f'\nFrames: {len(moseq_df)}   '
          f'subjects: {moseq_df["subject"].nunique()}   '
          f'syllables found: {moseq_df["syllable"].nunique()}')
    for cond in CONDITIONS:
        n = moseq_df.loc[moseq_df['condition'] == cond, 'subject'].nunique()
        print(f'  {cond}: {n} subjects')

    # Sanity check on the signed angular velocity, since this was the thing
    # that looked wrong before.
    av = moseq_df['angular_velocity']
    print(f'\nangular_velocity: min={av.min():.3f}  max={av.max():.3f}  '
          f'fraction negative={(av < 0).mean():.3f}')
    spikes = int((av.abs() > 100).sum())
    if spikes:
        print(f'WARNING: {spikes} frames have |angular velocity| > 100 rad/s. '
              f'These are heading-wraparound artifacts (kpms differences a '
              f'wrapped angle), and they can distort the means below.')

    syllables = select_syllables(moseq_df)

    plot_syllable_locations(moseq_df, syllables, 'syllable_location')

    plot_metric_per_syllable(
        moseq_df, syllables, 'velocity_cm_s',
        'Velocity (cm/s)', 'syllable_velocity_cm_s')

    plot_metric_per_syllable(
        moseq_df, syllables, 'angular_velocity',
        'Angular velocity (rad/s)', 'syllable_angular_velocity',
        signed_legend=True)

    plot_metric_by_condition(
        moseq_df, syllables, 'velocity_cm_s',
        'Velocity (cm/s)', 'syllable_velocity_by_condition')

    plot_metric_by_condition(
        moseq_df, syllables, 'angular_velocity',
        'Angular velocity (rad/s)', 'syllable_angular_velocity_by_condition')

    plot_syllable_frequency_by_condition(
        new_analysis_df, syllables, 'syllable_frequency')

    run_regressions(new_analysis_df, syllables,
                    'syllable_regressions',
                    'syllable_regressions_grouped_by_condition')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Put the traceback in the report too, so a failed run still leaves a
        # readable file rather than an empty one.
        import traceback
        detail = traceback.format_exc()
        try:
            print('\n' + '=' * 74)
            print('RUN FAILED')
            print('=' * 74)
            print(detail)
        except Exception:
            pass
        console(detail)
        raise
    finally:
        finalise()
