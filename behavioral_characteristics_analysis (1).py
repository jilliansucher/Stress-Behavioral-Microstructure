"""
Behavioral characteristics analysis.

Converted from behavioral_characteristics_analysis_FINAL.ipynb and reordered so
that the script runs top to bottom with every name defined before it is used.

Cells moved during the reorder:
  * frame / x_pic / y_pic (notebook cell 94) now load before the location
    scatterplot (cell 73) that draws them.
  * The velocity calculation (cell 58) now runs before everything that reads
    velocity_df or pts['velocity_cm'].
  * `merged` (cell 78) now builds before the cells that display or aggregate it.
  * avg_decision_time (cell 88) now builds before the decision-time figures.
  * The three cells that merge extra columns into avg_duration_df and plot them
    (notebook cells 55, 56, 57, 87) moved down below cell 103, which is where
    avg_duration_df, sig_label and add_sig_bracket are first defined.

Cell removed:
  * Notebook cell 25 was a near-duplicate of cell 24 whose significance-star
    block was dedented out of the `for` loop, so it evaluated only the last
    barrier, and it wrote to the same PDF path as cell 24 - overwriting the
    correct figure with the broken one. Cell 24 is kept; cell 25 is deleted.
    It bound no name that cell 24 does not also bind.

Other edits, all marked inline with "# NOTE (conversion)":
  * Output is one timestamped report PDF (every figure as a page, preceded by
    the statistics printed for it) plus a figures_<timestamp>/ folder holding
    each figure on its own with no statistics. Nothing is ever overwritten.
  * The "Stress correlations" section (notebook cells 108-112) was deleted on
    request. Nothing later depended on it.
  * IPython magics commented out; duplicate import block removed.
  * Bare display expressions wrapped in print().
  * One intra-cell ordering fix in the start-box aggregation.
  * Imports that the notebook relied on from later cells hoisted here.
"""

import os
import sys
import io
import re
import fnmatch
import glob as glob
import atexit
import datetime
import textwrap

import cv2
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import Affine2D
import seaborn as sns

import scipy as scipy
import scipy.stats as stats
from scipy.stats import ttest_ind, pearsonr, linregress

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# NOTE (conversion): the notebook imported `util`, but nothing in it is ever
# called. Made optional so a missing util.py does not stop the run.
try:
    import util
except ImportError:
    util = None
    _UTIL_MISSING = True

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 10})

# NOTE (conversion): the notebook ran "%load_ext autoreload" / "%autoreload 2"
# here. Those are IPython magics and have no effect in a plain script.

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
# Everything this script produces goes into ONE timestamped folder containing
# ONE report PDF. Nothing existing is ever overwritten, because the run ID is
# part of the name. Change OUTPUT_ROOT if you want the runs somewhere else.
OUTPUT_ROOT = os.path.join(os.getcwd(), 'analysis_output')

RUN_ID = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
_suffix = 2
while os.path.exists(os.path.join(OUTPUT_ROOT, f'analysis_report_{RUN_ID}.pdf')):
    RUN_ID = f'{RUN_ID}_{_suffix}'   # two runs in the same second
    _suffix += 1
os.makedirs(OUTPUT_ROOT, exist_ok=True)

REPORT_PATH = os.path.join(OUTPUT_ROOT, f'analysis_report_{RUN_ID}.pdf')

# Each figure is ALSO written on its own, without any statistics, into a
# fixed folder beside the reports. This one is NOT timestamped: every run
# overwrites it, so it always holds the current version of each figure.
# (The report PDFs above stay timestamped and still accumulate.)
FIGURE_DIR = os.path.join(OUTPUT_ROOT, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# There are ~20 plt.show() calls. In a script each opens a window and blocks
# until closed. Leave False to run unattended; the figures go to the report
# either way.
SHOW_PLOTS = False

# The script also dumps three intermediate data tables (analysis_df.csv,
# pts.csv, velocity_bin_stats.csv). pts.csv is the per-frame DLC export and is
# large. Off by default - the figures and stats go to the report either way.
# Set True if you ever need the underlying tables.
SAVE_INTERMEDIATE_CSVS = False

if not SHOW_PLOTS:
    matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 0

# --- grayscale figures: Control dark, CNSDS light ---------------------------
# Every condition colour in the script resolves to one of these two, so the
# contrast direction is consistent across every figure. 0.0 is black, 1.0 is
# white; lower CONTROL_GRAY or raise CNSDS_GRAY to widen the separation.
# Decision fork region, in normalized pixels relative to the tracked maze
# centre (normalized_x/y = avg_x/y - Maze_Center_x/y). Defined once here;
# avg_decision_time_secs and everything downstream of it - the decision-time
# figures and the regression - all follow from these four numbers.
DECISION_FORK_X = (-200, 200)
DECISION_FORK_Y = (-100, 150)

# One size for every significance asterisk in the file.
SIG_STAR_FONTSIZE = 24
# 'ns' reads smaller than the asterisk - a star needs the size, 'ns' does not.
SIG_NS_FONTSIZE = 13

CONTROL_GRAY = '0.20'
CNSDS_GRAY = '0.65'
CONDITION_PALETTE = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}

# Anything not explicitly coloured falls back to the same greys.
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=[CONTROL_GRAY, CNSDS_GRAY, '0.40', '0.80'])
matplotlib.rcParams['image.cmap'] = 'gray'
sns.set_palette([CONTROL_GRAY, CNSDS_GRAY, '0.40', '0.80'])


# plt.bar draws yerr bars in the current cycle colour and ignores edgecolor,
# which leaves coloured error bars on otherwise grayscale figures. Default
# ecolor to the bar's own edge colour instead.
def _gray_errorbars(bar_function):
    def wrapper(*args, **kwargs):
        if kwargs.get('yerr') is not None and 'ecolor' not in kwargs:
            edge = kwargs.get('edgecolor')
            kwargs['ecolor'] = 'black' if edge in (None, 'none') else edge
        return bar_function(*args, **kwargs)
    return wrapper


plt.bar = _gray_errorbars(plt.bar)
matplotlib.axes.Axes.bar = _gray_errorbars(matplotlib.axes.Axes.bar)


# --- capture everything printed, so the stats land in the report -------------
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


_log_buffer = io.StringIO()
_real_stdout = sys.stdout
sys.stdout = _Tee(_real_stdout, _log_buffer)


def console(*args, **kwargs):
    """Progress and diagnostics: terminal only, kept out of the report."""
    kwargs['file'] = _real_stdout
    print(*args, **kwargs)

_report = PdfPages(REPORT_PATH)
_page_count = {'figures': 0, 'text': 0, 'files': 0}
_log_mark = [0]


def _emit_text_pages(title):
    """Flush stdout printed since the last page into monospace PDF pages."""
    text = _log_buffer.getvalue()[_log_mark[0]:]
    _log_mark[0] = _log_buffer.tell()
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
        _page_count['text'] += 1


# --- route every savefig() into the single report ---------------------------
_figure_names_used = {}


def _figure_path(stem):
    """Overwrite last run's file of the same name, but keep the two figures
    drawn under one name in a single run distinct (startbox_time is drawn
    twice, so it becomes startbox_time.pdf and startbox_time_2.pdf). The
    counter is per-run, not per-folder, so names stay stable across runs."""
    seen = _figure_names_used.get(stem, 0) + 1
    _figure_names_used[stem] = seen
    name = stem if seen == 1 else f'{stem}_{seen}'
    return os.path.join(FIGURE_DIR, f'{name}.pdf')


def _savefig_to_report(fname, *args, **kwargs):
    label = os.path.basename(str(fname).replace('\\', '/'))
    figure = plt.gcf()
    kwargs.pop('format', None)
    # Trim to the drawn content so nothing at the edges - the outermost x tick
    # label in particular - is cut off at the canvas boundary.
    kwargs.setdefault('bbox_inches', 'tight')

    # 1. standalone copy first, before the report's filename banner is added,
    #    so the individual file is the bare figure and nothing else.
    single_path = _figure_path(os.path.splitext(label)[0])
    figure.savefig(single_path, **kwargs)
    _page_count['files'] += 1

    # 2. then the report page, preceded by the stats printed for it.
    _emit_text_pages(f'Output preceding: {label}')
    figure.suptitle(label, fontsize=8, color='0.45', x=0.01, ha='left', y=0.995)
    _report.savefig(figure, **kwargs)
    plt.close(figure)
    _page_count['figures'] += 1
    console(f'  [report] page {_page_count["figures"] + _page_count["text"]}: '
            f'{label}  ->  {os.path.basename(single_path)}')


plt.savefig = _savefig_to_report
if not SHOW_PLOTS:
    plt.show = lambda *a, **k: None


def _finalise_report():
    """Always close the PDF, even if the script raises partway through."""
    if _report is None:
        return
    try:
        _emit_text_pages('Final output')
        info = _report.infodict()
        info['Title'] = f'Behavioral characteristics analysis - run {RUN_ID}'
        info['CreationDate'] = datetime.datetime.now()
    finally:
        _report.close()
        sys.stdout = _real_stdout
        console(f'\nReport : {REPORT_PATH}')
        console(f'Pages  : {_page_count["figures"]} figure, '
                f'{_page_count["text"]} text')
        console(f'Figures: {_page_count["files"]} standalone PDFs in '
                f'{FIGURE_DIR} (overwritten each run)')


atexit.register(_finalise_report)
console(f'Writing this run to: {REPORT_PATH}\n')


# ==========================================================================
# IMPORT AND CONCATENATE FILES
# ==========================================================================

# ---- notebook cell 2 -----------------------------------------------------------
# # a function to find files based on a pattern and a path name
# def find(pattern,path):
#     result = []
#     for root, dirs, files in os.walk(path):
#         for name in files:
#             if fnmatch.fnmatch(name,pattern):
#                 result.append(os.path.join(root,name))
#     return result

# # find all files with the string '2Jul30shuffle1' in folder D:\Barrier_testing_day1_videos
# files = find('*2Jul30shuffle1*', r'D:\Barrier_testing_day1_videos')

# # loop through and delete the files
# for file in files:
#     os.remove(file)
#     print(f"Deleted file: {file}")

# ---- notebook cell 4 -----------------------------------------------------------
summary_dir = r"C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1"
dlc_dir = r"D:\Barrier_testing_day1_videos"

# ---- notebook cell 5 -----------------------------------------------------------
# Column names for the summary data
summary_columns = [
    "date", "subject", "sex", "condition", "experiment_name",
    "HR_Arm", "arm_choice", "eat", "start_time", "choice_time", "eat_time"
]

# ---- notebook cell 6 -----------------------------------------------------------
# Function to collect specific files based on a naming pattern
def get_files_by_pattern(directory, pattern):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if pattern in file:
                file_paths.append(os.path.join(root, file))
    return file_paths

# ---- notebook cell 7 -----------------------------------------------------------
summary_dir = r"C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1"
for root, _, files in os.walk(summary_dir):
    for file in files:
        console(file)

# ---- notebook cell 8 -----------------------------------------------------------
# Get all relevant files
summary_files = get_files_by_pattern(summary_dir, "TrialData.csv")
dlc_files = get_files_by_pattern(dlc_dir, "filtered.h5")
#SIT_files = get_files_by_pattern(SIT_dir, "VideoData.csv")

# ---- notebook cell 9 -----------------------------------------------------------
# Process summary files
summary_combined = []
for summary_file in summary_files:
    summary_df = pd.read_csv(summary_file, header=None, names=summary_columns)

    # Filter rows with *Barrier_Testing* in the experiment_name column
    #summary_df = summary_df[summary_df['experiment_name'].str.contains("Barrier_Testing", na=False)]
    summary_df = summary_df[summary_df['experiment_name'].str.contains("barrier_testing", case=False, na=False)]
    
    # Add a 'trial' column based on the row index (1-based indexing for trial numbers)
    summary_df['trial'] = summary_df.index + 1
    summary_combined.append(summary_df)

# ---- notebook cell 10 ----------------------------------------------------------
# Combine all summary data
summary_df = pd.concat(summary_combined, ignore_index=True)


# ==========================================================================
# TRIAL DURATION
# ==========================================================================

# ---- notebook cell 12 ----------------------------------------------------------
#convert time stamps into seconds 
# go through summary_df, for each row find the video (chopped), load it in using open cv and get the total number of frames
#get frame rate from the video
#frames/framespersecond
#load the side video

# --------------------------------------------------
# Helper: find files recursively with a pattern
# --------------------------------------------------
def find(pattern, path):
    matches = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(root, name))
    return matches


# --------------------------------------------------
# Helper: get video filepath from a dataframe row
# --------------------------------------------------
def get_vidname_from_row(row, video_dir):

    date = str(row['date'])
    subject = str(row['subject'])
    trial = int(row['trial']) - 1

    patterns = []

    barrier = '[Bb][Aa][Rr][Rr][Ii][Ee][Rr]'
    
    for ext in ('avi', 'mp4v'):
        patterns.extend([
            date.zfill(6) + f'*{barrier}*{subject}*StartArm_{trial}*.{ext}',
            date.zfill(8) + f'*{barrier}*{subject}*StartArm_{trial}*.{ext}',
            date.zfill(6) + f'*{barrier}*StartArm_{trial}*.{ext}',
            date.zfill(8) + f'*{barrier}*StartArm_{trial}*.{ext}',
        ])

    

    for pattern in patterns:
        files = find(pattern, video_dir)
        if files:
            return files[0]   # RETURN STRING PATH ONLY

    raise FileNotFoundError(f"No video found for row {row.name}")


# --------------------------------------------------
# Ensure dataframe dtypes are safe
# --------------------------------------------------
summary_df['subject'] = summary_df['subject'].astype(str)
summary_df['date'] = summary_df['date'].astype(str)
summary_df['trial'] = summary_df['trial'].astype(int)


# --------------------------------------------------
# Main loop: compute trial duration
# --------------------------------------------------
trial_duration = []
bad_trials = []

for index, row in summary_df.iterrows():

    try:
        video_path = get_vidname_from_row(row, dlc_dir)
    except FileNotFoundError as e:
        console(e)
        trial_duration.append(np.nan)
        bad_trials.append(index)
        continue

    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        console(f"Could not open video for row {index}")
        trial_duration.append(np.nan)
        bad_trials.append(index)
        continue

    tot_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)

    if fps == 0 or np.isnan(fps):
        trial_duration.append(np.nan)
        bad_trials.append(index)
    else:
        trial_duration.append(tot_frames / fps)

    vid.release()


# --------------------------------------------------
# Save results
# --------------------------------------------------
summary_df['trial_duration'] = trial_duration

console(f"Finished. Bad trials: {len(bad_trials)}")

# ---- notebook cell 13 ----------------------------------------------------------
# row = summary_df.loc[1814]
# print(row[['date', 'subject', 'trial', 'experiment']])


# ==========================================================================
# STANDARDIZE DATA
# ==========================================================================

# ---- notebook cell 15 ----------------------------------------------------------
def extract_condition(name):
    if pd.isna(name):
        return np.nan
    
    match = re.search(r'(control|cnsds)', str(name), re.IGNORECASE)
    
    if match:
        # NOTE (conversion): was `return match.group(1).upper()`, giving
        # 'CONTROL'; everything downstream compares against 'Control'.
        # (This function is currently dead code - summary_columns always
        # defines a 'condition' column, so the guard below never fires.)
        return 'Control' if match.group(1).upper() == 'CONTROL' else 'CNSDS'
    else:
        return np.nan

# Only create column if it doesn't exist
if 'condition' not in summary_df.columns:
    summary_df['condition'] = summary_df['experiment_name'].apply(extract_condition)

# ---- notebook cell 16 ----------------------------------------------------------
def standardize_experiment_name(name):
    if pd.isna(name):
        return np.nan
    
    name = str(name).lower().strip()
    
    # Remove extra underscores
    name = re.sub(r'__+', '_', name)
    
    # Remove unexpected characters (keep letters, numbers, underscore)
    name = re.sub(r'[^a-z0-9_]', '', name)
    
    # Extract distance and day
    match = re.search(r'(10|15|20)cm.*day[_]?([1-3])', name)
    
    if match:
        distance = match.group(1)
        day = match.group(2)
        return f"{distance}cm_Barrier_Day{day}"
    else:
        return np.nan

# Apply cleaning
summary_df['experiment_name'] = summary_df['experiment_name'].apply(standardize_experiment_name)

# Keep only valid rows (Day 1–3, 10/15/20cm)
summary_df = summary_df.dropna(subset=['experiment_name'])

# ---- notebook cell 17 ----------------------------------------------------------
# Lowercase the values in the 'HR_Arm' column
summary_df['HR_Arm'] = summary_df['HR_Arm'].str.lower()

# Create the 'choice' column based on comparison
summary_df['choice'] = summary_df['arm_choice'] == summary_df['HR_Arm']

# ---- notebook cell 18 ----------------------------------------------------------
summary_df.loc[
    (summary_df['subject'].isin(['WT041', 'WT047'])) &
    (summary_df['condition'].isna()),
    'condition'
] = 'Control'

# NOTE (conversion): normalise condition spelling. Later cells run
# pd.Categorical(condition, categories=['Control', 'CNSDS']), which turns
# any other spelling ('CONTROL', 'control', stray whitespace) into NaN and
# drops those rows from the ANOVAs without warning. No-op if already clean.
def _normalise_condition(value):
    if pd.isna(value):
        return value
    text = str(value).strip().upper()
    if text == 'CONTROL':
        return 'Control'
    if text == 'CNSDS':
        return 'CNSDS'
    return value

_before = summary_df['condition'].dropna().unique().tolist()
summary_df['condition'] = summary_df['condition'].apply(_normalise_condition)
_after = summary_df['condition'].dropna().unique().tolist()
if sorted(map(str, _before)) != sorted(map(str, _after)):
    print(f"[note] condition values normalised: {_before} -> {_after}")

# ---- notebook cell 19 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# summary_df.loc[
#     summary_df['condition'].isna(),
#     ['subject', 'experiment_name', 'condition', 'trial_duration']
# ]


# ==========================================================================
# HR CHOICE
# ==========================================================================

# ---- notebook cell 21 ----------------------------------------------------------
# Ensure 'choice' is boolean
summary_df['choice'] = summary_df['choice'].astype(bool)

# Group by experiment_name and subject
grouped = summary_df.groupby(['experiment_name', 'subject'])

analysis_df = grouped['choice'].agg(
    HR_percentage=lambda x: x.sum() / len(x)
).reset_index()

# Get condition and sex info (drop duplicates to avoid merge issues)
meta_info = summary_df.drop_duplicates(subset=['experiment_name', 'subject'])[
    ['date', 'experiment_name', 'subject', 'condition', 'sex']
]

# Merge meta info into the analysis_df
analysis_df = analysis_df.merge(meta_info, on=['experiment_name', 'subject'], how='left')

# ---- notebook cell 22 ----------------------------------------------------------
# Count number of True in 'choice' per 'experiment_name' and 'subject'
true_counts = summary_df.groupby(['experiment_name', 'subject'])['choice'].sum().reset_index()

# Count the total number of rows per 'experiment_name' and 'subject' (i.e., total trials)
total_counts = summary_df.groupby(['experiment_name', 'subject']).size().reset_index(name='total_trials')

# Merge the two counts together
hr_ratio_df = pd.merge(true_counts, total_counts, on=['experiment_name', 'subject'])

# Calculate the HR ratio (True count / Total trials)
hr_ratio_df['HR_ratio'] = hr_ratio_df['choice'] / hr_ratio_df['total_trials']

# Merge HR_ratio into analysis_df based on 'experiment_name' and 'subject'
analysis_df = analysis_df.merge(hr_ratio_df, on=['experiment_name', 'subject'], how='left')

# ---- notebook cell 23 ----------------------------------------------------------
# Preprocessing & Aggregation
# --------------------------------------------------
df = analysis_df.copy()

# Extract barrier and day from experiment name
df['barrier'] = df['experiment_name'].str.extract(r'(\d+cm)')
df['day'] = df['experiment_name'].str.extract(r'(Day\d)')

# Infer condition if missing
if 'condition' not in df.columns:
    df['condition'] = df['experiment_name'].apply(
        lambda x: 'CNSDS' if 'CNSDS' in x
        else 'Control' if 'Control' in x
        else None
    )

# Ensure HR_ratio is numeric
df['HR_ratio'] = pd.to_numeric(df['HR_ratio'], errors='coerce')

# Drop invalid rows
df = df.dropna(subset=['HR_ratio', 'barrier', 'subject'])

# --------------------------------------------------
# ⭐ Create subject-level averages across Day 1–3
# --------------------------------------------------
df_avg = (
    df
    .groupby(['subject', 'condition', 'barrier'], observed=True)['HR_ratio']
    .mean()
    .reset_index()
)

# Final cleanup
df_avg = df_avg.dropna(subset=['HR_ratio'])

# ---- notebook cell 24 ----------------------------------------------------------
# ---------------------------------------------------------
# 🟦 1. Configuration
# ---------------------------------------------------------
barriers = ['10cm', '15cm', '20cm']
colors = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.3
x_locs = np.arange(len(barriers))

sns.set(style='white')
fig, ax = plt.subplots(figsize=(6, 4))

print("\n==============================")
print("BOOTSTRAP RESULTS")
print("==============================")

# ---------------------------------------------------------
# 🟧 2. Plotting Loop + Bootstrap
# ---------------------------------------------------------
for i, barrier in enumerate(barriers):
    barrier_max_y = 0

    sub_df = df_avg[df_avg['barrier'] == barrier]

    ctrl_vals = sub_df[sub_df['condition'] == 'Control']['HR_ratio'].values
    cnsds_vals = sub_df[sub_df['condition'] == 'CNSDS']['HR_ratio'].values

    p_boot = 1.0  # default

    # ---------------------------
    # Bootstrap p-value
    # ---------------------------
    if len(ctrl_vals) > 1 and len(cnsds_vals) > 1:
        boot_diffs = []

        for _ in range(5000):
            b_ctrl = np.random.choice(ctrl_vals, size=len(ctrl_vals), replace=True)
            b_cnsds = np.random.choice(cnsds_vals, size=len(cnsds_vals), replace=True)
            boot_diffs.append(np.mean(b_ctrl) - np.mean(b_cnsds))

        boot_diffs = np.array(boot_diffs)
        mean_diff = np.mean(boot_diffs)

        if mean_diff > 0:
            p_boot = np.mean(boot_diffs <= 0)
        else:
            p_boot = np.mean(boot_diffs >= 0)

    # ---------------------------
    # PRINT (always)
    # ---------------------------
    sig_label = "Significant" if p_boot < 0.05 else "Not significant"

    print(
        f"{barrier}: "
        f"{sig_label} (p={p_boot:.4f}) | "
        f"n_control={len(ctrl_vals)}, n_CNSDS={len(cnsds_vals)}"
    )

    # ---------------------------
    # Plot bars + points
    # ---------------------------
    for j, cond in enumerate(['Control', 'CNSDS']):
        pos = i + (j * bar_width) - (bar_width / 2)
        data = sub_df[sub_df['condition'] == cond]['HR_ratio']

        if data.empty:
            continue

        mean_val = data.mean()
        sem_val = data.sem()

        barrier_max_y = max(barrier_max_y, data.max(), mean_val + sem_val)

        ax.bar(
            pos,
            mean_val,
            width=bar_width,
            color='none',
            edgecolor=colors[cond],
            linewidth=3,
            label=cond if i == 0 else ""
        )

        ax.errorbar(
            pos,
            mean_val,
            yerr=sem_val,
            fmt='none',
            ecolor=colors[cond],
            capsize=5,
            lw=2
        )

        x_jitter = np.random.normal(pos, 0.04, size=len(data))
        ax.scatter(
            x_jitter,
            data,
            color=colors[cond],
            s=40,
            alpha=1.0,
            zorder=3
        )

    # ---------------------------
    # ⭐ Significance star
    # ---------------------------
    if p_boot < 0.05:
        star_y = barrier_max_y + 0.05
        ax.text(
            i,
            star_y,
            "*",
            ha='center',
            va='bottom',
            fontsize=28,
            color='black',
            fontweight='bold'
        )

# ---------------------------------------------------------
# 🟩 3. Formatting
# ---------------------------------------------------------
ax.set_xticks(x_locs)
ax.set_xticklabels(barriers, fontsize=16)
ax.set_ylabel('HR Ratio', fontsize=20)
ax.set_xlabel('Barrier', fontsize=20)
ax.set_ylim(0, 1.2)

sns.despine()
ax.legend(frameon=False, fontsize=14, loc='upper right')

plt.tight_layout()

plt.savefig(
    r'D:\Figures\Choice_across_days_Bootstrap.pdf',
    dpi=300,
    format='pdf',
    bbox_inches='tight'
)

plt.show()

# ---- notebook cell 26 ----------------------------------------------------------
# NOTE (conversion): intermediate data dump, off by default.
if SAVE_INTERMEDIATE_CSVS:
    analysis_df.to_csv(os.path.join(OUTPUT_ROOT, f"analysis_df_{RUN_ID}.csv"), index=False)

# ---- notebook cell 27 ----------------------------------------------------------
# Convert all values to string first, then strip and capitalize
summary_df['choice'] = summary_df['choice'].astype(str).str.strip().str.upper()

# Optional: replace 'NAN' strings (from actual NaNs) with np.nan
summary_df.loc[summary_df['choice'] == 'NAN', 'choice'] = np.nan

# Drop rows where trial_duration or choice is still missing
summary_df = summary_df.dropna(subset=['trial_duration', 'choice', 'subject'])

# ---- notebook cell 28 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# analysis_df

# ---- notebook cell 29 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# summary_df

# ---- notebook cell 30 ----------------------------------------------------------
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Ensure choice column is consistent
summary_df['choice'] = summary_df['choice'].str.upper().str.strip()  # 'HR' / 'LOW'
summary_df['trial_duration'] = pd.to_numeric(summary_df['trial_duration'], errors='coerce')
summary_df = summary_df.dropna(subset=['trial_duration', 'choice', 'subject'])

# Simple ANOVA: trial_duration ~ choice
model = ols('trial_duration ~ C(choice)', data=summary_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA for trial_duration by choice (summary_df):")
print(anova_table)

# Tukey post hoc if you had more than 2 choice types
tukey = pairwise_tukeyhsd(endog=summary_df['trial_duration'],
                          groups=summary_df['choice'],
                          alpha=0.05)
print("\nTukey HSD post hoc results:")
print(tukey.summary())

# ---- notebook cell 31 ----------------------------------------------------------
# List of dataframes to update
dfs = [analysis_df, summary_df]

for df in dfs:
    # Ensure the experiment_name is lowercase
    df['experiment_name'] = df['experiment_name'].str.lower()
    
    # Split by underscore
    parts = df['experiment_name'].str.split('_', expand=True)
    
    # Extract barrier (first part) and day (last part)
    df['barrier'] = parts[0]  # '10cm', '15cm', '20cm'
    df['day'] = parts[2]      # 'day1', 'day2', 'day3'

# ---- notebook cell 32 ----------------------------------------------------------
# ---------------------------
# 1️⃣ Normalize keys for safe merge
# ---------------------------
for df in [summary_df, df_avg]:
    df['subject'] = df['subject'].astype(str).str.strip().str.lower()
    df['barrier'] = df['barrier'].astype(str).str.strip().str.lower()

# ---------------------------
# 2️⃣ Average trial_duration across all days in summary_df
# ---------------------------
df_trial_avg = (
    summary_df.groupby(['subject', 'barrier'], as_index=False)['trial_duration']
    .mean()
)

# ---------------------------
# 3️⃣ Merge into df_avg in place
# ---------------------------
df_avg = df_avg.merge(
    df_trial_avg,
    on=['subject', 'barrier'],
    how='left',       # keep all rows in df_avg
    validate='m:1'    # optional: ensures one match per row in df_avg
)

# ---- notebook cell 33 ----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1️⃣ Normalize keys
# ---------------------------
df_avg['subject'] = df_avg['subject'].astype(str).str.strip().str.lower()
df_avg['barrier'] = df_avg['barrier'].astype(str).str.strip().str.lower()
# NOTE (conversion): capitalize() yields 'Cnsds'; force the house spelling
df_avg['condition'] = (df_avg['condition'].str.capitalize()
                       .replace({'Cnsds': 'CNSDS'}))  # Control / CNSDS

barriers = sorted(df_avg['barrier'].unique(), key=lambda x: int(x.replace('cm','')))
conditions = ['Control', 'CNSDS']
palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35

# ---------------------------
# 2️⃣ Bootstrap function
# ---------------------------
def bootstrap_diff(control, cnsds, n_boot=10000):
    diffs = []
    for _ in range(n_boot):
        c_sample = np.random.choice(control, size=len(control), replace=True)
        s_sample = np.random.choice(cnsds, size=len(cnsds), replace=True)
        diffs.append(s_sample.mean() - c_sample.mean())
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_val = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return ci_lower, ci_upper, p_val

# ---------------------------
# 3️⃣ Compute summary statistics
# ---------------------------
summary = (
    df_avg
    .groupby(['barrier','condition'])['trial_duration']
    .agg(['mean','sem'])
    .reset_index()
)

# ---------------------------
# 4️⃣ Plot bars + individual subject points
# ---------------------------
plt.figure(figsize=(5,3))
x = np.arange(len(barriers))

for i, barrier in enumerate(barriers):
    for j, cond in enumerate(conditions):
        row = summary[(summary['barrier']==barrier) & (summary['condition']==cond)]
        if not row.empty:
            xpos = i - bar_width/2 if cond=='Control' else i + bar_width/2
            plt.bar(
                xpos, row['mean'].values[0],
                width=bar_width,
                yerr=row['sem'].values[0],
                facecolor='white',
                edgecolor=palette[cond],
                linewidth=1.5,
                capsize=5
            )
        
        # Overlay individual datapoints
        subset = df_avg[(df_avg['barrier']==barrier) & (df_avg['condition']==cond)]
        jitter = np.random.normal(loc=0, scale=0.04, size=len(subset))
        xpos_points = np.full(len(subset), xpos) + jitter
        plt.scatter(xpos_points, subset['trial_duration'], color=palette[cond], zorder=10, alpha=1.0, s=10)

# ---------------------------
# 5️⃣ Bootstrap asterisks
# ---------------------------
y_max = df_avg['trial_duration'].max()
offset = y_max * 0.05

for i, barrier in enumerate(barriers):
    control_vals = df_avg[(df_avg['barrier']==barrier) & (df_avg['condition']=='Control')]['trial_duration'].values
    cnsds_vals = df_avg[(df_avg['barrier']==barrier) & (df_avg['condition']=='CNSDS')]['trial_duration'].values
    if len(control_vals)==0 or len(cnsds_vals)==0:
        continue
    ci_low, ci_high, p_val = bootstrap_diff(control_vals, cnsds_vals)
    print(f"Barrier {barrier}: n_control={len(control_vals)}, n_cnsds={len(cnsds_vals)}, bootstrap p={p_val:.4f}, CI=({ci_low:.2f}, {ci_high:.2f})")
    if ci_low > 0 or ci_high < 0:
        x1, x2 = i - bar_width/2, i + bar_width/2
        y = max(control_vals.max(), cnsds_vals.max()) + offset
        # NOTE (conversion): significance bracket removed for consistency
        # with the other barrier figures - asterisk only.
        plt.text(i, y, '*', ha='center', va='bottom', fontsize=SIG_STAR_FONTSIZE)

# ---------------------------
# 6️⃣ Final formatting
# ---------------------------
plt.xticks(ticks=x, labels=[b.capitalize() for b in barriers], fontsize=12)
plt.ylabel('Trial Duration (s)', fontsize=14)
plt.xlabel('Barrier', fontsize=14)
sns.despine()
plt.legend(
    conditions, 
    title='Condition', 
    loc='upper left',        # position relative to axes
    bbox_to_anchor=(1.02, 1), # move it to the right of the plot
    borderaxespad=0,         # minimal padding
    fontsize=12
)
plt.tight_layout()
plt.savefig(r"c:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\trial_duration_bootstrap.pdf", format="pdf")
plt.show()

# ---- notebook cell 34 ----------------------------------------------------------
# Normalize the 'subject' column in both DataFrames to lowercase
summary_df['subject'] = summary_df['subject'].str.lower()
analysis_df['subject'] = analysis_df['subject'].str.lower()

# Group by 'subject', 'date', and 'experiment_name' and calculate the average trial_duration
avg_trial_duration_per_date = (
    summary_df
    .groupby(['subject', 'date', 'experiment_name'])['trial_duration']
    .mean()
    .reset_index(name='avg_trial_duration_per_date')
)

# Merge the averaged trial duration into analysis_df
analysis_df = analysis_df.merge(
    avg_trial_duration_per_date,
    on=['subject', 'experiment_name'],
    how='left'
)

# ---- notebook cell 35 ----------------------------------------------------------
analysis_df = analysis_df.rename(columns={
    'avg_trial_duration_per_date_x': 'trial_duration',
    'avg_trial_duration_per_date': 'trial_duration'
})
analysis_df = analysis_df.drop(columns=[col for col in ['avg_trial_duration_per_date_y'] if col in analysis_df.columns])

# ---- notebook cell 36 ----------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Average trial_duration across subject and day for each barrier
# ---------------------------
avg_trial_duration = (
    analysis_df
    .groupby(['barrier', 'condition', 'subject'])['trial_duration']
    .mean()
    .reset_index()
)

# ---------------------------
# 2️⃣ Run ANOVA
# ---------------------------
model = ols('trial_duration ~ C(barrier) * C(condition)', data=avg_trial_duration).fit()
anova_table = anova_lm(model, typ=2)
print("ANOVA Results:")
print(anova_table)

# ---------------------------
# 3️⃣ Tukey HSD post hoc test
# ---------------------------
tukey = pairwise_tukeyhsd(
    endog=avg_trial_duration['trial_duration'],
    groups=avg_trial_duration['barrier'] + "_" + avg_trial_duration['condition'],
    alpha=0.05
)
print("\nTukey HSD Results:")
print(tukey.summary())

# ---------------------------
# 4️⃣ Prepare for plotting
# ---------------------------
palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35
barriers = avg_trial_duration['barrier'].unique()
conditions = ['Control', 'CNSDS']

# Calculate means and SEMs for plotting
summary = (
    avg_trial_duration
    .groupby(['barrier', 'condition'])['trial_duration']
    .agg(['mean', 'sem'])
    .reset_index()
)

# ---------------------------
# 5️⃣ Plot bar graph
# ---------------------------
plt.figure(figsize=(4, 2))

for i, barrier in enumerate(barriers):
    for j, condition in enumerate(conditions):
        row = summary[(summary['barrier'] == barrier) & (summary['condition'] == condition)]
        if not row.empty:
            y = row['mean'].values[0]
            yerr = row['sem'].values[0]
            xpos = i - bar_width / 2 if condition == 'Control' else i + bar_width / 2
            plt.bar(
                xpos, y, width=bar_width, yerr=yerr,
                facecolor='white', edgecolor=palette[condition], linewidth=2, capsize=5
            )

sns.stripplot(
    data=avg_trial_duration,
    x='barrier',
    y='trial_duration',
    hue='condition',
    hue_order=['Control', 'CNSDS'],   # <-- force order
    dodge=True,
    palette=palette,
    jitter=True,
    marker='o',
    alpha=0.7
)

# Remove duplicate legend
legend = plt.gca().get_legend()
if legend:
    legend.remove()
# ---------------------------
# 6️⃣ Add asterisks using Tukey results (corrected)
# ---------------------------
tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

y_max = avg_trial_duration['trial_duration'].max()
offset = 0.05 * y_max

for i, barrier in enumerate(barriers):
    group1 = f"{barrier}_Control"
    group2 = f"{barrier}_CNSDS"

    match = tukey_df[
        ((tukey_df['group1'] == group1) & (tukey_df['group2'] == group2)) |
        ((tukey_df['group1'] == group2) & (tukey_df['group2'] == group1))
    ]

    if not match.empty and match['reject'].values[0]:  # significant after correction
        control_vals = avg_trial_duration[
            (avg_trial_duration['barrier'] == barrier) & (avg_trial_duration['condition'] == 'Control')
        ]['trial_duration']
        cnsds_vals = avg_trial_duration[
            (avg_trial_duration['barrier'] == barrier) & (avg_trial_duration['condition'] == 'CNSDS')
        ]['trial_duration']

        max_y = max(control_vals.max(), cnsds_vals.max())
        plt.text(i, max_y + offset, '*', ha='center', va='bottom', fontsize=SIG_STAR_FONTSIZE, color='black')


# ==========================================================================
# IMPORT DLC FILES
# ==========================================================================

# ---- notebook cell 38 ----------------------------------------------------------
import os
import re
import pandas as pd

# Function to extract metadata
def extract_metadata(filename):
    trial_match = re.search(r'_(\d+)DLC', filename)
    trial = int(trial_match.group(1)) + 1 if trial_match else None
    
    experiment_match = re.search(r'Barrier_Testing__?(\d+cm)_(Day_?([123]))', filename, re.IGNORECASE)
    experiment = f"Barrier_Testing_{experiment_match.group(1)}_Day_{experiment_match.group(3)}" if experiment_match else None
    
    subject_match = re.search(r'(WT\d+|MCN\d+)', filename)
    subject = subject_match.group(1) if subject_match else None
    
    condition_match = re.search(r'(CNSDS|Control)', filename, re.IGNORECASE)
    condition = condition_match.group(1) if condition_match else None
    # NOTE (conversion): normalise case so pts['condition'] matches the
    # 'Control' / 'CNSDS' spelling used everywhere else.
    if condition is not None:
        condition = 'Control' if condition.upper() == 'CONTROL' else 'CNSDS'
    
    return trial, experiment, subject, condition

# Read all DLC files into a list
dfs = []
for file in dlc_files:
    try:
        df = pd.read_hdf(file)
        
        # Add metadata immediately
        filename = os.path.splitext(os.path.basename(file))[0]
        trial, experiment, subject, condition = extract_metadata(os.path.basename(file))
        df['name'] = filename
        df['trial'] = trial
        df['experiment'] = experiment
        df['subject'] = subject
        df['condition'] = condition
        
        dfs.append(df)
    except Exception as e:
        console(f"Error reading {file}: {e}")

# Concatenate all files
pts = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ---- notebook cell 39 ----------------------------------------------------------
pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]

# ---- notebook cell 40 ----------------------------------------------------------
pts.columns = list(pts.columns[:-5]) + ['name', 'trial', 'experiment', 'subject', 'condition']

# ---- notebook cell 41 ----------------------------------------------------------
# import os
# import re
# import pandas as pd

# # Column name for the file
# filename_column = 'name'

# # Initialize list to store DataFrames
# dataframes = []

# # Function to extract metadata from filename
# def extract_metadata_from_filename(filename):
#     trial_match = re.search(r'_(\d+)DLC', filename)
#     trial = int(trial_match.group(1)) + 1 if trial_match else None
    
#     experiment_match = re.search(r'Barrier_Testing__?(\d+cm)_(Day_?([123]))', filename, re.IGNORECASE)
#     if experiment_match:
#         barrier = experiment_match.group(1)
#         day_number = experiment_match.group(3)
#         experiment = f"Barrier_Testing_{barrier}_Day_{day_number}"
#     else:
#         experiment = None

#     subject_match = re.search(r'(WT\d+|MCN\d+)', filename)
#     subject = subject_match.group(1) if subject_match else None

#     condition_match = re.search(r'(CNSDS|Control)', filename, re.IGNORECASE)
#     condition = condition_match.group(1) if condition_match else None

#     return trial, experiment, subject, condition

# # Loop through DLC files
# for file in dlc_files:
#     try:
#         df = pd.read_hdf(file)
        
#         # Extract clean filename
#         df[filename_column] = os.path.splitext(os.path.basename(file))[0]
        
#         # Extract metadata
#         trial, experiment, subject, condition = extract_metadata_from_filename(os.path.basename(file))
#         df['trial'] = trial
#         df['experiment'] = experiment
#         df['subject'] = subject
#         df['condition'] = condition
        
#         dataframes.append(df)
        
#     except Exception as e:
#         print(f"Error reading file {file}: {e}")

# # Concatenate all DataFrames
# if dataframes:
#     pts = pd.concat(dataframes, ignore_index=True)
#     print("Successfully concatenated DLC files into a single DataFrame.")
# else:
#     pts = pd.DataFrame()
#     print("No valid DLC files were imported.")

# # If hierarchical columns exist, flatten them
# if isinstance(pts.columns, pd.MultiIndex):
#     pts.columns = [' '.join(col[1:3]).strip() for col in pts.columns.values]

# # --- Ensure metadata columns exist before trying to reorder ---
# for col in ['trial', 'experiment', 'subject', 'condition', filename_column]:
#     if col not in pts.columns:
#         pts[col] = None  # Fill missing metadata columns with None

# # Optional: reorder so metadata columns are at the end
# metadata_cols = ['trial', 'experiment', 'subject', 'condition', filename_column]
# other_cols = [col for col in pts.columns if col not in metadata_cols]
# pts = pts[other_cols + metadata_cols]

# ---- notebook cell 42 ----------------------------------------------------------
# # Initialize an empty list to store DataFrames
# dataframes = []

# # Loop through each file and import its contents
# for file in dlc_files:
#     try:
#         # Read the HDF file
#         df = pd.read_hdf(file)
        
#         # Append to the list of DataFrames
#         dataframes.append(df)
#     except Exception as e:
#         print(f"Error reading file {file}: {e}")

# # Concatenate all the DataFrames into a single DataFrame
# if dataframes:
#     pts = pd.concat(dataframes, ignore_index=True)
#     print("Successfully concatenated DLC files into a single DataFrame.")
# else:
#     pts = pd.DataFrame()  # If no files are read successfully
#     print("No valid DLC files were imported.")

# # # Display the concatenated DataFrame
# # print(pts.head())

# ---- notebook cell 43 ----------------------------------------------------------
# pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]

# ---- notebook cell 44 ----------------------------------------------------------
# pts.columns = list(pts.columns[:-4]) + ['trial', 'experiment', 'subject', 'condition']

# ---- notebook cell 45 ----------------------------------------------------------
# NOTE (conversion): intermediate data dump, off by default (large file).
if SAVE_INTERMEDIATE_CSVS:
    pts.to_csv(os.path.join(OUTPUT_ROOT, f"pts_{RUN_ID}.csv"), index=False)

# ---- notebook cell 46 ----------------------------------------------------------
# Step 1: Convert 'experiment', 'trial', and 'subject' columns to strings (if they aren't already)
pts['experiment'] = pts['experiment'].astype(str)
pts['trial'] = pts['trial'].astype(str)
pts['subject'] = pts['subject'].astype(str)

#summary_df['experiment_name'] = summary_df['experiment_name'].astype(str)
#summary_df['trial'] = summary_df['trial'].astype(str)
#summary_df['subject'] = summary_df['subject'].astype(str)

# Step 2: Clean the 'experiment', 'trial', and 'subject' columns (strip spaces)
pts['experiment'] = pts['experiment'].str.strip()
pts['trial'] = pts['trial'].str.strip()
pts['subject'] = pts['subject'].str.strip()

#summary_df['experiment_name'] = summary_df['experiment_name'].str.strip()
#summary_df['trial'] = summary_df['trial'].str.strip()
#summary_df['subject'] = summary_df['subject'].str.strip()

# Step 3: Check unique values to verify they match (optional, just for debugging)
console("Unique experiment values in pts:", pts['experiment'].unique())
#print("Unique experiment values in summary_df:", summary_df['experiment_name'].unique())


# ==========================================================================
# CREATE MERGED DF
# ==========================================================================

# ---- notebook cell 48 ----------------------------------------------------------
# Rename 'experiment_name' to 'experiment' in summary_df
#summary_df.rename(columns={'experiment_name': 'experiment'}, inplace=True)
analysis_df.rename(columns={'experiment_name': 'experiment'}, inplace=True)

# Step 4: Merge the dataframes on 'experiment', 'trial', and 'subject'
#merged_df = pd.merge(pts, summary_df, on=['experiment', 'trial', 'subject'], how='left')

# Step 5: Check the merged DataFrame
#print(merged_df.head())


# ==========================================================================
# CALCULATE AVG LOCATION
# ==========================================================================

# ---- notebook cell 50 ----------------------------------------------------------
# Replace spaces with underscores in all column names
pts.columns = pts.columns.str.replace(' ', '_')

# Verify the updated column names
console(pts.columns)

# ---- notebook cell 51 ----------------------------------------------------------
# Now, proceed with your logic to compute the averages
body_parts = ['Nose_', 'Right_Ear_', 'Left_Ear_', 'Middle_Neck_', 'Middle_Spine_', 
              'Tail_Base_', 'Tail_Mid_', 'Tail_End_', 'Front_Left_Leg_', 
              'Front_Right_Leg_', 'Back_Left_Leg_', 'Back_Right_Leg_']

coordinates = ['x', 'y']
likelihood_suffix = 'likelihood'

avg_x = []
avg_y = []

# Loop through each row of the DataFrame
for index, row in pts.iterrows():
    x_vals = []
    y_vals = []
    
    # Loop through each body part
    for part in body_parts:
        x_col = f"{part}x"
        y_col = f"{part}y"
        likelihood_col = f"{part}{likelihood_suffix}"
        
        # Check if the likelihood for this body part is greater than 0.7
        if row[likelihood_col] > 0.7:  # Assuming 0.7 is the threshold for likelihood
            x_vals.append(row[x_col])
            y_vals.append(row[y_col])
    
    # Exclude 'Maze_Center' columns
    if 'Maze_Center_x' in pts.columns and 'Maze_Center_y' in pts.columns:
        maze_center_x = row.get('Maze_Center_x', None)
        maze_center_y = row.get('Maze_Center_y', None)
        if maze_center_x and maze_center_y:
            # Remove the maze center values if present
            x_vals = [val for val in x_vals if val != maze_center_x]
            y_vals = [val for val in y_vals if val != maze_center_y]

    # Calculate the average of x and y coordinates (if there are valid points)
    if x_vals:
        avg_x.append(sum(x_vals) / len(x_vals))  # Average of x coordinates
        avg_y.append(sum(y_vals) / len(y_vals))  # Average of y coordinates
    else:
        avg_x.append(None)  # If no valid points, append None
        avg_y.append(None)

# Append the average columns to the DataFrame
pts['avg_x'] = avg_x
pts['avg_y'] = avg_y


# ==========================================================================
# CALCULATE VELOCITIES
# ==========================================================================

# ---- notebook cell 53 ----------------------------------------------------------
pts['experiment'] = (
    pts['experiment']
    .str.replace(r'Barrier_Testing_(\d+cm)_Day_(\d)', r'\1_barrier_day\2', regex=True)
)

pts['subject'] = pts['subject'].str.strip().str.lower()

# ---- notebook cell 54 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# pts.columns

# ---- notebook cell 58 ----------------------------------------------------------
# NOTE (conversion): MOVED (was notebook cell 58, after the cells that use velocity_df).
# Constants
frame_rate = 60  # frames per second

# Scaling factors (convert pixels to cm)
scale_x = 7.62 / 230
scale_y = 38.735 / 885
scale_velocity = np.sqrt(scale_x**2 + scale_y**2)

# Sort pts to ensure correct frame order
pts_sorted = pts.sort_values(by=['experiment', 'subject', 'trial']).copy()

# Initialize column for per-frame velocity
pts_sorted['velocity_cm'] = np.nan

# Container for trial-level mean velocities
velocity_records = []

# Group by experiment → subject → trial
grouped = pts_sorted.groupby(['experiment', 'subject', 'trial'])

for (experiment, subject, trial), group in grouped:
    group = group.reset_index()  # keep original indices
    original_indices = group['index'].tolist()

    velocities_cm = [np.nan]  # First frame has no previous frame

    for i in range(1, len(group)):
        x1, y1 = group.loc[i - 1, ['avg_x', 'avg_y']]
        x2, y2 = group.loc[i, ['avg_x', 'avg_y']]

        # Distance in pixels
        distance_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Distance in cm
        distance_cm = distance_pixels * scale_velocity

        # Velocity in cm/s
        velocity_cm = distance_cm / (1 / frame_rate)

        velocities_cm.append(velocity_cm)

    # Update the main DataFrame with computed velocities
    pts_sorted.loc[original_indices, 'velocity_cm'] = velocities_cm

    # Record mean velocity for the trial (excluding NaN)
    mean_velocity = np.nanmean(velocities_cm)
    velocity_records.append({
        'experiment': experiment,
        'subject': subject,
        'trial': trial,
        'mean_velocity_cm': mean_velocity
    })

# Create a DataFrame with per-trial mean velocities
velocity_df = pd.DataFrame(velocity_records)

# Calculate subject-level average velocity across trials
subject_avg_velocity = (
    velocity_df
    .groupby(['experiment', 'subject'])['mean_velocity_cm']
    .mean()
    .reset_index()
    .rename(columns={'mean_velocity_cm': 'avg_velocity_across_trials'})
)

# Optional: overwrite pts with updated version
pts = pts_sorted

# ---- notebook cell 59 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# analysis_df

# ---- notebook cell 60 ----------------------------------------------------------
analysis_df = analysis_df.merge(
    subject_avg_velocity,
    left_on=['experiment', 'subject'],
    right_on=['experiment', 'subject'],
    how='left'
)

# ---- notebook cell 61 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# subject_avg_velocity

# ---- notebook cell 62 ----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------------
# 1️⃣ Average subjects across 3 days per barrier
# ---------------------------
analysis_df['barrier'] = analysis_df['experiment'].str.extract(r'(\d+cm)')

subject_barrier_avg = (
    analysis_df
    .groupby(['barrier', 'subject', 'condition'])['avg_velocity_across_trials']
    .mean()
    .reset_index()
)

# ---------------------------
# 2️⃣ ANOVA
# ---------------------------
# formula: avg_velocity ~ barrier * condition
anova_model = ols('avg_velocity_across_trials ~ C(barrier) * C(condition)', data=subject_barrier_avg).fit()
anova_table = anova_lm(anova_model, typ=2)
print("ANOVA Results:")
print(anova_table)

# ---------------------------
# 3️⃣ Tukey HSD post hoc
# ---------------------------
# Combine barrier + condition as groups
subject_barrier_avg['group'] = subject_barrier_avg['barrier'] + "_" + subject_barrier_avg['condition']

tukey = pairwise_tukeyhsd(
    endog=subject_barrier_avg['avg_velocity_across_trials'],
    groups=subject_barrier_avg['group'],
    alpha=0.05
)

print("\nTukey HSD Results:")
print(tukey.summary())

# ---------------------------
# 4️⃣ Plotting
# ---------------------------
sns.set_style("white")
palette = {"Control": CONTROL_GRAY, "CNSDS": CNSDS_GRAY}

# Calculate mean and SEM for bars
group_stats = (
    subject_barrier_avg
    .groupby(['barrier', 'condition'])['avg_velocity_across_trials']
    .agg(['mean', 'sem'])
    .reset_index()
)

barriers = ['10cm', '15cm', '20cm']
conditions = ['Control', 'CNSDS']

plt.figure(figsize=(5, 3))
bar_width = 0.35
x = range(len(barriers))

# Plot bars and subject points
for i, condition in enumerate(conditions):
    color = palette[condition]
    means = []
    sems = []

    for barrier in barriers:
        row = group_stats[
            (group_stats['barrier'] == barrier) &
            (group_stats['condition'] == condition)
        ]
        if not row.empty:
            means.append(row['mean'].values[0])
            sems.append(row['sem'].values[0])
        else:
            means.append(np.nan)
            sems.append(np.nan)

    bar_positions = [xi + (i - 0.5) * bar_width for xi in x]

    plt.bar(
        bar_positions,
        means,
        yerr=sems,
        capsize=5,
        width=bar_width,
        edgecolor=color,
        facecolor='white',
        linewidth=1.5,
        zorder=2
    )

    # Overlay subject points
    for j, barrier in enumerate(barriers):
        subject_points = subject_barrier_avg[
            (subject_barrier_avg['barrier'] == barrier) &
            (subject_barrier_avg['condition'] == condition)
        ]

        jitter_strength = 0.08
        x_jittered = bar_positions[j] + np.random.uniform(
            -jitter_strength, jitter_strength, size=len(subject_points)
        )

        plt.scatter(
            x_jittered,
            subject_points['avg_velocity_across_trials'],
            facecolor=color,
            edgecolor=color,
            alpha=1,
            zorder=3,
            linewidth=0.5,
            s=10
        )

# ---------------------------
# 5️⃣ Add asterisks for Tukey significant differences
# ---------------------------
tukey_df = pd.DataFrame(
    data=tukey._results_table.data[1:], 
    columns=tukey._results_table.data[0]
)

y_max = subject_barrier_avg['avg_velocity_across_trials'].max()
offset = 0.05 * y_max

for i, barrier in enumerate(barriers):
    group1 = f"{barrier}_Control"
    group2 = f"{barrier}_CNSDS"

    match = tukey_df[
        ((tukey_df['group1'] == group1) & (tukey_df['group2'] == group2)) |
        ((tukey_df['group1'] == group2) & (tukey_df['group2'] == group1))
    ]

    if not match.empty and match['reject'].values[0]:
        control_vals = subject_barrier_avg[
            (subject_barrier_avg['barrier'] == barrier) & (subject_barrier_avg['condition'] == 'Control')
        ]['avg_velocity_across_trials']
        cnsds_vals = subject_barrier_avg[
            (subject_barrier_avg['barrier'] == barrier) & (subject_barrier_avg['condition'] == 'CNSDS')
        ]['avg_velocity_across_trials']

        t_stat = (control_vals.mean() - cnsds_vals.mean()) / np.sqrt(control_vals.var()/len(control_vals) + cnsds_vals.var()/len(cnsds_vals))
        n_control = len(control_vals)
        n_cnsds = len(cnsds_vals)
        p_val = match['p-adj'].values[0]

        print(f"{barrier}: t={t_stat:.2f}, n_control={n_control}, n_CNSDS={n_cnsds}, p={p_val:.3f}")

        # Add asterisk above bars
        max_y = max(control_vals.max(), cnsds_vals.max())
        plt.text(i, max_y + offset, '*', ha='center', va='bottom', fontsize=SIG_STAR_FONTSIZE, color='black')

# ---------------------------
# 6️⃣ Final formatting
# ---------------------------
plt.xticks(x, barriers, fontsize=12)
plt.ylabel("Velocity (cm/s)", fontsize=14)
plt.xlabel("Barrier", fontsize=14)
sns.despine()
plt.tight_layout()
plt.savefig(r"C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\velocities.pdf", format="pdf")
plt.show()

# ---- notebook cell 63 ----------------------------------------------------------
# Normalize the avg_x and avg_y coordinates around the Maze_Center_x and Maze_Center_y columns
pts['normalized_x'] = pts['avg_x'] - pts['Maze_Center_x']
pts['normalized_y'] = pts['avg_y'] - pts['Maze_Center_y']

# ---- notebook cell 64 ----------------------------------------------------------
from scipy.stats import ttest_ind

# -------------------------------------------------------
# 1️⃣ Bin normalized Y location
# -------------------------------------------------------
bin_edges = range(-200, 850, 50)
pts['y_bin'] = pd.cut(pts['normalized_y'], bins=bin_edges, labels=range(-175, 825, 50))

# -------------------------------------------------------
# 2️⃣ Average velocity per subject per bin
# -------------------------------------------------------
anova_df = (
    pts.groupby(['subject','condition','y_bin'])
    .agg(avg_velocity_cm=('velocity_cm','mean'))
    .reset_index()
)

anova_df['y_bin'] = anova_df['y_bin'].astype(float)

# -------------------------------------------------------
# 3️⃣ Run t-tests for Control vs CNSDS in each bin
# -------------------------------------------------------
results = []

for bin_val in sorted(anova_df['y_bin'].dropna().unique()):

    bin_data = anova_df[anova_df['y_bin'] == bin_val]

    control = bin_data[bin_data['condition'] == 'Control']['avg_velocity_cm']
    cnsds = bin_data[bin_data['condition'] == 'CNSDS']['avg_velocity_cm']

    if len(control) > 1 and len(cnsds) > 1:

        t_stat, p_val = ttest_ind(control, cnsds, equal_var=False)

        n_control = len(control)
        n_cnsds = len(cnsds)

        sig = "* " if p_val < 0.05 else ""

        results.append({
            "y_bin": bin_val,
            "t": t_stat,
            "p": p_val,
            "n_control": n_control,
            "n_cnsds": n_cnsds,
            "significant": sig
        })

# -------------------------------------------------------
# 4️⃣ Print results table
# -------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("y_bin")

print("\n===== BIN COMPARISONS (Control vs CNSDS) =====\n")

for _, row in results_df.iterrows():

    star = "*" if row["p"] < 0.05 else ""

    print(
        f"{star} y_bin {row['y_bin']:>5} | "
        f"t = {row['t']:.3f} | "
        f"p = {row['p']:.4f} | "
        f"n_control = {row['n_control']} | "
        f"n_cnsds = {row['n_cnsds']}"
    )

print("\n* = p < 0.05")

# Optional: save results
# NOTE (conversion): intermediate data dump, off by default. The same
# per-bin statistics are printed above and appear in the report.
if SAVE_INTERMEDIATE_CSVS:
    results_df.to_csv(os.path.join(OUTPUT_ROOT, f"velocity_bin_stats_{RUN_ID}.csv"), index=False)

# ---- notebook cell 65 ----------------------------------------------------------
# Step 1: Bin the normalized Y location into bins (adjust bin edges as needed)
bin_edges = range(-200, 850, 50)  # example bins every 50 units from -200 to 800
pts['y_bin'] = pd.cut(pts['normalized_y'], bins=bin_edges, labels=range(-175, 825, 50))

# Step 2: Calculate average velocity at each y_bin per condition
binned = (
    pts.groupby(['condition', 'y_bin'])
    .agg(avg_velocity_cm=('velocity_cm', 'mean'))
    .reset_index()
)

# Step 3: Prepare plot
fig, axs = plt.subplots(2, 1, figsize=(4, 3), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

colors = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
conditions = ['Control', 'CNSDS']

for i, condition in enumerate(conditions):
    ax = axs[i]
    data = binned[binned['condition'] == condition].sort_values('y_bin')
    
    # Convert y_bin labels to floats for plotting on x-axis
    x_vals = data['y_bin'].astype(float)
    y_vals = data['avg_velocity_cm']

    ax.bar(
        x_vals,
        y_vals,
        width=30,
        color=colors[condition],
        alpha=0.8,
        edgecolor='none'
    )
    ax.set_ylabel('Velocity (cm/sec)', fontsize=10)
    ax.set_title(condition, fontsize=14)
    ax.grid(False)
    ax.set_ylim(0, 150)

# Shared x-axis label
axs[-1].set_xlabel('Y Location', fontsize=12)
axs[-1].set_xlim(800, -200)

plt.tight_layout()
plt.savefig(r"C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\velocity_by_y_location.pdf", format="pdf")
plt.show()

# ---- notebook cell 66 ----------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Assume pts is your DataFrame and conditions is a list of the two conditions, e.g.:
# conditions = ['CNSDS', 'Control']

# Combine data for bin edges
combined_data_x = pts['normalized_x'].dropna()
combined_data_y = pts['normalized_y'].dropna()

# Define number of bins
bins = 30
bin_edges_x = np.linspace(combined_data_x.min(), combined_data_x.max(), bins + 1)
bin_edges_y = np.linspace(combined_data_y.min(), combined_data_y.max(), bins + 1)

# Create figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

for i, condition in enumerate(conditions):
    # Get data for current condition
    condition_data_x = pts[pts['condition'] == condition]['normalized_x'].dropna()
    condition_data_y = pts[pts['condition'] == condition]['normalized_y'].dropna()

    # Calculate weights for percentage histograms
    weights_x = np.ones_like(condition_data_x) / len(condition_data_x) * 100
    weights_y = np.ones_like(condition_data_y) / len(condition_data_y) * 100

    # --- X-coordinate histogram (vertical bars) ---
    ax_x = axs[i, 0]
    ax_x.hist(condition_data_x, bins=bin_edges_x, weights=weights_x,
              alpha=0.7,
              color=CNSDS_GRAY if condition == 'CNSDS' else CONTROL_GRAY,
              histtype='bar', rwidth=0.9)
    ax_x.set_xlabel('Distance From Maze Center', fontsize=18)
    ax_x.set_ylabel('% of Trial', fontsize=25)
    ax_x.tick_params(axis='both', labelsize=18)
    # NOTE (conversion): was set_ylim(1, 45). Starting at 1 hid every
    # bin under 1%, and 45 clipped anything taller. Autoscale from 0
    # so the panel shows all of the data it is counting.
    ax_x.set_ylim(bottom=0)
    ax_x.set_xlim(-200, 200)
    ax_x.grid(False)
    ax_x.legend([condition], loc='upper right', fontsize=18)

    # --- Y-coordinate histogram (horizontal bars) ---
    ax_y = axs[i, 1]
    ax_y.hist(condition_data_y, bins=bin_edges_y, weights=weights_y,
              alpha=0.7,
              color=CNSDS_GRAY if condition == 'CNSDS' else CONTROL_GRAY,
              histtype='bar', rwidth=0.9,
              orientation='horizontal')  # ✅ Horizontal bars
    ax_y.set_xlabel('% of Trial', fontsize=25)
    ax_y.set_ylabel('Distance From Maze Center', fontsize=18)
    ax_y.tick_params(axis='both', labelsize=18)
    # NOTE (conversion): was set_xlim(1, 20), same issue as above.
    ax_y.set_xlim(left=0)
    ax_y.set_ylim(800, -200)
    ax_y.grid(False)
    ax_y.legend([condition], loc='upper right', fontsize=18)

# Final layout and save
plt.tight_layout()
plt.savefig("location_histogram.pdf", format="pdf")
plt.show()


# ==========================================================================
# LOCATIONS SCATTERPLOT
# ==========================================================================

# ---- notebook cell 68 ----------------------------------------------------------
summary_df_copy=summary_df # do i need this?

# ---- notebook cell 69 ----------------------------------------------------------
pts_copy=pts

# ---- notebook cell 70 ----------------------------------------------------------
# Deduplicate summary_df_copy to ensure one row per subject
summary_unique = summary_df_copy[['subject', 'HR_Arm']].drop_duplicates(subset='subject')

# Merge without duplicating pts_copy
pts = pts_copy.copy()
pts['HR_Arm'] = pts['subject'].map(summary_unique.set_index('subject')['HR_Arm'])


# ==========================================================================
# GET AN EXAMPLE IMAGE FRAME FROM A VIDEO FOR PLOTTING
# ==========================================================================

# ---- notebook cell 72 ----------------------------------------------------------
# vid_pic_file = r'C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\05052024\WT045\050524_Barrier_Testing_10cm_Day_1_WT045FCNSDS_StartArm_3.avi'
# vid = cv2.VideoCapture(vid_pic_file)
# vid.set(cv2.CAP_PROP_POS_FRAMES,170)
# ret, frame = vid.read()
# console(ret)

# vid.release()

# plt.imshow(frame)

# vid_dlc_file = r'C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\05052024\WT045\050524_Barrier_Testing_10cm_Day_1_WT045FCNSDS_StartArm_3DLC_resnet50_Effort_Related_Choice_2Jul30shuffle1_100000_filtered.h5'
# dlc_df = pd.read_hdf(vid_dlc_file)
# dlc_df.columns = [' '.join(col[:][1:3]).strip() for col in dlc_df.columns.values]
# console(dlc_df.keys())
# low_like = dlc_df['Maze_Center likelihood']<0.9
# x = dlc_df['Maze_Center x']
# x[low_like]=np.nan
# x_pic = int(np.nanmean(x))
# y = dlc_df['Maze_Center y']
# y[low_like]=np.nan
# y_pic = int(np.nanmean(y))
# plt.plot(x_pic,y_pic,'ro')

# # plt.figure()
# # frame_2 = np.roll(a=frame,shift=-x_pic,axis=0)
# # frame_2 = np.roll(a=frame_2,shift=-y_pic,axis=1)
# # plt.imshow(frame_2)

# ---- notebook cell 94 ----------------------------------------------------------
# NOTE (conversion): MOVED (was notebook cell 94, after the scatterplot that uses `frame`).
vid_pic_file = r'C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\05052024\WT045\050524_Barrier_Testing_10cm_Day_1_WT045FCNSDS_StartArm_3.avi'
vid = cv2.VideoCapture(vid_pic_file)
vid.set(cv2.CAP_PROP_POS_FRAMES,170)
ret, frame = vid.read()
console(ret)

vid_dlc_file = r'C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\05052024\WT045\050524_Barrier_Testing_10cm_Day_1_WT045FCNSDS_StartArm_3DLC_resnet50_Effort_Related_Choice_2Jul30shuffle1_100000_filtered.h5'
dlc_df = pd.read_hdf(vid_dlc_file)
dlc_df.columns = [' '.join(col[:][1:3]).strip() for col in dlc_df.columns.values]
console(dlc_df.keys())
low_like = dlc_df['Maze_Center likelihood']<0.9
x = dlc_df['Maze_Center x']
x[low_like]=np.nan
x_pic = int(np.nanmean(x))
y = dlc_df['Maze_Center y']
y[low_like]=np.nan
y_pic = int(np.nanmean(y))
plt.plot(x_pic,y_pic,'ro')

# ---- notebook cell 73 ----------------------------------------------------------
import matplotlib.pyplot as plt


# NOTE (conversion): scatterplot rebuilt on request.
#   * Rows are condition, columns are barrier height, and days 1-3 are pooled
#     within each panel - 6 panels, 3 across the page.
#   * The previous version made one column per value of pts['experiment'],
#     which is 9 values (3 barriers x 3 days) rather than 3, but took the
#     title from the column index. Only the first three panels were labelled
#     correctly; the other six all read "20cm Barrier".
#   * Conditions are on separate panels so the two point clouds no longer have
#     to be told apart by shade.
#   * Background frame dimmed, tight spacing, smaller fonts, y limits 600 to
#     -200 to match the other spatial figures.

scatter_pts = pts.copy()
scatter_pts['barrier_label'] = (
    scatter_pts['experiment'].astype(str).str.extract(r'(\d+\s*cm)', expand=False)
    .str.replace(' ', '', regex=False).str.lower())

barrier_order = [b for b in ['10cm', '15cm', '20cm']
                 if b in set(scatter_pts['barrier_label'].dropna())]
if not barrier_order:
    console('[scatterplot] could not parse barrier from experiment names.')
    barrier_order = ['all']
    scatter_pts['barrier_label'] = 'all'

scatter_conditions = ['Control', 'CNSDS']

# The x range is 400 units and the y range 800, and imshow keeps the maze at
# true aspect, so each panel is inherently twice as tall as it is wide. Sizing
# the panels to that ratio is what removes the side gaps - with a wider panel
# the plot just gets letterboxed inside it and the extra shows as white space.
X_LIMITS = (200, -200)
# NOTE (conversion): y upper bound restored to 800 so the start box region
# (y 600-800) is not cut off the top of the panels. Panel proportions are
# derived from these limits, so the layout resizes itself.
Y_LIMITS = (800, -200)
_panel_height = 3.3
_panel_width = _panel_height * (abs(X_LIMITS[0] - X_LIMITS[1]) /
                                abs(Y_LIMITS[0] - Y_LIMITS[1]))

fig, axes = plt.subplots(
    len(scatter_conditions), len(barrier_order),
    figsize=(_panel_width * len(barrier_order) + 0.5,
             _panel_height * len(scatter_conditions) + 0.6),
    sharex=True, sharey=True, squeeze=False)
# The x limits are reversed (200 on the left, -200 on the right), so the right
# edge of one panel and the left edge of the next both carry a 3-4 character
# label. wspace has to clear both, and the right margin has to leave room for
# the final panel's label or it gets clipped at the page edge.
fig.subplots_adjust(left=0.11, right=0.955, top=0.94, bottom=0.07,
                    wspace=0.22, hspace=0.08)

height, width, chan = frame.shape
extent = [-x_pic, width - x_pic, height - y_pic, -y_pic]
angle = 2  # counter-clockwise, matches the camera tilt

for row, condition in enumerate(scatter_conditions):
    for col, barrier_value in enumerate(barrier_order):
        ax = axes[row][col]

        transform = Affine2D().rotate_deg_around(0, 0, angle) + ax.transData
        ax.imshow(frame, extent=extent, origin='upper', transform=transform,
                  alpha=0.40, zorder=0)

        panel = scatter_pts[
            (scatter_pts['condition'] == condition) &
            (scatter_pts['barrier_label'] == barrier_value)
        ].copy()

        if not panel.empty:
            # Reflect x so every animal's high-reward arm is on the same side.
            panel.loc[
                panel['HR_Arm'].astype(str).str.lower() == 'right', 'normalized_x'
            ] *= -1
            ax.scatter(panel['normalized_x'], panel['normalized_y'],
                       alpha=0.45, marker='.', s=0.35, zorder=2,
                       color=CONTROL_GRAY, edgecolors='none')

        if row == 0:
            ax.set_title(f'{barrier_value} Barrier', fontsize=16, pad=6)
        if col == 0:
            ax.set_ylabel(condition, fontsize=16, labelpad=7)

        ax.set_xlim(*X_LIMITS)
        ax.set_ylim(*Y_LIMITS)
        ax.tick_params(axis='both', labelsize=10, pad=1.5)
        ax.grid(True, alpha=0.25)

plt.savefig("location_scatterplot.pdf", format="pdf")
plt.show()


# ==========================================================================
# TIME SPENT IN START BOX
# ==========================================================================

# ---- notebook cell 78 ----------------------------------------------------------
# NOTE (conversion): MOVED (was notebook cell 78, after the cells that display `merged`).
# ---------------------------
# 1️⃣ Identify start box frames
# ---------------------------
start_box_df = pts[
    (pts['normalized_y'] >= 600) & (pts['normalized_y'] <= 800)
]

# ---------------------------
# 2️⃣ Count total frames per trial
# ---------------------------
total_frames_per_trial = (
    pts.groupby(['experiment', 'subject', 'trial'])
    .size()
    .reset_index(name='total_frames')
)

# ---------------------------
# 3️⃣ Count start box frames per trial
# ---------------------------
start_box_frames_per_trial = (
    start_box_df.groupby(['experiment', 'subject', 'trial'])
    .size()
    .reset_index(name='start_box_frames')
)

# ---------------------------
# 4️⃣ Merge frame counts
# ---------------------------
merged = pd.merge(
    total_frames_per_trial,
    start_box_frames_per_trial,
    on=['experiment', 'subject', 'trial'],
    how='left'
)

merged['start_box_frames'] = merged['start_box_frames'].fillna(0)

# ---------------------------
# 5️⃣ Convert frames → seconds
# ---------------------------
FPS = 60

merged['start_box_time_sec'] = merged['start_box_frames'] / FPS
merged['total_time_sec'] = merged['total_frames'] / FPS

# ---------------------------
# 6️⃣ Average across trials → per subject per day (experiment)
# ---------------------------
avg_start_box_time = (
    merged.groupby(['experiment', 'subject'])['start_box_time_sec']
    .mean()
    .reset_index(name='avg_start_box_time_sec')
)

# ---------------------------
# 7️⃣ Merge into analysis_df
# ---------------------------
analysis_df = analysis_df.merge(
    avg_start_box_time,
    on=['experiment', 'subject'],
    how='left'
)

# ---- notebook cell 75 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# merged

# ---- notebook cell 76 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# summary_df

# ---- notebook cell 77 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# pts.columns

# ---- notebook cell 79 ----------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------------
# 0️⃣ Extract barrier
# ---------------------------
analysis_df['barrier'] = analysis_df['experiment'].str.extract(r'(\d+cm)')

analysis_df['condition'] = pd.Categorical(
    analysis_df['condition'],
    categories=['Control', 'CNSDS'],
    ordered=True
)

palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35
barriers = ['10cm', '15cm', '20cm']

# ---------------------------
# 1️⃣ Subject-level averaging
# ---------------------------
subject_barrier_avg = (
    analysis_df
    .groupby(['barrier', 'subject', 'condition'])['avg_start_box_time_sec']
    .mean()
    .reset_index()
)

# ---------------------------
# 2️⃣ ANOVA
# ---------------------------
model = ols(
    'avg_start_box_time_sec ~ C(barrier) * C(condition)',
    data=subject_barrier_avg
).fit()

anova_table = anova_lm(model, typ=2)

print("\n====================")
print("ANOVA RESULTS")
print("====================")
print(anova_table)

# ---------------------------
# 3️⃣ Tukey HSD
# ---------------------------
subject_barrier_avg = subject_barrier_avg.dropna(
    subset=['avg_start_box_time_sec']
)

subject_barrier_avg['group'] = (
    subject_barrier_avg['barrier'].astype(str) + "_" +
    subject_barrier_avg['condition'].astype(str)
)

tukey = pairwise_tukeyhsd(
    endog=subject_barrier_avg['avg_start_box_time_sec'],
    groups=subject_barrier_avg['group'],
    alpha=0.05
)

print("\n====================")
print("Tukey Results")
print("====================")
print(tukey.summary())

tukey_df = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

# ---------------------------
# 4️⃣ Means + SEM
# ---------------------------
summary = (
    subject_barrier_avg
    .groupby(['barrier', 'condition'])['avg_start_box_time_sec']
    .agg(['mean', 'sem'])
    .reset_index()
)

# ---------------------------
# 5️⃣ Plot
# ---------------------------
plt.figure(figsize=(6, 4))

for i, cond in enumerate(['Control', 'CNSDS']):
    for j, barrier in enumerate(barriers):
        row = summary[
            (summary['barrier'] == barrier) &
            (summary['condition'] == cond)
        ]
        if not row.empty:
            y = row['mean'].values[0]
            yerr = row['sem'].values[0]
            xpos = j - bar_width/2 if cond == 'Control' else j + bar_width/2

            plt.bar(
                xpos, y,
                width=bar_width,
                yerr=yerr,
                facecolor='white',
                edgecolor=palette[cond],
                linewidth=2,
                capsize=5
            )

# overlay subject points
sns.stripplot(
    data=subject_barrier_avg,
    x='barrier',
    y='avg_start_box_time_sec',
    hue='condition',
    dodge=True,
    palette=palette,
    jitter=True,
    alpha=0.7
)

# NOTE (conversion): legend added on request.
plt.legend(title='Condition', frameon=False)

# ---------------------------
# 6️⃣ Tukey + EFFECT SIZE (Cohen's d)
# ---------------------------
y_max = subject_barrier_avg['avg_start_box_time_sec'].max()
offset = 0.05 * y_max

print("\n====================")
print("PAIRWISE RESULTS (with effect size)")
print("====================")

for i, barrier in enumerate(barriers):

    g1 = f"{barrier}_Control"
    g2 = f"{barrier}_CNSDS"

    match = tukey_df[
        ((tukey_df['group1'] == g1) & (tukey_df['group2'] == g2)) |
        ((tukey_df['group1'] == g2) & (tukey_df['group2'] == g1))
    ]

    if not match.empty and match['reject'].values[0]:

        vals1 = subject_barrier_avg[
            (subject_barrier_avg['barrier'] == barrier) &
            (subject_barrier_avg['condition'] == 'Control')
        ]['avg_start_box_time_sec']

        vals2 = subject_barrier_avg[
            (subject_barrier_avg['barrier'] == barrier) &
            (subject_barrier_avg['condition'] == 'CNSDS')
        ]['avg_start_box_time_sec']

        # ---------------------------
        # stats
        # ---------------------------
        n1, n2 = len(vals1), len(vals2)
        m1, m2 = vals1.mean(), vals2.mean()

        v1, v2 = vals1.var(ddof=1), vals2.var(ddof=1)

        # t-test
        t_stat = (m1 - m2) / np.sqrt(v1/n1 + v2/n2)
        p_val = match['p-adj'].values[0]

        # Cohen's d (pooled SD)
        sp = np.sqrt(((n1 - 1)*v1 + (n2 - 1)*v2) / (n1 + n2 - 2))
        cohens_d = (m1 - m2) / sp

        print(
            f"{barrier}: "
            f"t={t_stat:.2f}, "
            f"p={p_val:.4f}, "
            f"n_control={n1}, n_CNSDS={n2}, "
            f"Cohen's d={cohens_d:.2f}"
        )

        # significance marker
        max_y = max(vals1.max(), vals2.max())
        plt.text(i, max_y + offset, '*', ha='center', fontsize=SIG_STAR_FONTSIZE)

# ---------------------------
# 7️⃣ Final formatting
# ---------------------------
plt.xticks(ticks=range(len(barriers)), labels=barriers, fontsize=16)
plt.ylabel('Start Box Time (s)', fontsize=20)
plt.xlabel('Barrier', fontsize=20)
sns.despine()

plt.tight_layout()
plt.savefig(
    r"c:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\startbox_time.pdf",
    format="pdf"
)
plt.show()

# ---- notebook cell 80 ----------------------------------------------------------
# ---------------------------
# 0️⃣ Extract barrier
# ---------------------------
analysis_df['barrier'] = analysis_df['experiment'].str.extract(r'(\d+cm)')

# Ensure condition order
analysis_df['condition'] = pd.Categorical(
    analysis_df['condition'],
    categories=['Control', 'CNSDS'],
    ordered=True
)

palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35
barriers = ['10cm', '15cm', '20cm']

# ---------------------------
# 1️⃣ Average across 3 days per subject per barrier
# ---------------------------
subject_barrier_avg = (
    analysis_df
    .groupby(['barrier', 'subject', 'condition'])['avg_start_box_time_sec']
    .mean()
    .reset_index()
)

# ---------------------------
# 2️⃣ ANOVA
# ---------------------------
model = ols(
    'avg_start_box_time_sec ~ C(barrier) * C(condition)',
    data=subject_barrier_avg
).fit()

anova_table = anova_lm(model, typ=2)
print("ANOVA Results:")
print(anova_table)

# ---------------------------
# 3️⃣ Tukey HSD
# ---------------------------
subject_barrier_avg = subject_barrier_avg.dropna(
    subset=['avg_start_box_time_sec']
)

subject_barrier_avg['group'] = (
    subject_barrier_avg['barrier'].astype(str) + "_" +
    subject_barrier_avg['condition'].astype(str)
)

tukey = pairwise_tukeyhsd(
    endog=subject_barrier_avg['avg_start_box_time_sec'],
    groups=subject_barrier_avg['group'],
    alpha=0.05
)

print("\nTukey Results:")
print(tukey.summary())

tukey_df = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

# ---------------------------
# 4️⃣ Compute means + SEM
# ---------------------------
summary = (
    subject_barrier_avg
    .groupby(['barrier', 'condition'])['avg_start_box_time_sec']
    .agg(['mean', 'sem'])
    .reset_index()
)

# ---------------------------
# 5️⃣ Plot
# ---------------------------
plt.figure(figsize=(6, 4))
x = range(len(barriers))

for i, cond in enumerate(['Control', 'CNSDS']):
    for j, barrier in enumerate(barriers):
        row = summary[
            (summary['barrier'] == barrier) &
            (summary['condition'] == cond)
        ]
        if not row.empty:
            y = row['mean'].values[0]
            yerr = row['sem'].values[0]
            xpos = j - bar_width/2 if cond == 'Control' else j + bar_width/2

            plt.bar(
                xpos, y, width=bar_width, yerr=yerr,
                facecolor='white', edgecolor=palette[cond],
                linewidth=2, capsize=5
            )

# ---------------------------
# 6️⃣ Overlay subject points
# ---------------------------
sns.stripplot(
    data=subject_barrier_avg,
    x='barrier',
    y='avg_start_box_time_sec',
    hue='condition',
    dodge=True,
    palette=palette,
    jitter=True,
    alpha=0.7
)

# NOTE (conversion): legend added on request.
plt.legend(title='Condition', frameon=False)

# ---------------------------
# 7️⃣ Add Tukey-based asterisks
# ---------------------------
y_max = subject_barrier_avg['avg_start_box_time_sec'].max()
offset = 0.05 * y_max

for i, barrier in enumerate(barriers):
    g1 = f"{barrier}_Control"
    g2 = f"{barrier}_CNSDS"

    match = tukey_df[
        ((tukey_df['group1'] == g1) & (tukey_df['group2'] == g2)) |
        ((tukey_df['group1'] == g2) & (tukey_df['group2'] == g1))
    ]

    if not match.empty and match['reject'].values[0]:
        vals1 = subject_barrier_avg[
            (subject_barrier_avg['barrier'] == barrier) &
            (subject_barrier_avg['condition'] == 'Control')
        ]['avg_start_box_time_sec']

        vals2 = subject_barrier_avg[
            (subject_barrier_avg['barrier'] == barrier) &
            (subject_barrier_avg['condition'] == 'CNSDS')
        ]['avg_start_box_time_sec']

        # Print stats
        n1, n2 = len(vals1), len(vals2)
        t_stat = (vals1.mean() - vals2.mean()) / np.sqrt(vals1.var()/n1 + vals2.var()/n2)
        p_val = match['p-adj'].values[0]

        print(f"{barrier}: t={t_stat:.2f}, n_control={n1}, n_CNSDS={n2}, p={p_val:.4f}")

        max_y = max(vals1.max(), vals2.max())
        plt.text(i, max_y + offset, '*', ha='center', fontsize=SIG_STAR_FONTSIZE)

# ---------------------------
# 8️⃣ Final formatting
# ---------------------------
plt.xticks(ticks=range(len(barriers)), labels=barriers, fontsize=16)
plt.ylabel('Start Box Time (s)', fontsize=20)
plt.xlabel('Barrier', fontsize=20)
sns.despine()

plt.tight_layout()
plt.savefig(r"c:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\startbox_time.pdf", format="pdf")
plt.show()


# ==========================================================================
# TOTAL DISTANCE TRAVELED
# ==========================================================================

# ---- notebook cell 82 ----------------------------------------------------------
# 1. Function to calculate total distance per trial
def calculate_total_distance_per_trial(data):
    if len(data) > 1:
        diff_x = np.diff(data['normalized_x'])
        diff_y = np.diff(data['normalized_y'])
        distances = np.sqrt(diff_x**2 + diff_y**2)
        return np.sum(distances)
    return 0

# 2. Calculate total distance per trial
total_distance_per_trial = (
    pts.groupby(['experiment', 'subject', 'trial'])
    .apply(calculate_total_distance_per_trial)
    .reset_index(name='total_distance')
)

# 3. Average distance per experiment and subject
avg_distance_per_subject = (
    total_distance_per_trial
    .groupby(['experiment', 'subject'])['total_distance']
    .mean()
    .reset_index()
    .rename(columns={'total_distance': 'avg_total_distance'})
)

# # ✅ If 'experiment' column isn't in analysis_df yet, create it from 'experiment_name'
# if 'experiment' not in analysis_df.columns and 'experiment' in analysis_df.columns:
#     analysis_df['experiment'] = analysis_df['experiment']

# 4. Merge based on 'experiment' and 'subject'
analysis_df = analysis_df.merge(
    avg_distance_per_subject,
    on=['experiment', 'subject'],
    how='left'
)

# ✅ Final output
console(analysis_df[['experiment', 'subject', 'avg_total_distance']].head())

# ---- notebook cell 83 ----------------------------------------------------------
# ---------------------------
# 0️⃣ Define conversion factors (pixels → cm)
# ---------------------------
scale_x = 7.62 / 230
scale_y = 38.735 / 885

# ---------------------------
# 1️⃣ Function: distance per trial (in cm)
# ---------------------------
def calculate_total_distance_per_trial(data):
    if len(data) > 1:
        diff_x = np.diff(data['normalized_x']) * scale_x
        diff_y = np.diff(data['normalized_y']) * scale_y
        distances = np.sqrt(diff_x**2 + diff_y**2)
        return np.sum(distances)
    return 0

# ---------------------------
# 2️⃣ Total distance per trial
# ---------------------------
total_distance_per_trial = (
    pts.groupby(['experiment', 'subject', 'trial'])
    .apply(calculate_total_distance_per_trial)
    .reset_index(name='total_distance_cm')
)

# ---------------------------
# 3️⃣ Average per day (experiment)
# ---------------------------
avg_distance_per_day = (
    total_distance_per_trial
    .groupby(['experiment', 'subject'])['total_distance_cm']
    .mean()
    .reset_index()
)

# ---------------------------
# 4️⃣ Extract barrier
# ---------------------------
avg_distance_per_day['barrier'] = avg_distance_per_day['experiment'].str.extract(r'(\d+cm)')

# ---------------------------
# 5️⃣ Average across 3 days → per subject per barrier
# ---------------------------
subject_barrier_avg = (
    avg_distance_per_day
    .groupby(['barrier', 'subject'])['total_distance_cm']
    .mean()
    .reset_index()
    .rename(columns={'total_distance_cm': 'avg_total_distance_cm'})
)

# ---------------------------
# 6️⃣ Add condition info
# ---------------------------
subject_conditions = (
    analysis_df[['experiment', 'subject', 'condition']]
    .drop_duplicates()
)

# Extract barrier in analysis_df too
analysis_df['barrier'] = analysis_df['experiment'].str.extract(r'(\d+cm)')

subject_conditions['barrier'] = subject_conditions['experiment'].str.extract(r'(\d+cm)')

# Merge condition onto subject_barrier_avg
subject_barrier_avg = subject_barrier_avg.merge(
    subject_conditions[['barrier', 'subject', 'condition']].drop_duplicates(),
    on=['barrier', 'subject'],
    how='left'
)

# NOTE (conversion): the notebook computed avg_total_distance_cm here but
# never merged it onto analysis_df, so the stress-correlation cells below
# had no cm distance column to read. Merging it on now.
analysis_df = analysis_df.merge(
    subject_barrier_avg[['barrier', 'subject', 'avg_total_distance_cm']],
    on=['barrier', 'subject'],
    how='left'
)

# ---------------------------
# 7️⃣ ANOVA
# ---------------------------
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols(
    'avg_total_distance_cm ~ C(barrier) * C(condition)',
    data=subject_barrier_avg
).fit()

anova_table = anova_lm(model, typ=2)

print("ANOVA Results (Total Distance):")
print(anova_table)

# ---------------------------
# 8️⃣ Tukey HSD
# ---------------------------
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Drop NaNs (critical!)
subject_barrier_avg = subject_barrier_avg.dropna(subset=['avg_total_distance_cm'])

subject_barrier_avg['group'] = (
    subject_barrier_avg['barrier'].astype(str) + "_" +
    subject_barrier_avg['condition'].astype(str)
)

tukey = pairwise_tukeyhsd(
    endog=subject_barrier_avg['avg_total_distance_cm'],
    groups=subject_barrier_avg['group'],
    alpha=0.05
)

print("\nTukey HSD Results:")
print(tukey.summary())

# ---- notebook cell 84 ----------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure condition order
subject_barrier_avg['condition'] = pd.Categorical(
    subject_barrier_avg['condition'],
    categories=['Control', 'CNSDS'],
    ordered=True
)

palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35
barriers = ['10cm', '15cm', '20cm']

# ---------------------------
# 1️⃣ Summary stats
# ---------------------------
summary = (
    subject_barrier_avg
    .groupby(['barrier', 'condition'])['avg_total_distance_cm']
    .agg(['mean', 'sem'])
    .reset_index()
)

# ---------------------------
# 2️⃣ Plot bars
# ---------------------------
plt.figure(figsize=(6, 4))
x = range(len(barriers))

for i, cond in enumerate(['Control', 'CNSDS']):
    for j, barrier in enumerate(barriers):
        row = summary[
            (summary['barrier'] == barrier) &
            (summary['condition'] == cond)
        ]
        if not row.empty:
            y = row['mean'].values[0]
            yerr = row['sem'].values[0]
            xpos = j - bar_width/2 if cond == 'Control' else j + bar_width/2

            plt.bar(
                xpos, y, width=bar_width, yerr=yerr,
                facecolor='white',
                edgecolor=palette[cond],
                linewidth=2,
                capsize=5,
                zorder=2
            )

# ---------------------------
# 3️⃣ Overlay subject points
# ---------------------------
sns.stripplot(
    data=subject_barrier_avg,
    x='barrier',
    y='avg_total_distance_cm',
    hue='condition',
    dodge=True,
    palette=palette,
    jitter=True,
    alpha=0.7,
    zorder=3
)

plt.legend().remove()

# ---------------------------
# 4️⃣ Tukey asterisks
# ---------------------------
tukey_df = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

y_max = subject_barrier_avg['avg_total_distance_cm'].max()
offset = 0.05 * y_max

for i, barrier in enumerate(barriers):
    g1 = f"{barrier}_Control"
    g2 = f"{barrier}_CNSDS"

    match = tukey_df[
        ((tukey_df['group1'] == g1) & (tukey_df['group2'] == g2)) |
        ((tukey_df['group1'] == g2) & (tukey_df['group2'] == g1))
    ]

    if not match.empty and match['reject'].values[0]:
        vals1 = subject_barrier_avg[
            (subject_barrier_avg['barrier'] == barrier) &
            (subject_barrier_avg['condition'] == 'Control')
        ]['avg_total_distance_cm']

        vals2 = subject_barrier_avg[
            (subject_barrier_avg['barrier'] == barrier) &
            (subject_barrier_avg['condition'] == 'CNSDS')
        ]['avg_total_distance_cm']

        # Print stats
        n1, n2 = len(vals1), len(vals2)
        t_stat = (vals1.mean() - vals2.mean()) / np.sqrt(vals1.var()/n1 + vals2.var()/n2)
        p_val = match['p-adj'].values[0]

        print(f"{barrier}: t={t_stat:.2f}, n_control={n1}, n_CNSDS={n2}, p={p_val:.4f}")

        max_y = max(vals1.max(), vals2.max())
        plt.text(i, max_y + offset, '*', ha='center', fontsize=SIG_STAR_FONTSIZE)

# ---------------------------
# 5️⃣ Formatting
# ---------------------------
plt.xticks(ticks=range(len(barriers)), labels=barriers, fontsize=16)
plt.ylabel('Total Distance (cm)', fontsize=20)
plt.xlabel('Barrier', fontsize=20)
sns.despine()

plt.tight_layout()
plt.savefig("total_distance.pdf", format="pdf")
plt.show()

# ---- notebook cell 85 ----------------------------------------------------------
# # Define date cohorts
# dates_to_analyze = [
#     ['31524', '31824', '32124'],   # Cohort 1
#     ['50524', '50824', '51124'],   # Cohort 2
#     ['62624', '62924', '70224']    # Cohort 3
# ]

# # Plot setup: 2x2 grid (3 used)
# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# axes = axes.flatten()  # To index 0, 1, 2

# palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
# bar_width = 0.35

# for idx, cohort_dates in enumerate(dates_to_analyze):
#     ax = axes[idx]
    
#     cohort_df = analysis_df[analysis_df['date'].isin(cohort_dates)]
#     experiments = cohort_df['experiment'].unique()
    
#     # Group summary
#     summary = (
#         cohort_df
#         .groupby(['experiment', 'condition'])['total_distance_cm']
#         .agg(['mean', 'sem'])
#         .reset_index()
#     )

#     # ---- Run Welch's t-tests ---- #
#     results = []
#     for exp in experiments:
#         subset = cohort_df[cohort_df['experiment'] == exp]
#         control = subset[subset['condition'] == 'Control']['total_distance_cm']
#         cnsds = subset[subset['condition'] == 'CNSDS']['total_distance_cm']

#         if len(control) > 1 and len(cnsds) > 1:
#             t_stat, p = ttest_ind(control, cnsds, equal_var=False)
#             s1_sq, s2_sq = np.var(control, ddof=1), np.var(cnsds, ddof=1)
#             n1, n2 = len(control), len(cnsds)
#             df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq**2)/(n1**2*(n1-1)) + (s2_sq**2)/(n2**2*(n2-1)))

#             results.append({
#                 'experiment': exp,
#                 'p_raw': p,
#                 'mean_control': np.mean(control),
#                 'mean_cnsds': np.mean(cnsds)
#             })

#     # ---- Bonferroni correction ---- #
#     if results:
#         p_vals = [r['p_raw'] for r in results]
#         reject, corrected_pvals, _, _ = smm.multipletests(p_vals, method='bonferroni')
#         for i, r in enumerate(results):
#             r['p_corrected'] = corrected_pvals[i]
#             r['significant'] = reject[i]
#     # 🖨️ Print statistical results
#         print(f"\n📊 Cohort {idx + 1} - Dates: {cohort_dates}")
#         for r in results:
#             print(f"  Experiment: {r['experiment']}")
#             print(f"    Mean (Control): {r['mean_control']:.2f}")
#             print(f"    Mean (CNSDS):   {r['mean_cnsds']:.2f}")
#             print(f"    Raw p-value:    {r['p_raw']:.4f}")
#             print(f"    Corrected p:    {r['p_corrected']:.4f}")
#             print(f"    Significant?:   {'Yes' if r['significant'] else 'No'}")
#         print("-" * 50)

#     # ---- Plot bars ---- #
#     for i, exp in enumerate(experiments):
#         for cond in ['Control', 'CNSDS']:
#             row = summary[(summary['experiment'] == exp) & (summary['condition'] == cond)]
#             if not row.empty:
#                 y = row['mean'].values[0]
#                 yerr = row['sem'].values[0]
#                 xpos = i - bar_width / 2 if cond == 'Control' else i + bar_width / 2
#                 ax.bar(
#                     xpos, y, width=bar_width, yerr=yerr,
#                     facecolor='white', edgecolor=palette[cond], linewidth=2, capsize=4
#                 )

#     # ---- Plot individual data points ---- #
#     sns.stripplot(
#         data=cohort_df,
#         x='experiment',
#         y='total_distance_cm',
#         hue='condition',
#         palette=palette,
#         dodge=True,
#         jitter=True,
#         alpha=0.6,
#         ax=ax
#     )

#     # ---- Asterisk for significance ---- #
#     y_max = cohort_df['total_distance_cm'].max()
#     offset = 0.05 * y_max

#     for i, r in enumerate(results):
#         if r['significant']:
#             max_y = max(r['mean_control'], r['mean_cnsds'])
#             ax.text(i, max_y + offset, '*', ha='center', va='bottom', fontsize=SIG_STAR_FONTSIZE, color='black')

#     ax.set_title(f'Cohort {idx+1}', fontsize=24)
#     ax.set_ylabel("Total Distance (cm)", fontsize = 22)
#     ax.set_xlabel("Barrier", fontsize = 22)
#     ax.set_xticks(range(len(experiments)))
#     ax.set_xticklabels(experiments, fontsize=18)
#     ax.tick_params(axis='y', labelsize=18)
#     #ax.set_ylim(0, y_max + 2 * offset)
#     sns.despine(ax=ax, top=True, right=True)
#     ax.legend(loc='upper right', fontsize=18)



# # Hide unused subplot
# axes[3].axis('off')

# plt.tight_layout()
# plt.savefig("distance_by_cohort.pdf", format="pdf")
# plt.show()


# ==========================================================================
# DECISION FORK TIME
# ==========================================================================

# ---- notebook cell 88 ----------------------------------------------------------
# NOTE (conversion): MOVED (was notebook cell 88, after the plots that use avg_decision_time).
# 1. Filter frames in the decision area
# NOTE (conversion): decision fork region updated. The notebook used
# x -100..100, y 200..600; the bounds now live in DECISION_FORK_X /
# DECISION_FORK_Y at the top of this file so there is one definition.
decision_area_df = pts[
    (pts['normalized_x'] >= DECISION_FORK_X[0]) &
    (pts['normalized_x'] <= DECISION_FORK_X[1]) &
    (pts['normalized_y'] >= DECISION_FORK_Y[0]) &
    (pts['normalized_y'] <= DECISION_FORK_Y[1])
]
console(f'[decision fork] x {DECISION_FORK_X}, y {DECISION_FORK_Y} -> '
        f'{len(decision_area_df)} of {len(pts)} frames')

# 2. Count frames in decision area per trial
decision_counts = decision_area_df.groupby(['experiment', 'subject', 'trial']).size().reset_index(name='decision_frame_count')

# 3. Convert frame count to time (seconds)
decision_counts['decision_time_secs'] = decision_counts['decision_frame_count'] / 60  # assuming 60 fps

# 4. Average time per subject and experiment across trials
avg_decision_time = decision_counts.groupby(['experiment', 'subject'])['decision_time_secs'].mean().reset_index()
avg_decision_time.rename(columns={'decision_time_secs': 'avg_decision_time_secs'}, inplace=True)

# 5. Merge with analysis_df
analysis_df = analysis_df.merge(
    avg_decision_time,
    on=['experiment', 'subject'],
    how='left'
)

# ---- notebook cell 89 ----------------------------------------------------------
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# analysis_df

# ---- notebook cell 90 ----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# # ---------------------------
# # 0️⃣ Clean merged columns
# # ---------------------------
# if 'avg_decision_time_sec_x' in analysis_df.columns:
#     analysis_df = analysis_df.drop(columns=['avg_decision_time_sec_x'])
# if 'avg_decision_time_sec_y' in analysis_df.columns:
#     analysis_df = analysis_df.rename(columns={'avg_decision_time_sec_y': 'avg_decision_time_sec'})

# ---------------------------
# 1️⃣ Extract barrier & ensure condition order
# ---------------------------
# analysis_df['barrier'] = analysis_df['experiment'].str.extract(r'(\d+cm)')
analysis_df['condition'] = pd.Categorical(
    analysis_df['condition'],
    categories=['Control', 'CNSDS'],
    ordered=True
)

palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35
barriers = ['10cm', '15cm', '20cm']

# ---------------------------
# 2️⃣ Average per subject/barrier/condition
# ---------------------------
subject_barrier_avg = (
    analysis_df
    .groupby(['barrier', 'subject', 'condition'], as_index=False)['avg_decision_time_secs']
    .mean()
)

# ---------------------------
# 3️⃣ ANOVA
# ---------------------------
model = ols(
    'avg_decision_time_secs ~ C(barrier) * C(condition)',
    data=subject_barrier_avg
).fit()

anova_table = anova_lm(model, typ=2)
print("ANOVA Results:")
print(anova_table)

# ---------------------------
# 4️⃣ Tukey HSD
# ---------------------------
subject_barrier_avg = subject_barrier_avg.dropna(subset=['avg_decision_time_secs'])
subject_barrier_avg['group'] = (
    subject_barrier_avg['barrier'].astype(str) + "_" +
    subject_barrier_avg['condition'].astype(str)
)

tukey = pairwise_tukeyhsd(
    endog=subject_barrier_avg['avg_decision_time_secs'],
    groups=subject_barrier_avg['group'],
    alpha=0.05
)
print("\nTukey Results:")
print(tukey.summary())

tukey_df = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

# ---------------------------
# 5️⃣ Compute means + SEM for plotting
# ---------------------------
summary = (
    subject_barrier_avg
    .groupby(['barrier', 'condition'])['avg_decision_time_secs']
    .agg(['mean','sem'])
    .reset_index()
)

# ---------------------------
# 6️⃣ Plot
# ---------------------------
plt.figure(figsize=(5,3))

for i, cond in enumerate(['Control','CNSDS']):
    for j, barrier in enumerate(barriers):
        row = summary[(summary['barrier']==barrier) & (summary['condition']==cond)]
        if not row.empty:
            y = row['mean'].values[0]
            yerr = row['sem'].values[0]
            xpos = j - bar_width/2 if cond=='Control' else j + bar_width/2
            plt.bar(
                xpos, y, width=bar_width, yerr=yerr,
                facecolor='white', edgecolor=palette[cond],
                linewidth=1.5, capsize=5
            )

# ---------------------------
# Overlay subject points
# ---------------------------
sns.stripplot(
    data=subject_barrier_avg,
    x='barrier',
    y='avg_decision_time_secs',
    hue='condition',
    dodge=True,
    palette=palette,
    jitter=True,
    alpha=1,
    s=5
)

# NOTE (conversion): legend removed on request.
_lg = plt.gca().get_legend()
if _lg is not None:
    _lg.remove()

# ---------------------------
# 7️⃣ Add Tukey-based asterisks
# ---------------------------
y_max = subject_barrier_avg['avg_decision_time_secs'].max()
offset = 0.05 * y_max

for i, barrier in enumerate(barriers):
    g1 = f"{barrier}_Control"
    g2 = f"{barrier}_CNSDS"

    match = tukey_df[
        ((tukey_df['group1']==g1) & (tukey_df['group2']==g2)) |
        ((tukey_df['group1']==g2) & (tukey_df['group2']==g1))
    ]

    if not match.empty and match['reject'].values[0]:
        vals1 = subject_barrier_avg[(subject_barrier_avg['barrier']==barrier) & (subject_barrier_avg['condition']=='Control')]['avg_decision_time_secs']
        vals2 = subject_barrier_avg[(subject_barrier_avg['barrier']==barrier) & (subject_barrier_avg['condition']=='CNSDS')]['avg_decision_time_secs']

        max_y = max(vals1.max(), vals2.max())
        plt.text(i, max_y + offset, '*', ha='center', fontsize=SIG_STAR_FONTSIZE)

# ---------------------------
# 8️⃣ Final formatting
# ---------------------------
plt.xticks(ticks=range(len(barriers)), labels=barriers, fontsize=12)
plt.ylabel('Decision Fork Time (s)', fontsize=14)
plt.xlabel('Barrier', fontsize=14)
sns.despine()
plt.tight_layout()

# ---------------------------
# 9️⃣ Save PDF
# ---------------------------
plt.savefig(r"C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\decision_time.pdf", format="pdf")
plt.show()


# ==========================================================================
# HEAD ANGLE
# ==========================================================================

# ---- notebook cell 92 ----------------------------------------------------------
# # ---------------------------
# # 0️⃣ Compute normalized head angle (same as before)
# # ---------------------------
# normalized_head_angles = []

# for index, row in pts.iterrows():
#     if row['Nose_likelihood'] > 0.7 and row['Left_Ear_likelihood'] > 0.7:
#         normalized_nose_x = row['Nose_x'] - row['Maze_Center_x']
#         normalized_nose_y = row['Nose_y'] - row['Maze_Center_y']
#         normalized_left_ear_x = row['Left_Ear_x'] - row['Maze_Center_x']
#         normalized_left_ear_y = row['Left_Ear_y'] - row['Maze_Center_y']
#         normalized_right_ear_x = row['Right_Ear_x'] - row['Maze_Center_x']
#         normalized_right_ear_y = row['Right_Ear_y'] - row['Maze_Center_y']

#         ear_x = np.nanmean([normalized_left_ear_x, normalized_right_ear_x])
#         ear_y = np.nanmean([normalized_left_ear_y, normalized_right_ear_y])

#         delta_x = ear_x - normalized_nose_x
#         delta_y = ear_y - normalized_nose_y
#         head_angle = np.arctan2(delta_y, delta_x) - np.pi/2
#     else:
#         head_angle = np.nan

#     angle_deg = (np.degrees(head_angle) + 180) % 360 - 180
#     normalized_head_angles.append(angle_deg)

# pts['normalized_head_angle'] = normalized_head_angles

# # ---------------------------
# # 1️⃣ Average head angle per trial
# # ---------------------------
# head_angle_trial_avg = (
#     pts
#     .groupby(['experiment', 'subject', 'trial'])['normalized_head_angle']
#     .mean()
#     .reset_index(name='head_angle_trial_avg')
# )

# # ---------------------------
# # 2️⃣ Average across days/trials per subject for each barrier
# # ---------------------------
# subject_barrier_avg = (
#     head_angle_trial_avg
#     .groupby(['experiment', 'subject'])['head_angle_trial_avg']
#     .mean()
#     .reset_index()
#     .rename(columns={'head_angle_trial_avg': 'avg_head_angle_deg'})
# )

# # ---------------------------
# # 3️⃣ Merge into analysis_df and extract barrier height
# # ---------------------------
# analysis_df = analysis_df.merge(
#     subject_barrier_avg,
#     on=['experiment', 'subject'],
#     how='left'
# )
# analysis_df['barrier'] = analysis_df['experiment'].str.extract(r'(\d+cm)')

# # Now analysis_df has one row per subject × experiment with avg_head_angle_deg
# # You can then do ANOVA, Tukey, bar plots, etc., exactly like your decision time block.

# ---- notebook cell 95 ----------------------------------------------------------
from matplotlib.transforms import Affine2D

# ---------------------------
# 1️⃣ Ensure analysis_df has averages per subject/barrier
# ---------------------------
# Columns needed: ['experiment','subject','condition','barrier','avg_head_angle_deg']
# Make sure you already computed avg_head_angle_deg as in previous steps

conditions = ['Control', 'CNSDS']

# Create figure with 2 subplots (one per condition)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

for i, condition in enumerate(conditions):
    ax = axes[i]
    
    # Filter data for this condition
    condition_data = analysis_df[analysis_df['condition'] == condition]

    # Display maze frame (replace frame, x_pic, y_pic with actual values)
    height, width, chan = frame.shape
    extent = [-x_pic, width - x_pic, height - y_pic, -y_pic]
    
    angle = 2  # rotation in degrees
    transform = Affine2D().rotate_deg_around(0, 0, angle) + ax.transData
    ax.imshow(frame, extent=extent, origin='upper', transform=transform)
    
    # # Scatter plot using the **averaged head angle per subject**
    # # Here we plot one point per subject at their mean position (normalized_x, normalized_y)
    # # If you don’t have mean X/Y per subject, you can use median across frames
    # subj_positions = (
    #     pts.groupby(['experiment','subject','condition'])
    #     .agg({'normalized_x':'median', 'normalized_y':'median'})
    #     .reset_index()
    # )
    # subj_positions = subj_positions[subj_positions['condition'] == condition]

    # # Merge the avg_head_angle_deg from analysis_df
    # plot_data = subj_positions.merge(
    #     condition_data[['experiment','subject','avg_head_angle_deg']],
    #     on=['experiment','subject'],
    #     how='left'
    # )
    
    # scatter = ax.scatter(
    #     plot_data['normalized_x'],
    #     plot_data['normalized_y'],
    #     c=plot_data['avg_head_angle_deg'],
    #     cmap='viridis',
    #     s=40,
    #     alpha=0.8,
    #     vmin=-180,
    #     vmax=180
    # )
    
    # ax.set_title(f'{condition}', fontsize=16)
    # ax.set_xlabel('X', fontsize=14)
    # ax.set_ylabel('Y', fontsize=14)
    ax.set_xlim(200, -200)
    # ax.set_ylim(800, -200)
    # ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
#cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', pad=0.02)
#cbar.set_label('Average Head Angle (deg)', fontsize=14)

sns.despine(top=True, right=True)
plt.savefig("headangle_locations_avg.pdf", format="pdf")
plt.show()

# ---- notebook cell 96 ----------------------------------------------------------
# # Set number of bins
# bins = 30

# # Define shared bin edges
# angle_min = pts['normalized_head_angle'].min()
# angle_max = pts['normalized_head_angle'].max()
# x_bins = np.linspace(angle_min, angle_max, bins + 1)
# bin_centers = 0.5 * (x_bins[:-1] + x_bins[1:])

# # Function to compute normalized histogram for each trial
# def compute_normalized_histogram(group):
#     counts, _ = np.histogram(group['normalized_head_angle'], bins=x_bins)
#     norm = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
#     return pd.Series(norm, index=range(bins))

# # Compute normalized histogram for each trial
# trial_histograms = pts.groupby(['subject', 'experiment', 'condition']).apply(compute_normalized_histogram).reset_index()
# trial_histograms = trial_histograms.rename(columns={i: f'bin_{i}' for i in range(bins)})

# # Melt to long format
# long_df = trial_histograms.melt(id_vars=['subject', 'experiment', 'condition'],
#                                 var_name='bin', value_name='proportion')
# long_df['bin'] = long_df['bin'].str.extract(r'bin_(\d+)').astype(int)
# long_df['bin_center'] = bin_centers[long_df['bin']]

# # Average per group
# mean_df = long_df.groupby(['condition', 'bin_center'])['proportion'].mean().reset_index()

# # Plot
# plt.figure(figsize=(8, 6))

# # Bar width
# bar_width = (x_bins[1] - x_bins[0]) * 0.9

# # Plot CNSDS
# cnsds_data = mean_df[mean_df['condition'] == 'CNSDS']
# plt.bar(cnsds_data['bin_center'], cnsds_data['proportion'],
#         width=bar_width, color=CNSDS_GRAY, alpha=0.7, label='CNSDS')

# # Plot Control
# control_data = mean_df[mean_df['condition'] == 'Control']
# plt.bar(control_data['bin_center'], control_data['proportion'],
#         width=bar_width, color=CONTROL_GRAY, alpha=0.7, label='Control')

# # Formatting
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.xlabel('Head Angle (degrees)', fontsize=20)
# plt.ylabel('% per Mouse', fontsize=20)
# plt.xlim(-150,150)
# plt.legend()
# #plt.legend(fontsize=13)
# plt.tight_layout()
# sns.despine(top=True, right=True)
# plt.savefig("D:/Figures/_norm_headangle_distribution.pdf", format="pdf")
# plt.show()

# ---- notebook cell 97 ----------------------------------------------------------
# # Number of bins
# num_bins = bins

# # Store p-values
# p_values = []

# # Perform t-tests for each bin
# for b in range(num_bins):
#     bin_data = long_df[long_df['bin'] == b]
#     cnsds_vals = bin_data[bin_data['condition'] == 'CNSDS']['proportion']
#     control_vals = bin_data[bin_data['condition'] == 'Control']['proportion']

#     # T-test
#     t_stat, p = ttest_ind(cnsds_vals, control_vals, equal_var=False, nan_policy='omit')
#     p_values.append(p)

# # Apply Bonferroni correction
# rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

# # Print p-values
# print("Bin\tP-value\t\tCorrected\tSignificant")
# for i, (p, p_corr, sig) in enumerate(zip(p_values, pvals_corrected, rejected)):
#     print(f"{i}\t{p:.4e}\t{p_corr:.4e}\t{sig}")

# # Plot with significance asterisks
# plt.figure(figsize=(8, 6))

# # Bar width
# bar_width = (x_bins[1] - x_bins[0]) * 0.9

# # Plot CNSDS
# cnsds_data = mean_df[mean_df['condition'] == 'CNSDS']
# plt.bar(cnsds_data['bin_center'], cnsds_data['proportion'],
#         width=bar_width, color=CNSDS_GRAY, alpha=0.7, label='CNSDS')

# # Plot Control
# control_data = mean_df[mean_df['condition'] == 'Control']
# plt.bar(control_data['bin_center'], control_data['proportion'],
#         width=bar_width, color=CONTROL_GRAY, alpha=0.7, label='Control')

# # Add asterisks for significant bins
# for i, sig in enumerate(rejected):
#     if sig:
#         x = bin_centers[i]
#         y = max(cnsds_data[cnsds_data['bin_center'] == x]['proportion'].values[0],
#                 control_data[control_data['bin_center'] == x]['proportion'].values[0])
#         plt.text(x, y + 0.01, '*', ha='center', va='bottom', color='black', fontsize=SIG_STAR_FONTSIZE)

# # Formatting
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.xlabel('Head Angle (degrees)', fontsize=20)
# plt.ylabel('% of Trial', fontsize=20)
# plt.xlim(-150, 150)
# plt.legend(fontsize=16)
# plt.tight_layout()
# sns.despine(top=True, right=True)

# # Save and show
# plt.savefig("D:/Figures/_norm_headangle_distribution_with_stats.pdf", format="pdf")
# plt.show()


# ==========================================================================
# TRY TO FLIP THESE BASED ON WHICH SIDE IS HR (YOU'LL HAVE TO USE SIMILAR CODE YOU USED FOR PLOTTING THE X/Y POSITIONS)
# ==========================================================================

# ---- notebook cell 99 ----------------------------------------------------------
# ### TRY TO FLIP THESE BASED ON WHICH SIDE IS HR (YOU'LL HAVE TO USE SIMILAR CODE YOU USED FOR PLOTTING THE X/Y POSITIONS)

# # Convert degrees to radians — retain negative values!
# mean_df['theta'] = np.deg2rad(mean_df['bin_center'])

# # Polar bar width in radians
# bar_width_rad = np.deg2rad((x_bins[1] - x_bins[0]) * 0.9)

# # Set up polar plot
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# # Plot Control
# control_data = mean_df[mean_df['condition'] == 'Control']
# ax.bar(control_data['theta'], control_data['proportion'],
#        width=bar_width_rad, color=CONTROL_GRAY, alpha=0.5, label='Control')

# # Plot CNSDS
# cnsds_data = mean_df[mean_df['condition'] == 'CNSDS']
# ax.bar(cnsds_data['theta'], cnsds_data['proportion'],
#        width=bar_width_rad, color=CNSDS_GRAY, alpha=0.5, label='CNSDS')

# # Fix angle mapping and axis
# ax.set_theta_zero_location('N')  # 0° at top
# ax.set_theta_direction(-1)       # Clockwise
# ax.set_rlabel_position(270)
# ax.set_ylim(0, mean_df['proportion'].max() * 1.1)

# # Custom angle labels for true -180° to 180° display
# tick_locs = np.deg2rad([0, 315, 270, 90, 45])
# tick_labels = ['0°', '-45°', '-90°', '90°', '45°']
# ax.set_xticks(tick_locs)
# ax.set_xticklabels(tick_labels, fontsize = 18)
# #ax.set_xlabel("% per Mouse")
# fig.text(0.5, 0.2, '% per Mouse', ha='center', va='center', fontsize=18)

# ax.set_thetamin(-90)
# ax.set_thetamax(90)

# ax.tick_params(axis='y', labelsize=12)

# # Labels and legend
# #ax.legend(loc='upper right', fontsize=13)
# #plt.title("% per Mouse", fontsize=16)
# plt.tight_layout()
# plt.savefig("D:/Figures/headangle_polar_distribution", format="pdf")
# plt.show()


# ==========================================================================
# CORRELATIONS
# ==========================================================================

# ---- notebook cell 101 ---------------------------------------------------------
# import matplotlib.patches as mpatches

# # Variables and custom labels
# variables = {
#     'avg_trial_duration': 'Trial Duration (s)',
#     'avg_velocity_across_trials': 'Velocity (cm)',
#     'avg_start_box_time_secs': 'Start Box Time (s)',
#     'total_distance_cm': 'Distance (cm)',
#     'avg_decision_time_secs': 'Decision Fork Time (s)'
# }

# # Drop rows with missing data
# clean_df = analysis_df.dropna(subset=list(variables.keys()) + ['HR_ratio', 'condition', 'experiment'])

# # Get list of unique experiments
# experiments = clean_df['experiment'].unique()
# n_experiments = len(experiments)
# n_vars = len(variables)

# # Create subplots: one row per experiment, one column per variable
# fig, axes = plt.subplots(nrows=n_experiments, ncols=n_vars, figsize=(5 * n_vars, 4 * n_experiments), sharey=True)

# # Handle single-row case (flatten if necessary)
# if n_experiments == 1:
#     axes = [axes]

# for i, experiment in enumerate(experiments):
#     exp_df = clean_df[clean_df['experiment'] == experiment]
#     for j, (var, label) in enumerate(variables.items()):
#         ax = axes[i][j] if n_experiments > 1 else axes[0][j]

#         # Plot Control group (no label here)
#         sns.regplot(
#             data=exp_df[exp_df['condition'] == 'Control'],
#             x=var, y='HR_ratio',
#             color=CONTROL_GRAY, ax=ax,
#             scatter_kws={'s': 25},
#             ci=68  # ±1 standard error band
#         )

#         # Plot CNSDS group (no label here)
#         sns.regplot(
#             data=exp_df[exp_df['condition'] == 'CNSDS'],
#             x=var, y='HR_ratio',
#             color=CNSDS_GRAY, ax=ax,
#             scatter_kws={'s': 25},
#             ci=68  # ±1 standard error band
#         )

#         # Annotate significant correlations with asterisk
#         for cond, color in [('Control', CONTROL_GRAY), ('CNSDS', CNSDS_GRAY)]:
#             subset = exp_df[exp_df['condition'] == cond]
#             if len(subset) > 2:
#                 r, p = pearsonr(subset[var], subset['HR_ratio'])
#                 if p < 0.05:
#                     x_pos = subset[var].max()
#                     y_pos = subset['HR_ratio'].max() - 0.3
#                     ax.text(x_pos, y_pos, '*', color=color, fontsize=SIG_STAR_FONTSIZE,
#                             ha='right', va='top', fontweight='bold')

#                 # Print correlation results
#                 print(f"Experiment: {experiment} | Condition: {cond} | Variable: {label} | r = {r:.3f}, p = {p:.4f}")

#         # Title and axis labels
#         #ax.set_title(f"{experiment} | {label}", fontsize=25)
#         ax.set_xlabel(label, fontsize=25)
#         if j == 0:
#             ax.set_ylabel("HR Ratio", fontsize=25)
#         else:
#             ax.set_ylabel("")
#         ax.tick_params(axis='both', which='major', labelsize=20)

# # Create custom legend handles
# control_patch = mpatches.Patch(color=CONTROL_GRAY, label='Control')
# cnsds_patch = mpatches.Patch(color=CNSDS_GRAY, label='CNSDS')

# # Place legend outside to the right, adjust bbox_to_anchor to move closer/farther
# fig.legend(handles=[control_patch, cnsds_patch],
#            loc='upper left',
#            bbox_to_anchor=(.85, 1),
#            fontsize=18,
#            title_fontsize=20)

# plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legend

# plt.savefig("correlations_by_experiment.pdf", format="pdf")
# plt.show()

# ---- notebook cell 102 ---------------------------------------------------------
behavior_avg = (
    analysis_df
    .groupby(['subject', 'condition', 'barrier'], as_index=False)
    .mean(numeric_only=True)
)

# ---- notebook cell 103 ---------------------------------------------------------
# ── 1. Build avg duration df ──────────────────────────────────────────────────
avg_duration_df = (
    summary_df
    .dropna(subset=['subject', 'choice', 'trial_duration', 'condition'])
    .groupby(['subject', 'choice', 'condition'])['trial_duration']
    .mean()
    .reset_index()
    .rename(columns={'trial_duration': 'avg_trial_duration'})
)

# ── 2. Prep plot df ───────────────────────────────────────────────────────────
plot_df2 = avg_duration_df.copy()
plot_df2['choice'] = plot_df2['choice'].astype(str).str.upper()
plot_df2 = plot_df2[plot_df2['choice'].isin(['TRUE', 'FALSE'])].dropna(subset=['avg_trial_duration'])
plot_df2['choice_label'] = plot_df2['choice'].map({'FALSE': 'LR', 'TRUE': 'HR'})
plot_df2['condition'] = pd.Categorical(plot_df2['condition'], categories=['Control', 'CNSDS'], ordered=True)

palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35
choices = ['LR', 'HR']

summary = (
    plot_df2
    .groupby(['choice_label', 'condition'])['avg_trial_duration']
    .agg(['mean', 'sem'])
    .reset_index()
)

# ── 3. Statistics ─────────────────────────────────────────────────────────────
p_within_choice = {}
p_within_cond = {}

print("=== Control vs CNSDS within each choice ===")
for choice in choices:
    ctrl  = plot_df2[(plot_df2['choice_label']==choice) & (plot_df2['condition']=='Control')]['avg_trial_duration'].dropna()
    cnsds = plot_df2[(plot_df2['choice_label']==choice) & (plot_df2['condition']=='CNSDS')]['avg_trial_duration'].dropna()
    _, p = stats.ttest_ind(ctrl, cnsds)
    p_within_choice[choice] = p
    print(f"  {choice}: n_Control={len(ctrl)}, n_CNSDS={len(cnsds)}, p={p:.4f}")

print("\n=== LR vs HR within each condition ===")
for cond in ['Control', 'CNSDS']:
    lr = plot_df2[(plot_df2['choice_label']=='LR') & (plot_df2['condition']==cond)]['avg_trial_duration'].dropna()
    hr = plot_df2[(plot_df2['choice_label']=='HR') & (plot_df2['condition']==cond)]['avg_trial_duration'].dropna()
    _, p = stats.ttest_ind(lr, hr)
    p_within_cond[cond] = p
    print(f"  {cond}: n_LR={len(lr)}, n_HR={len(hr)}, p={p:.4f}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
def sig_label(p):
    # NOTE (conversion): one asterisk for significance, to match the other
    # figures - no ** / *** tiers.
    return '*' if p < 0.05 else 'ns'

def add_sig_bracket(ax, x1, x2, y, p, h):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='black', linewidth=1)
    # NOTE (conversion): the asterisk sits lower than 'ns' - a 24pt star
    # needs less clearance above the bracket than the small 'ns' does.
    _label = sig_label(p)
    _is_star = _label == '*'
    ax.text((x1+x2)/2, y + h * (0.02 if _is_star else 0.55), _label,
            ha='center', va='bottom',
            fontsize=SIG_STAR_FONTSIZE if _is_star else SIG_NS_FONTSIZE)

fig, ax = plt.subplots(figsize=(6, 5))

# bars
for cond in ['Control', 'CNSDS']:
    for j, choice in enumerate(choices):
        row = summary[(summary['choice_label']==choice) & (summary['condition']==cond)]
        if not row.empty:
            xpos = j - bar_width/2 if cond == 'Control' else j + bar_width/2
            ax.bar(xpos, row['mean'].values[0], width=bar_width, yerr=row['sem'].values[0],
                   facecolor='white', edgecolor=palette[cond], linewidth=2, capsize=5, zorder=2)

# scatter points
sns.stripplot(
    data=plot_df2, x='choice_label', y='avg_trial_duration',
    hue='condition', order=choices, hue_order=['Control', 'CNSDS'],
    dodge=True, palette=palette, jitter=True, alpha=0.7, zorder=3, ax=ax
)

y_max = plot_df2['avg_trial_duration'].max()
offset = 0.05 * y_max
h = offset * 0.4

# brackets: Control vs CNSDS within each choice
for j, choice in enumerate(choices):
    ctrl  = plot_df2[(plot_df2['choice_label']==choice) & (plot_df2['condition']=='Control')]['avg_trial_duration'].dropna()
    cnsds = plot_df2[(plot_df2['choice_label']==choice) & (plot_df2['condition']=='CNSDS')]['avg_trial_duration'].dropna()
    y = max(ctrl.max(), cnsds.max()) + offset * 0.5
    add_sig_bracket(ax, j - bar_width/2, j + bar_width/2, y, p_within_choice[choice], h)

# brackets: LR vs HR within each condition
for cond in ['Control', 'CNSDS']:
    sign = -1 if cond == 'Control' else 1
    x1 = 0 + sign * bar_width/2  # LR bar for this condition
    x2 = 1 + sign * bar_width/2  # HR bar for this condition
    lr = plot_df2[(plot_df2['choice_label']=='LR') & (plot_df2['condition']==cond)]['avg_trial_duration'].dropna()
    hr = plot_df2[(plot_df2['choice_label']=='HR') & (plot_df2['condition']==cond)]['avg_trial_duration'].dropna()
    y = y_max * 1.2 if cond == 'Control' else y_max * 1.3
    add_sig_bracket(ax, x1, x2, y, p_within_cond[cond], h)

# NOTE (conversion): legend removed on request.
_lg = ax.get_legend()
if _lg is not None:
    _lg.remove()
ax.set_xticks(range(len(choices)))
ax.set_xticklabels(choices, fontsize=16)
ax.set_ylabel('Trial Duration (s)', fontsize=20)
ax.set_xlabel('Choice', fontsize=20)
sns.despine()
plt.tight_layout()
plt.savefig(r"C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\trial_duration_by_choice.pdf", format="pdf")    
plt.show()

# ---- notebook cell 104 ---------------------------------------------------------
plot_df2 = avg_duration_df.copy()
plot_df2['choice'] = plot_df2['choice'].astype(str).str.upper()
plot_df2 = plot_df2[plot_df2['choice'].isin(['TRUE', 'FALSE'])].dropna(subset=['avg_trial_duration'])
plot_df2['choice_label'] = plot_df2['choice'].map({'FALSE': 'LR', 'TRUE': 'HR'})
plot_df2['condition'] = pd.Categorical(plot_df2['condition'], categories=['Control', 'CNSDS'], ordered=True)

palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35
choices = ['LR', 'HR']

summary = (
    plot_df2
    .groupby(['choice_label', 'condition'])['avg_trial_duration']
    .agg(['mean', 'sem'])
    .reset_index()
)

plt.figure(figsize=(6, 4))

for cond in ['Control', 'CNSDS']:
    for j, choice in enumerate(choices):
        row = summary[(summary['choice_label'] == choice) & (summary['condition'] == cond)]
        if not row.empty:
            xpos = j - bar_width/2 if cond == 'Control' else j + bar_width/2
            plt.bar(xpos, row['mean'].values[0], width=bar_width, yerr=row['sem'].values[0],
                    facecolor='white', edgecolor=palette[cond], linewidth=2, capsize=5, zorder=2)

sns.stripplot(
    data=plot_df2, x='choice_label', y='avg_trial_duration',
    hue='condition', order=choices, hue_order=['Control', 'CNSDS'],
    dodge=True, palette=palette, jitter=True, alpha=0.7, zorder=3
)

# NOTE (conversion): legend removed on request.
_lg = plt.gca().get_legend()
if _lg is not None:
    _lg.remove()

def sig_label(p):
    # NOTE (conversion): one asterisk for significance, to match the other
    # figures - no ** / *** tiers.
    return '*' if p < 0.05 else 'ns'

ax = plt.gca()
y_max = plot_df2['avg_trial_duration'].max()
offset = 0.05 * y_max

def add_sig_bracket(ax, x1, x2, y, p):
    h = offset * 0.4
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='black', linewidth=1)
    # NOTE (conversion): the asterisk sits lower than 'ns' - a 24pt star
    # needs less clearance above the bracket than the small 'ns' does.
    _label = sig_label(p)
    _is_star = _label == '*'
    ax.text((x1+x2)/2, y + h * (0.02 if _is_star else 0.55), _label,
            ha='center', va='bottom',
            fontsize=SIG_STAR_FONTSIZE if _is_star else SIG_NS_FONTSIZE)

for j, choice in enumerate(choices):
    ctrl = plot_df2[(plot_df2['choice_label']==choice)&(plot_df2['condition']=='Control')]['avg_trial_duration'].dropna()
    cnsds = plot_df2[(plot_df2['choice_label']==choice)&(plot_df2['condition']=='CNSDS')]['avg_trial_duration'].dropna()
    _, p = stats.f_oneway(ctrl, cnsds)
    add_sig_bracket(ax, j - bar_width/2, j + bar_width/2, max(ctrl.max(), cnsds.max()) + offset * 0.5, p)

print("=== Within-choice comparisons (Control vs CNSDS) ===")
for j, choice in enumerate(choices):
    ctrl = plot_df2[(plot_df2['choice_label']==choice)&(plot_df2['condition']=='Control')]['avg_trial_duration'].dropna()
    cnsds = plot_df2[(plot_df2['choice_label']==choice)&(plot_df2['condition']=='CNSDS')]['avg_trial_duration'].dropna()
    _, p = stats.ttest_ind(ctrl, cnsds)
    print(f"  {choice}: n_Control={len(ctrl)}, n_CNSDS={len(cnsds)}, p={p:.4f}")

print("\n=== Between-choice comparison (LR vs HR) ===")
lr = plot_df2[plot_df2['choice_label']=='LR']['avg_trial_duration'].dropna()
hr = plot_df2[plot_df2['choice_label']=='HR']['avg_trial_duration'].dropna()
_, p_between = stats.ttest_ind(lr, hr)
print(f"  LR vs HR: n_LR={len(lr)}, n_HR={len(hr)}, p={p_between:.4f}")

lr = plot_df2[plot_df2['choice_label']=='LR']['avg_trial_duration'].dropna()
hr = plot_df2[plot_df2['choice_label']=='HR']['avg_trial_duration'].dropna()
_, p_between = stats.f_oneway(lr, hr)
add_sig_bracket(ax, 0, 1, y_max * 1.15, p_between)

plt.xticks(ticks=range(len(choices)), labels=choices, fontsize=16)
plt.ylabel('Average Trial Duration', fontsize=20)
plt.xlabel('Choice', fontsize=20)
sns.despine()
plt.tight_layout()
plt.show()

# ---- notebook cell 105 ---------------------------------------------------------
summary_df['trial'] = summary_df['trial'].astype(str)

# ── 1. Aggregate start_box_time_sec per subject, trial, experiment ────────────
merged_agg = (
    merged
    .groupby(['subject', 'trial', 'experiment'])['start_box_time_sec']
    .sum()
    .reset_index()
    .rename(columns={'experiment': 'experiment_name'})
)

# NOTE (conversion): this cast was the notebook cell's first line, which ran
# before merged_agg existed. Moved below the assignment that creates it.
merged_agg['trial'] = merged_agg['trial'].astype(str)

# ── 2. Merge with summary_df to get choice, condition per trial ───────────────
trial_df = merged_agg.merge(
    summary_df[['subject', 'trial', 'experiment_name', 'choice', 'condition', 'trial_duration']],
    on=['subject', 'trial', 'experiment_name'],
    how='inner'
)

# ── 3. Average start_box_time_sec per subject, choice, condition ──────────────
avg_start_box = (
    trial_df
    .dropna(subset=['subject', 'choice', 'condition'])
    .groupby(['subject', 'choice', 'condition'])['start_box_time_sec']
    .mean()
    .reset_index()
    .rename(columns={'start_box_time_sec': 'avg_start_box_time_sec'})
)

# ── 4. Merge with avg_duration_df ─────────────────────────────────────────────
avg_duration_df = avg_duration_df.merge(
    avg_start_box,
    on=['subject', 'choice', 'condition'],
    how='left'
)

# ---- notebook cell 106 ---------------------------------------------------------
plot_df3 = avg_duration_df.copy()
plot_df3['choice'] = plot_df3['choice'].astype(str).str.upper()
plot_df3 = plot_df3[plot_df3['choice'].isin(['TRUE', 'FALSE'])].dropna(subset=['avg_start_box_time_sec'])
plot_df3['choice_label'] = plot_df3['choice'].map({'FALSE': 'LR', 'TRUE': 'HR'})
plot_df3['condition'] = pd.Categorical(plot_df3['condition'], categories=['Control', 'CNSDS'], ordered=True)

palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35
choices = ['LR', 'HR']

summary = (
    plot_df3
    .groupby(['choice_label', 'condition'])['avg_start_box_time_sec']
    .agg(['mean', 'sem'])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(6, 4))

for cond in ['Control', 'CNSDS']:
    for j, choice in enumerate(choices):
        row = summary[(summary['choice_label'] == choice) & (summary['condition'] == cond)]
        if not row.empty:
            xpos = j - bar_width/2 if cond == 'Control' else j + bar_width/2
            ax.bar(xpos, row['mean'].values[0], width=bar_width, yerr=row['sem'].values[0],
                   facecolor='white', edgecolor=palette[cond], linewidth=2, capsize=5, zorder=2)

sns.stripplot(
    data=plot_df3, x='choice_label', y='avg_start_box_time_sec',
    hue='condition', order=choices, hue_order=['Control', 'CNSDS'],
    dodge=True, palette=palette, jitter=True, alpha=0.7, zorder=3, ax=ax
)

# NOTE (conversion): legend removed on request.
_lg = ax.get_legend()
if _lg is not None:
    _lg.remove()

def sig_label(p):
    # NOTE (conversion): one asterisk for significance, to match the other
    # figures - no ** / *** tiers.
    return '*' if p < 0.05 else 'ns'

y_max = plot_df3['avg_start_box_time_sec'].max()
offset = 0.05 * y_max

def add_sig_bracket(ax, x1, x2, y, p):
    h = offset * 0.4
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='black', linewidth=1)
    # NOTE (conversion): the asterisk sits lower than 'ns' - a 24pt star
    # needs less clearance above the bracket than the small 'ns' does.
    _label = sig_label(p)
    _is_star = _label == '*'
    ax.text((x1+x2)/2, y + h * (0.02 if _is_star else 0.55), _label,
            ha='center', va='bottom',
            fontsize=SIG_STAR_FONTSIZE if _is_star else SIG_NS_FONTSIZE)

print("=== Control vs CNSDS within each choice ===")
for j, choice in enumerate(choices):
    ctrl  = plot_df3[(plot_df3['choice_label']==choice)&(plot_df3['condition']=='Control')]['avg_start_box_time_sec'].dropna()
    cnsds = plot_df3[(plot_df3['choice_label']==choice)&(plot_df3['condition']=='CNSDS')]['avg_start_box_time_sec'].dropna()
    _, p = stats.ttest_ind(ctrl, cnsds)
    print(f"  {choice}: n_Control={len(ctrl)}, n_CNSDS={len(cnsds)}, p={p:.4f}")
    add_sig_bracket(ax, j - bar_width/2, j + bar_width/2, max(ctrl.max(), cnsds.max()) + offset * 0.5, p)

print("\n=== LR vs HR within each condition ===")
for cond, sign in [('Control', -1), ('CNSDS', 1)]:
    lr   = plot_df3[(plot_df3['choice_label']=='LR')&(plot_df3['condition']==cond)]['avg_start_box_time_sec'].dropna()
    hr   = plot_df3[(plot_df3['choice_label']=='HR')&(plot_df3['condition']==cond)]['avg_start_box_time_sec'].dropna()
    _, p = stats.ttest_ind(lr, hr)
    print(f"  {cond}: n_LR={len(lr)}, n_HR={len(hr)}, p={p:.4f}")
    x1 = 0 + sign * bar_width/2
    x2 = 1 + sign * bar_width/2
    y  = y_max * 1.15 if cond == 'Control' else y_max * 1.28
    add_sig_bracket(ax, x1, x2, y, p)

ax.set_xticks(range(len(choices)))
ax.set_xticklabels(choices, fontsize=16)
ax.set_ylabel('Start Box Time (s)', fontsize=20)
ax.set_xlabel('Choice', fontsize=20)
sns.despine()
plt.tight_layout()
plt.savefig(r"C:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\start_box_time_by_choice.pdf", format="pdf")
plt.show()

# ---- notebook cell 55 ----------------------------------------------------------
# NOTE (conversion): MOVED (was notebook cell 55, before avg_duration_df existed).
# ── 1. Cast trial to string to avoid type mismatch ────────────────────────────
velocity_df['trial'] = velocity_df['trial'].astype(str)

# ── 2. Merge velocity with summary_df to get choice and condition ─────────────
velocity_trial = velocity_df.merge(
    summary_df[['subject', 'trial', 'experiment_name', 'choice', 'condition']],
    left_on=['subject', 'trial', 'experiment'],
    right_on=['subject', 'trial', 'experiment_name'],
    how='inner'
)

# ── 3. Average velocity per subject, choice, condition ────────────────────────
avg_velocity = (
    velocity_trial
    .dropna(subset=['subject', 'choice', 'condition'])
    .groupby(['subject', 'choice', 'condition'])['mean_velocity_cm']
    .mean()
    .reset_index()
    .rename(columns={'mean_velocity_cm': 'avg_velocity_cm'})
)

# ── 4. Merge into avg_duration_df ─────────────────────────────────────────────
avg_duration_df = avg_duration_df.merge(
    avg_velocity,
    on=['subject', 'choice', 'condition'],
    how='left'
)

# ---- notebook cell 56 ----------------------------------------------------------
# NOTE (conversion): MOVED (was notebook cell 56, before avg_duration_df / sig_label existed).
plot_df4 = avg_duration_df.copy()
plot_df4['choice'] = plot_df4['choice'].astype(str).str.upper()
plot_df4 = plot_df4[plot_df4['choice'].isin(['TRUE', 'FALSE'])].dropna(subset=['avg_velocity_cm'])
plot_df4['choice_label'] = plot_df4['choice'].map({'FALSE': 'LR', 'TRUE': 'HR'})
plot_df4['condition'] = pd.Categorical(plot_df4['condition'], categories=['Control', 'CNSDS'], ordered=True)

palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}
bar_width = 0.35
choices = ['LR', 'HR']

summary = (
    plot_df4
    .groupby(['choice_label', 'condition'])['avg_velocity_cm']
    .agg(['mean', 'sem'])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(6, 4))

for cond in ['Control', 'CNSDS']:
    for j, choice in enumerate(choices):
        row = summary[(summary['choice_label'] == choice) & (summary['condition'] == cond)]
        if not row.empty:
            xpos = j - bar_width/2 if cond == 'Control' else j + bar_width/2
            ax.bar(xpos, row['mean'].values[0], width=bar_width, yerr=row['sem'].values[0],
                   facecolor='white', edgecolor=palette[cond], linewidth=2, capsize=5, zorder=2)

sns.stripplot(
    data=plot_df4, x='choice_label', y='avg_velocity_cm',
    hue='condition', order=choices, hue_order=['Control', 'CNSDS'],
    dodge=True, palette=palette, jitter=True, alpha=0.7, zorder=3, ax=ax
)

# NOTE (conversion): legend removed on request.
_lg = ax.get_legend()
if _lg is not None:
    _lg.remove()

y_max = plot_df4['avg_velocity_cm'].max()
offset = 0.05 * y_max

def add_sig_bracket(ax, x1, x2, y, p):
    h = offset * 0.4
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='black', linewidth=1)
    # NOTE (conversion): the asterisk sits lower than 'ns' - a 24pt star
    # needs less clearance above the bracket than the small 'ns' does.
    _label = sig_label(p)
    _is_star = _label == '*'
    ax.text((x1+x2)/2, y + h * (0.02 if _is_star else 0.55), _label,
            ha='center', va='bottom',
            fontsize=SIG_STAR_FONTSIZE if _is_star else SIG_NS_FONTSIZE)

print("=== Control vs CNSDS within each choice ===")
for j, choice in enumerate(choices):
    ctrl  = plot_df4[(plot_df4['choice_label']==choice)&(plot_df4['condition']=='Control')]['avg_velocity_cm'].dropna()
    cnsds = plot_df4[(plot_df4['choice_label']==choice)&(plot_df4['condition']=='CNSDS')]['avg_velocity_cm'].dropna()
    _, p  = stats.ttest_ind(ctrl, cnsds)
    print(f"  {choice}: n_Control={len(ctrl)}, n_CNSDS={len(cnsds)}, p={p:.4f}")
    add_sig_bracket(ax, j - bar_width/2, j + bar_width/2, max(ctrl.max(), cnsds.max()) + offset * 0.5, p)

print("\n=== LR vs HR within each condition ===")
for cond, sign in [('Control', -1), ('CNSDS', 1)]:
    lr   = plot_df4[(plot_df4['choice_label']=='LR')&(plot_df4['condition']==cond)]['avg_velocity_cm'].dropna()
    hr   = plot_df4[(plot_df4['choice_label']=='HR')&(plot_df4['condition']==cond)]['avg_velocity_cm'].dropna()
    _, p = stats.ttest_ind(lr, hr)
    print(f"  {cond}: n_LR={len(lr)}, n_HR={len(hr)}, p={p:.4f}")
    x1 = 0 + sign * bar_width/2
    x2 = 1 + sign * bar_width/2
    y  = y_max * 1.15 if cond == 'Control' else y_max * 1.28
    add_sig_bracket(ax, x1, x2, y, p)

ax.set_xticks(range(len(choices)))
ax.set_xticklabels(choices, fontsize=16)
# NOTE (conversion): was 'Average Velocity (cm)'; velocity is cm/s, and
# this now matches the by-barrier velocity figure.
ax.set_ylabel('Velocity (cm/s)', fontsize=20)
ax.set_xlabel('Choice', fontsize=20)
sns.despine()
plt.tight_layout()
plt.savefig(r'c:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\Choice_velocity.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.show()

# ---- notebook cell 57 ----------------------------------------------------------
# NOTE (conversion): MOVED (was notebook cell 57, before avg_duration_df existed).
# NOTE (conversion): notebook display echo - it rendered a table in
# Jupyter but is a no-op in a script, and dumping it would bury the
# statistics in the report. Left commented out.
# avg_duration_df

# ---- notebook cell 87 ----------------------------------------------------------
# NOTE (conversion): MOVED (was notebook cell 87, before avg_decision_time / add_sig_bracket existed).
# ── 1. Merge to get choice and condition per trial ────────────────────────────
avg_decision_time['experiment'] = avg_decision_time['experiment'].astype(str)
summary_df['experiment_name'] = summary_df['experiment_name'].astype(str)
summary_df['trial'] = summary_df['trial'].astype(str)

decision_trial = avg_decision_time.merge(
    summary_df[['subject', 'trial', 'experiment_name', 'choice', 'condition']],
    left_on=['subject', 'experiment'],
    right_on=['subject', 'experiment_name'],
    how='inner'
)

# ── 2. Average per subject, choice, condition ─────────────────────────────────
avg_decision = (
    decision_trial
    .dropna(subset=['subject', 'choice', 'condition'])
    .groupby(['subject', 'choice', 'condition'])['avg_decision_time_secs']
    .mean()
    .reset_index()
)

# ── 3. Plot ───────────────────────────────────────────────────────────────────
plot_df5 = avg_decision.copy()
plot_df5['choice'] = plot_df5['choice'].astype(str).str.upper()
plot_df5 = plot_df5[plot_df5['choice'].isin(['TRUE', 'FALSE'])].dropna(subset=['avg_decision_time_secs'])
plot_df5['choice_label'] = plot_df5['choice'].map({'FALSE': 'LR', 'TRUE': 'HR'})
plot_df5['condition'] = pd.Categorical(plot_df5['condition'], categories=['Control', 'CNSDS'], ordered=True)

summary5 = (
    plot_df5
    .groupby(['choice_label', 'condition'])['avg_decision_time_secs']
    .agg(['mean', 'sem'])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(6, 4))

for cond in ['Control', 'CNSDS']:
    for j, choice in enumerate(choices):
        row = summary5[(summary5['choice_label'] == choice) & (summary5['condition'] == cond)]
        if not row.empty:
            xpos = j - bar_width/2 if cond == 'Control' else j + bar_width/2
            ax.bar(xpos, row['mean'].values[0], width=bar_width, yerr=row['sem'].values[0],
                   facecolor='white', edgecolor=palette[cond], linewidth=2, capsize=5, zorder=2)

sns.stripplot(
    data=plot_df5, x='choice_label', y='avg_decision_time_secs',
    hue='condition', order=choices, hue_order=['Control', 'CNSDS'],
    dodge=True, palette=palette, jitter=True, alpha=0.7, zorder=3, ax=ax
)

# NOTE (conversion): legend removed on request.
_lg = ax.get_legend()
if _lg is not None:
    _lg.remove()

y_max = plot_df5['avg_decision_time_secs'].max()
offset = 0.05 * y_max

print("=== Control vs CNSDS within each choice ===")
for j, choice in enumerate(choices):
    ctrl  = plot_df5[(plot_df5['choice_label']==choice)&(plot_df5['condition']=='Control')]['avg_decision_time_secs'].dropna()
    cnsds = plot_df5[(plot_df5['choice_label']==choice)&(plot_df5['condition']=='CNSDS')]['avg_decision_time_secs'].dropna()
    _, p  = stats.ttest_ind(ctrl, cnsds)
    print(f"  {choice}: n_Control={len(ctrl)}, n_CNSDS={len(cnsds)}, p={p:.4f}")
    add_sig_bracket(ax, j - bar_width/2, j + bar_width/2, max(ctrl.max(), cnsds.max()) + offset * 0.5, p)

print("\n=== LR vs HR within each condition ===")
for cond, sign in [('Control', -1), ('CNSDS', 1)]:
    lr   = plot_df5[(plot_df5['choice_label']=='LR')&(plot_df5['condition']==cond)]['avg_decision_time_secs'].dropna()
    hr   = plot_df5[(plot_df5['choice_label']=='HR')&(plot_df5['condition']==cond)]['avg_decision_time_secs'].dropna()
    _, p = stats.ttest_ind(lr, hr)
    print(f"  {cond}: n_LR={len(lr)}, n_HR={len(hr)}, p={p:.4f}")
    x1 = 0 + sign * bar_width/2
    x2 = 1 + sign * bar_width/2
    y  = y_max * 1.15 if cond == 'Control' else y_max * 1.28
    add_sig_bracket(ax, x1, x2, y, p)

ax.set_xticks(range(len(choices)))
ax.set_xticklabels(choices, fontsize=16)
ax.set_ylabel('Decision Fork Time (s)', fontsize=16)
ax.set_xlabel('Choice', fontsize=20)
sns.despine()
plt.tight_layout()
plt.savefig(r"c:\Users\Jillian.Sucher\Documents\Stress_microstructure_testing_day_1\decision_time_by_choice.pdf", format="pdf")
plt.show()

# ---- notebook cell 107 ---------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Predictors and labels
# ---------------------------
predictors = [
    'trial_duration',
    'avg_velocity_across_trials',
    'avg_start_box_time_sec',
    'avg_decision_time_secs'
]

predictor_labels = {
    'trial_duration': 'Trial Duration',
    'avg_velocity_across_trials': 'Avg Velocity',
    'avg_start_box_time_sec': 'Start Box Time',
    'avg_decision_time_secs': 'Decision Fork Time'
}

conditions = ['Control', 'CNSDS']

# ---------------------------
# Containers
# ---------------------------
# NOTE (conversion): models reworked on request.
#   * GEE with an exchangeable working correlation, clustered on `subject`.
#     behavior_avg holds one row per subject x barrier, so each animal appears
#     about three times. OLS treats those rows as independent and understates
#     the standard errors; GEE corrects the inference for that clustering.
#   * Bars stay OLS partial R2, because GEE has no R2. So the figure shows an
#     OLS effect size with GEE-based significance - both are printed below.
#   * Bonferroni across the per-condition coefficient tests, and separately
#     across the interaction terms.
results = []
n_dict = {}
r2_dict = {}

# ---------------------------
# Compute models
# ---------------------------
for cond in conditions:
    df_all = behavior_avg[behavior_avg['condition'] == cond]
    df_c = df_all.dropna(subset=predictors + ['HR_ratio', 'subject']).copy()

    n_dict[cond] = len(df_c)
    if len(df_c) < len(df_all):
        print(f"[{cond}] {len(df_all) - len(df_c)} of {len(df_all)} rows dropped "
              f"for missing predictor values.")

    if df_c.shape[0] < len(predictors) + 2:
        print(f"Skipping condition {cond} (not enough data)")
        continue

    formula = 'HR_ratio ~ ' + ' + '.join(predictors)
    full_model = smf.ols(formula, data=df_c).fit()
    gee_model = smf.gee(formula, 'subject', df_c,
                        cov_struct=sm.cov_struct.Exchangeable()).fit()

    full_r2 = full_model.rsquared
    r2_dict[cond] = full_r2

    print(f"\n===== {cond} =====")
    print(f"n = {len(df_c)} rows from {df_c['subject'].nunique()} subjects "
          f"({len(df_c) / df_c['subject'].nunique():.1f} rows per subject)")
    print(f"OLS R2 = {full_r2:.4f}   adjusted R2 = {full_model.rsquared_adj:.4f}")
    print(f"Overall model F({int(full_model.df_model)},{int(full_model.df_resid)})"
          f" = {full_model.fvalue:.3f}, p = {full_model.f_pvalue:.4g}")

    for var in predictors:

        reduced_model = smf.ols(
            'HR_ratio ~ ' + ' + '.join([v for v in predictors if v != var]),
            data=df_c
        ).fit()

        results.append({
            'condition': cond,
            'variable': var,
            'label': predictor_labels[var],
            'partial_r2': full_r2 - reduced_model.rsquared,
            'corr': df_c[[var, 'HR_ratio']].corr().iloc[0, 1],
            'pval_ols': full_model.pvalues[var],
            'pval_gee': gee_model.pvalues[var],
        })

results_df = pd.DataFrame(results)


if results_df.empty:
    print('No condition had enough complete rows to fit the model'
          ' - skipping the regression figure.')
else:
    # Bonferroni across the per-condition coefficient tests. 'pval' is the column
    # the figure stars, so the plotting code below needed no changes.
    results_df['pval'] = multipletests(
        results_df['pval_gee'], alpha=0.05, method='bonferroni')[1]

    print("\n" + "=" * 78)
    print("PER-CONDITION COEFFICIENTS")
    print("bars = OLS partial R2   |   stars = GEE p, Bonferroni x%d" % len(results_df))
    print("=" * 78)
    print(f"{'condition':9} {'predictor':20} {'partial R2':>11} {'OLS p':>9}"
          f" {'GEE p':>9} {'Bonf p':>9}  sig")
    for _, row in results_df.iterrows():
        print(f"{row['condition']:9} {row['label']:20} {row['partial_r2']:11.4f}"
              f" {row['pval_ols']:9.4f} {row['pval_gee']:9.4f} {row['pval']:9.4f}"
              f"  {'*' if row['pval'] < 0.05 else ''}")

    # ---------------------------
    # Does a predictor's effect DIFFER between conditions?
    # Fitting each condition separately cannot answer that - the interaction term
    # is the actual test. Needs both conditions present.
    # ---------------------------
    inter_df = behavior_avg.dropna(subset=predictors + ['HR_ratio', 'subject']).copy()

    if inter_df['condition'].nunique() < 2:
        print("\nOnly one condition available - skipping the interaction model.")
    else:
        inter_df['condition'] = pd.Categorical(inter_df['condition'],
                                               categories=['Control', 'CNSDS'])
        inter_model = smf.gee(
            'HR_ratio ~ (' + ' + '.join(predictors) + ') * C(condition)',
            'subject', inter_df, cov_struct=sm.cov_struct.Exchangeable()).fit()

        inter_terms = [t for t in inter_model.params.index if ':' in t]
        inter_padj = multipletests(inter_model.pvalues[inter_terms],
                                   alpha=0.05, method='bonferroni')[1]

        print("\n" + "=" * 78)
        print("CONDITION x PREDICTOR INTERACTIONS (GEE, Bonferroni x%d)" % len(inter_terms))
        print("=" * 78)
        print(f"{'term':46} {'raw p':>9} {'Bonf p':>9}  sig")
        for term, praw, padj in zip(inter_model.pvalues[inter_terms].index,
                                    inter_model.pvalues[inter_terms], inter_padj):
            print(f"{term:46} {praw:9.4f} {padj:9.4f}  {'*' if padj < 0.05 else ''}")


    # ---------------------------
    # Print sample sizes
    # ---------------------------
    print("\n====================")
    print("Sample sizes:")
    for k, v in n_dict.items():
        print(f"{k}: n = {v}")

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(6, 3.5))

    palette = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}

    ax = sns.barplot(
        data=results_df,
        x='label',
        y='partial_r2',
        hue='condition',
        palette=palette
    )

    # ---------------------------
    # Annotations
    # ---------------------------
    x_positions = {label: i for i, label in enumerate(results_df["label"].unique())}

    for _, row in results_df.iterrows():
        x = x_positions[row['label']]
        offset = -0.2 if row['condition'] == 'Control' else 0.2

        y = row['partial_r2']

        # significance star
        if row['pval'] < 0.05:
            ax.text(
                x + offset,
                y + 0.02,
                '*',
                ha='center',
                va='bottom',
                fontsize=SIG_STAR_FONTSIZE
            )

    plt.ylabel("Partial R²")
    plt.xlabel("")
    plt.xlabel('Predictor')
    plt.xticks(rotation=25)
    plt.title("Predicted HR Ratio", fontsize=16)
    # NOTE (conversion): legend removed on request.
    _lg = ax.get_legend()
    if _lg is not None:
        _lg.remove()

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"c:\\Users\\Jillian.Sucher\\Documents\\Stress_microstructure_testing_day_1\\regression_by_condition.pdf", format='pdf')
    plt.show()


# ==========================================================================
# SYLLABLES CORRELATIONS
# ==========================================================================

# NOTE (conversion): this section needs a CSV produced by a separate
# pipeline. Guarded so a missing file skips the section instead of
# aborting the run after every other figure has been written.
SYLLABLE_CSV = "new_analysis_df.csv"

if not os.path.exists(SYLLABLE_CSV):
    console(f"[skip] {SYLLABLE_CSV} not found - skipping the syllable correlation section.")
else:

    # ---- notebook cell 114 ---------------------------------------------------------
    # Load the exported DataFrame
    new_analysis_df = pd.read_csv(SYLLABLE_CSV)

    # ---- notebook cell 115 ---------------------------------------------------------
    # Step 1: Group by syllable, subject, and experiment, then average the ratio
    avg_ratios = (
        new_analysis_df
        .groupby(['syllable', 'subject', 'experiment'])['ratio']
        .mean()
        .reset_index()
    )

    condition_info = new_analysis_df[['subject', 'experiment', 'condition']].drop_duplicates()

    syllable_ratio_df = pd.merge(avg_ratios, condition_info, on=['subject', 'experiment'], how='left')

    syllable_ratio_df = pd.merge(
        analysis_df,
        syllable_ratio_df[['syllable', 'subject', 'experiment', 'ratio']], 
        on=['subject', 'experiment'],
        how='left'
    )

    # ---- notebook cell 116 ---------------------------------------------------------
    from scipy.stats import linregress, pearsonr
    import numpy as np
    import matplotlib.pyplot as plt

    colors = {'Control': CONTROL_GRAY, 'CNSDS': CNSDS_GRAY}

    syllables = syllable_ratio_df['syllable'].unique()
    n_cols = 3
    n_rows = (len(syllables) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    print("Pearson correlation results:\n")

    for idx, syll in enumerate(syllables):
        ax = axes[idx // n_cols, idx % n_cols]
        data = syllable_ratio_df[syllable_ratio_df['syllable'] == syll]
    
        title = f"Syllable: {syll}"
    
        sig_p = {}
    
        for condition, color in colors.items():
            subset = data[data['condition'] == condition]
            x = subset['ratio']
            y = subset['HR_ratio']

            if len(x) > 1 and len(y) > 1:
                r, p = pearsonr(x, y)
                sig_p[condition] = p  # Track p-value
                print(f"Syllable: {syll}, Condition: {condition}, r = {r:.3f}, p = {p:.4f}")

                ax.scatter(x, y, color=color, alpha=0.7, label=f"{condition} (r={r:.2f})")

                # Bin for SEM error bars
                bins = np.linspace(x.min(), x.max(), 10)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                means = []
                sems = []
            
                for start, end in zip(bins[:-1], bins[1:]):
                    bin_vals = y[(x >= start) & (x < end)]
                    means.append(bin_vals.mean() if len(bin_vals) > 0 else np.nan)
                    sems.append(bin_vals.sem() if len(bin_vals) > 1 else 0)

                means = np.array(means)
                sems = np.array(sems)
                valid = ~np.isnan(means)

                # Horizontal error bars: half bin width
                x_err = (bins[1] - bins[0]) / 2
                xerrs = np.full_like(means[valid], x_err)

                # Plot with larger points and both error bars
                ax.errorbar(
                    bin_centers[valid],
                    means[valid],
                    xerr=xerrs,
                    yerr=sems[valid],
                    fmt='o',
                    color=color,
                    alpha=0.5,
                    markersize=10,
                    capsize=4
                )
                # Regression line
                slope, intercept, _, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = intercept + slope * x_fit
                ax.plot(x_fit, y_fit, color=color, linestyle='--')

            else:
                print(f"Syllable: {syll}, Condition: {condition} — Not enough data for correlation")
                ax.scatter(x, y, color=color, alpha=0.7, label=condition)

        # Show asterisks for each condition that is significant
        y_pos = 0.95  # Start near the top of the plot
        for condition, p_val in sig_p.items():
            if p_val < 0.001:
                sig_label = '***'
            elif p_val < 0.01:
                sig_label = '**'
            elif p_val < 0.05:
                sig_label = '*'
            else:
                continue  # Skip if not significant

            ax.text(
                0.05, y_pos, sig_label,
                transform=ax.transAxes,
                fontsize=30,
                verticalalignment='top',
                horizontalalignment='left',
                color=colors[condition],
                weight='bold'
            )
            y_pos -= 0.08  # Move down for next asterisk if needed


        ax.set_title(title, fontsize=20)
        ax.set_xlabel('Fraction of Total Syllables', fontsize=18)
        ax.tick_params(axis='both', labelsize= 16)
        ax.set_ylabel('HR Ratio', fontsize=18)
        ax.grid(False)
        # ax.legend(fontsize=8)

    # Turn off unused axes
    for empty_ax in axes.flatten()[len(syllables):]:
        empty_ax.axis('off')

    plt.tight_layout()
    plt.savefig("D:/Figures/correlation_syllables.pdf", format="pdf")
    plt.show()

