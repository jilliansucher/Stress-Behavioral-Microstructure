"""
Syllable transition-matrix decoding: LDA + confusion matrices.

Ported from davidgwyrick/jumping_behavior (the repo referenced in
kpms_param_search_070826.py) and adapted to read a keypoint-MoSeq model.

  decode_labels                   <- decoding.py
  calculate_accuracy (L1O branch) <- decoding.py
  get_transition_count_matrices   <- util.py
  plot_decoding_accuracy          <- decoding.py

The model is READ-ONLY. This calls kpms.load_results() and nothing else - no
refitting, no reindexing, nothing written into the project directory.

Outputs
-------
    <FIGURE_DIR>/confusion_matrix.pdf
    <FIGURE_DIR>/decoder_weights.pdf
    <FIGURE_DIR>/decoding_per_barrier.pdf
    <FIGURE_DIR>/delta_transition_matrix.pdf

    <ANALYSIS_ROOT>/decoding_report_<stamp>.pdf
        every figure plus: decoder p-values and z-scores, the full transition
        probability matrices per condition, and the per-barrier results.

Figures are in colour, following the repo's formatting.
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

import scipy.stats as st
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import keypoint_moseq as kpms


# =============================================================================
#                                 CONFIGURATION
# =============================================================================

PROJECT_DIR = r'D:\Barrier_testing_day1_videos\KPMS'
MODEL_NAME = 'final-K8-kappa1e+03'

ANALYSIS_ROOT = (r'C:\Users\Jillian.Sucher\Documents'
                 r'\Stress_microstructure_testing_day_1\analysis_output')
FIGURE_DIR = os.path.join(ANALYSIS_ROOT, 'figures')

# Shuffle count for the null distribution. decoding.py uses 100.
N_SHUFFLES = 100
RANDOM_SEED = 0

# Transition matrices. lexical=True counts only state CHANGES and ignores
# self-transitions, which is what jumping_behavior uses by default.
LEXICAL_TRANSITIONS = True
NORMALIZE_TRANSITIONS = True

# Syllables to drop before building the matrices (matches the other scripts).
EXCLUDE_SYLLABLES = [6, 7]

# What counts as one sample for the decoder.
#   'animal'    - each animal's recordings are averaged into one matrix, and
#                 leave-one-out holds out a whole animal. Nothing from the
#                 held-out mouse is in the training set.
#   'recording' - every recording is its own sample. More samples and usually
#                 a higher accuracy, but recordings from the same mouse land on
#                 both sides of the split, so part of what the classifier picks
#                 up is animal identity rather than condition.
# Both are run when RUN_BOTH_LEVELS is True; DECODE_PER is the primary one
# reported in the figures.
DECODE_PER = 'animal'

# Run the whole analysis - figures AND stats - at both levels. Files are
# suffixed _per_animal / _per_recording so nothing overwrites.
RUN_BOTH_LEVELS = True

# Only keep recordings whose name matches this regex. The project holds both
# the filtered and unfiltered DLC outputs for each video, which is why the
# model reports ~479 recordings instead of 270.
#
# NOTE the negative lookbehind: a plain 'filtered' substring test also matches
# 'unfiltered', which keeps every duplicate and silently does nothing.
RECORDING_NAME_FILTER = r'(?<!un)filtered'

# How an animal's recordings are combined into one matrix.
#   'mean_of_probabilities' - average the per-recording normalised matrices.
#                             Every recording counts equally.
#   'pooled_counts'         - sum the raw counts across the animal's
#                             recordings, then normalise once. Longer
#                             recordings carry more weight.
ANIMAL_AGGREGATION = 'mean_of_probabilities'

# Cross-validation, following jumping_behavior/decoding.py cross_validate().
#   'kfold' - StratifiedKFold(n_splits=N_KFOLD). Each fold's confusion matrix
#             is normalised on its own, then averaged over folds. This is what
#             the original figures used (their values, e.g. 0.65 / 0.32, are
#             not attainable from leave-one-out, which can only produce k/N).
#   'L1O'   - StratifiedKFold(n_splits=n per smallest class), i.e. leave one
#             out. Hits are summed across folds and normalised once.
# His note: kfold approximates accuracy per fold and averages, which is the
# better approximation when there is enough data; L1O is for low-data cases,
# where the classifiers are very similar and share a lot of variance.
CV_METHOD = 'kfold'
N_KFOLD = 5

CONDITIONS = ['Control', 'CNSDS']

# Recordings whose name does not contain 'control' or 'cnsds' come back with a
# missing condition and would be dropped. The behavioural notebook fills these
# in by subject (cell 18: WT041 and WT047 -> Control); the same fix applies
# here. Keys are matched against the parsed subject id, lowercased.
CONDITION_OVERRIDES = {
    'wt041': 'Control',
    'wt047': 'Control',
}

# Colour, per the repo. cc is its accent colour list, used for the sig markers.
sns.set_style('ticks')
cc = sns.color_palette('husl', 8)
CONDITION_COLORS = {'Control': 'orange', 'CNSDS': 'blue'}
CONFUSION_CMAP = 'rocket'
DELTA_CMAP = 'RdBu_r'
TCM_CMAP = 'rocket'
WEIGHT_CMAP = 'RdBu_r'

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['figure.max_open_warning'] = 0


# =============================================================================
#                          OUTPUT / REPORT PLUMBING
# =============================================================================

RUN_ID = None
REPORT_PATH = None
_report = None
_log_buffer = None
_real_stdout = sys.stdout
_counts = {'figures': 0, 'text': 0}
_mark = [0]
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
    kwargs['file'] = _real_stdout
    print(*args, **kwargs)
    _real_stdout.flush()


def start_run():
    global RUN_ID, REPORT_PATH, _report, _log_buffer

    os.makedirs(ANALYSIS_ROOT, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    probe = os.path.join(FIGURE_DIR, '_write_test.tmp')
    with open(probe, 'w') as fh:
        fh.write('ok')
    os.remove(probe)

    RUN_ID = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    suffix = 2
    while os.path.exists(os.path.join(ANALYSIS_ROOT,
                                      f'decoding_report_{RUN_ID}.pdf')):
        RUN_ID = f'{RUN_ID}_{suffix}'
        suffix += 1
    REPORT_PATH = os.path.join(ANALYSIS_ROOT, f'decoding_report_{RUN_ID}.pdf')

    console(f'  figures -> {FIGURE_DIR}')
    console(f'  report  -> {REPORT_PATH}')

    _report = PdfPages(REPORT_PATH)
    _log_buffer = io.StringIO()
    sys.stdout = _Tee(_real_stdout, _log_buffer)


def _emit_text_pages(title):
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


def save_figures_multipage(figs, stem):
    """Several figures into ONE individual PDF (decoding_per_barrier is one
    file with a page per barrier), and each page into the report too."""
    path = os.path.join(FIGURE_DIR, f'{stem}.pdf')
    try:
        with PdfPages(path) as pp:
            for fig in figs:
                pp.savefig(fig, bbox_inches='tight')
        console(f'  saved {stem}.pdf  ({os.path.getsize(path):,} bytes, '
                f'{len(figs)} pages)')
    except Exception as exc:
        console(f'  !! FAILED to write {path}: {type(exc).__name__}: {exc}')

    if _report is not None:
        _emit_text_pages(f'Output preceding: {stem}.pdf')
        for fig in figs:
            _report.savefig(fig, bbox_inches='tight')
            _counts['figures'] += 1
    for fig in figs:
        plt.close(fig)


def save_figure(fig, stem):
    """Bare figure to FIGURE_DIR, then into the report with the stats printed
    just before it."""
    path = os.path.join(FIGURE_DIR, f'{stem}.pdf')
    try:
        fig.savefig(path, format='pdf', bbox_inches='tight')
        console(f'  saved {stem}.pdf  ({os.path.getsize(path):,} bytes)')
    except Exception as exc:
        console(f'  !! FAILED to write {path}: {type(exc).__name__}: {exc}')

    if _report is not None:
        _emit_text_pages(f'Output preceding: {stem}.pdf')
        _report.savefig(fig, bbox_inches='tight')
        _counts['figures'] += 1
    plt.close(fig)


def finalise():
    """Close the report deterministically. Must not rely on atexit: inside
    Spyder/IPython the interpreter stays alive, atexit never fires, and the
    report is left at 0 bytes - which opens as a corrupted PDF."""
    if _finalised[0]:
        return
    _finalised[0] = True
    try:
        if _report is not None:
            _emit_text_pages('Final output')
            info = _report.infodict()
            info['Title'] = f'Transition-matrix decoding - run {RUN_ID}'
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
#                     PORTED FROM jumping_behavior/util.py
# =============================================================================

def get_transition_count_matrices(trMAPs, trMasks, K, normalize=True,
                                  lexical=True):
    """Transition count matrix per trial from a MAP state sequence.

    Verbatim from jumping_behavior/util.py.
    """
    TCMs = np.zeros((len(trMAPs), K, K))

    for iTrial, (map_seq, mask) in enumerate(zip(trMAPs, trMasks)):
        MAP = np.squeeze(map_seq).copy()
        MAP[~mask] = -1

        if all(~mask):
            continue

        sMAP = [int(s) for s in MAP if s != -1]
        if not sMAP:
            continue

        prev_state = sMAP[0]
        for iT, state in enumerate(sMAP):
            if lexical:
                if prev_state != state:
                    TCMs[iTrial, prev_state, state] += 1
                    prev_state = state
            else:
                TCMs[iTrial, prev_state, state] += 1
                prev_state = state

        total = np.sum(TCMs[iTrial])
        if normalize and total > 0:
            TCMs[iTrial] = TCMs[iTrial] / total

    return TCMs


# =============================================================================
#                   PORTED FROM jumping_behavior/decoding.py
# =============================================================================

def decode_labels(X, Y, train_index, test_index, classifier='LDA',
                  clabels=None, shuffle=True, n_shuffles=N_SHUFFLES):
    """LDA decoding of one train/test split, with a shuffled null.

    Ported from decoding.py, trimmed to the LDA branch (the SVM / MLP /
    Euclidean_Dist branches are unused here).
    """
    train_index_sh = np.array(train_index).copy()
    np.random.shuffle(train_index_sh)

    X_train, X_test = X[train_index, :], X[test_index, :]
    Y_train, Y_test = Y[train_index], Y[test_index]

    class_labels, nTrials_class = np.unique(Y, return_counts=True)
    nClasses = len(class_labels)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, Y_train)
    Y_hat = clf.predict(X_test)
    decoding_weights = clf.coef_

    kfold_hits = confusion_matrix(Y_test, Y_hat, labels=clabels)
    nPredictors = decoding_weights.shape[1]

    if shuffle:
        kfold_shf = np.zeros((n_shuffles, nClasses, nClasses))
        decoding_weights_shf = np.zeros((n_shuffles, nPredictors))

        for iS in range(n_shuffles):
            np.random.shuffle(train_index_sh)
            Y_train_sh = Y[train_index_sh]

            clf_shf = LinearDiscriminantAnalysis()
            clf_shf.fit(X_train, Y_train_sh)
            Y_hat_shf = clf_shf.predict(X_test)
            decoding_weights_shf[iS] = clf_shf.coef_

            kfold_shf[iS] = confusion_matrix(Y_test, Y_hat_shf,
                                             labels=clabels)

        decoding_weights_m_shf = np.mean(decoding_weights_shf, axis=0)
        decoding_weights_s_shf = np.std(decoding_weights_shf, axis=0)
        decoding_weights_z = np.divide(
            decoding_weights - decoding_weights_m_shf,
            decoding_weights_s_shf,
            out=np.zeros(decoding_weights.shape, dtype=np.float32),
            where=decoding_weights_s_shf != 0)
    else:
        kfold_shf = np.zeros((nClasses, nClasses))
        decoding_weights_m_shf = np.zeros(nPredictors)
        decoding_weights_z = np.zeros(decoding_weights.shape)

    return (kfold_hits, kfold_shf, decoding_weights,
            decoding_weights_m_shf, decoding_weights_z)


def calculate_accuracy(results, method='kfold', n_shuffles=N_SHUFFLES):
    """Both branches of calculate_accuracy from jumping_behavior/decoding.py.

    L1O   : sum hits over folds, normalise once, then z-score against the
            shuffle distribution. Entries can only be multiples of 1/N.
    kfold : normalise EACH fold on its own, z-score each fold, then average
            the per-fold matrices. Entries are not constrained to 1/N.
    """
    nClasses = results[0][0].shape[0]

    if method == 'L1O':
        confusion_mat = np.zeros((nClasses, nClasses))
        c_shf = np.zeros((n_shuffles, nClasses, nClasses))

        for rTuple in results:
            confusion_mat += rTuple[0]
            for iS in range(n_shuffles):
                c_shf[iS] += rTuple[1][iS]

        row = np.sum(confusion_mat, axis=1).reshape(-1, 1)
        confusion_mat = np.divide(confusion_mat, row,
                                  out=np.zeros_like(confusion_mat),
                                  where=row != 0)
        for iS in range(n_shuffles):
            row_s = np.sum(c_shf[iS], axis=1).reshape(-1, 1)
            c_shf[iS] = np.divide(c_shf[iS], row_s,
                                  out=np.zeros_like(c_shf[iS]),
                                  where=row_s != 0)

        m_shf, s_shf = np.mean(c_shf, axis=0), np.std(c_shf, axis=0)
        confusion_shf = m_shf
        confusion_z = np.divide(confusion_mat - m_shf, s_shf,
                                out=np.zeros_like(confusion_mat),
                                where=s_shf != 0)

    elif method == 'kfold':
        kfold_accuracies, shf_accuracies, kfold_zscores = [], [], []

        for iK, rTuple in enumerate(results):
            kfold_hits, kfold_shf = rTuple[0], rTuple[1]

            row = np.sum(kfold_hits, axis=1).reshape(-1, 1)
            kfold_accuracies.append(
                np.divide(kfold_hits, row,
                          out=np.zeros_like(kfold_hits, dtype=float),
                          where=row != 0))

            c_shf = np.zeros((n_shuffles, nClasses, nClasses))
            for iS in range(n_shuffles):
                row_s = np.sum(kfold_shf[iS], axis=1).reshape(-1, 1)
                c_shf[iS] = np.divide(kfold_shf[iS], row_s,
                                      out=np.zeros_like(c_shf[iS]),
                                      where=row_s != 0)

            m_shf, s_shf = np.mean(c_shf, axis=0), np.std(c_shf, axis=0)
            shf_accuracies.append(m_shf)
            kfold_zscores.append(
                np.divide(kfold_accuracies[iK] - m_shf, s_shf,
                          out=np.zeros_like(m_shf), where=s_shf != 0))

        confusion_mat = np.mean(kfold_accuracies, axis=0)
        confusion_shf = np.mean(shf_accuracies, axis=0)
        confusion_z = np.mean(kfold_zscores, axis=0)

    else:
        raise ValueError(f"CV_METHOD must be 'kfold' or 'L1O', got {method!r}")

    pvalues = st.norm.sf(confusion_z)
    return confusion_mat, confusion_shf, confusion_z, pvalues


def make_folds(Y, method=None, n_kfold=None):
    """StratifiedKFold splitter, as in cross_validate().

    L1O uses n_splits = the size of the smallest class, which is what makes it
    leave-one-out while keeping the classes balanced across folds.
    """
    method = method or CV_METHOD
    n_kfold = n_kfold or N_KFOLD
    _, nTrials_class = np.unique(Y, return_counts=True)
    if method == 'L1O':
        n_splits = int(nTrials_class.min())
    else:
        n_splits = int(min(n_kfold, nTrials_class.min()))
    n_splits = max(2, n_splits)
    return StratifiedKFold(n_splits=n_splits), n_splits


def _confusion_panel(ax, confusion_mat, pvalues, class_labels,
                     title='Confusion Matrix', cbar=True, cbar_label=None):
    """Confusion heatmap with green significance stars.

    Star placement is the repo's: ax.text(j+0.75, i+0.25, '*', color='g',
    fontsize=20, fontweight='bold') from plot_confusion_matrices.
    """
    sns.heatmap(confusion_mat, annot=True, fmt='.2f',
                annot_kws={'fontsize': 16}, cmap=CONFUSION_CMAP,
                cbar=cbar, square=True, vmin=0, vmax=1,
                cbar_kws={'shrink': 0.5,
                          'label': cbar_label} if cbar else None,
                ax=ax)

    nClasses = confusion_mat.shape[0]
    for i in range(nClasses):
        for j in range(nClasses):
            if pvalues[i, j] < 0.05:
                ax.text(j + 0.75, i + 0.25, '*', color='g', fontsize=20,
                        fontweight='bold')

    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_xlabel('Decoded', fontsize=14)
    ax.set_yticks(np.arange(len(class_labels)) + 0.5)
    ax.set_xticks(np.arange(len(class_labels)) + 0.5)
    ax.set_yticklabels(class_labels, va='center', rotation=90, fontsize=12)
    ax.set_xticklabels(class_labels, va='center', rotation=0, fontsize=12)
    return ax


def _matrix_panel(ax, mat, labels, title, cmap, center=None, cbar_label=None,
                  annot=False, star_mask=None):
    """Square K x K heatmap used for the TCM / delta / weight panels."""
    kwargs = dict(cmap=cmap, square=True, ax=ax, annot=annot,
                  xticklabels=labels, yticklabels=labels,
                  cbar_kws={'shrink': 0.6, 'label': cbar_label})
    if center is not None:
        lim = np.abs(mat).max()
        kwargs.update(center=center, vmin=-lim, vmax=lim)
    if annot:
        kwargs.update(fmt='.2f', annot_kws={'fontsize': 9})
    sns.heatmap(mat, **kwargs)

    if star_mask is not None:
        x = np.arange(mat.shape[1]) + 0.5
        y = np.arange(mat.shape[0]) + 0.5
        X, Y = np.meshgrid(x, y)
        ax.scatter(X[star_mask], Y[star_mask], marker='*', s=130, c='black',
                   zorder=5)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('To Syllable', fontsize=12)
    ax.set_ylabel('From Syllable', fontsize=12)
    return ax


# =============================================================================
#                       DATA: read the model, build X and Y
# =============================================================================

def extract_metadata_from_name(name_col):
    norm = name_col.str.replace(
        r'Barrier_Testing_(\d+cm)_Day_(\d)', r'\1_barrier_day\2', regex=True)
    condition = (norm.str.extract(r'(cnsds|control)', flags=re.IGNORECASE)[0]
                 .str.lower().map({'control': 'Control', 'cnsds': 'CNSDS'}))
    barrier = norm.str.extract(r'(10|15|20)\s*cm?', flags=re.IGNORECASE)[0]
    day = norm.str.extract(r'day[_\s]?(1|2|3)', flags=re.IGNORECASE)[0]
    subject = norm.str.extract(r'(wt\d+)', flags=re.IGNORECASE)[0].str.lower()
    return pd.DataFrame({'condition': condition, 'barrier': barrier,
                         'day': day, 'subject': subject})


def build_transition_features(project_dir, model_name):
    """One transition matrix per recording, plus its metadata.

    Reads the model only - kpms.load_results() does not modify anything.
    """
    console('Loading model results (read-only)...')
    results = kpms.load_results(project_dir, model_name)

    names = list(results.keys())
    n_in_model = len(names)
    print('\n' + '=' * 74)
    print('RECORDING COUNT')
    print('=' * 74)
    print(f'  recordings in the model              : {n_in_model}')

    if RECORDING_NAME_FILTER:
        pattern = re.compile(RECORDING_NAME_FILTER, re.IGNORECASE)
        keep_names = [n for n in names if pattern.search(str(n))]
        removed = len(names) - len(keep_names)
        print(f'  filter                               : '
              f'{RECORDING_NAME_FILTER!r}')
        print(f'  after the name filter                : {len(keep_names)} '
              f'({removed} dropped as duplicates)')
        if not keep_names:
            raise RuntimeError(
                f'No recording name matches {RECORDING_NAME_FILTER!r}. '
                f'First few names are:\n    ' +
                '\n    '.join(str(n) for n in names[:5]) +
                '\nSet RECORDING_NAME_FILTER to a substring that matches, '
                'or None to disable filtering.')
        names = keep_names

    meta = extract_metadata_from_name(pd.Series(names))
    meta['name'] = names

    # Fill missing conditions by subject, mirroring the behavioural notebook.
    if CONDITION_OVERRIDES:
        filled = 0
        for subject, condition in CONDITION_OVERRIDES.items():
            hit = (meta['subject'] == subject) & (meta['condition'].isna())
            n = int(hit.sum())
            if n:
                meta.loc[hit, 'condition'] = condition
                print(f'  condition override: {subject} -> {condition} '
                      f'({n} recordings)')
                filled += n
        if filled:
            print(f'  {filled} recording(s) had their condition filled in.')

    still_missing = meta[meta['condition'].isna()]
    if len(still_missing):
        print(f'\n  {len(still_missing)} recording(s) STILL have no condition '
              f'and will be dropped.')
        subs = sorted(set(str(x) for x in still_missing['subject'].unique()))
        print(f'  subjects affected: {subs}')
        print('  first few names:')
        for n in still_missing['name'].head(5):
            print(f'    {n}')
        print('  (add these subjects to CONDITION_OVERRIDES if they should '
              'be kept)')

    sequences = [np.asarray(results[n]['syllable']) for n in names]

    all_syllables = np.unique(np.concatenate(sequences))
    K = int(all_syllables.max()) + 1
    print(f'Model has {len(all_syllables)} syllables in use, K = {K}')
    print(f'Excluding syllables {EXCLUDE_SYLLABLES} from the matrices')

    # Mask out excluded syllables; get_transition_count_matrices treats
    # masked frames as "not a state" and skips them.
    masks = [~np.isin(seq, EXCLUDE_SYLLABLES) for seq in sequences]

    TCMs = get_transition_count_matrices(
        sequences, masks, K,
        normalize=NORMALIZE_TRANSITIONS, lexical=LEXICAL_TRANSITIONS)

    # Unnormalised counts, needed for the 'pooled_counts' aggregation.
    RAW = get_transition_count_matrices(
        sequences, masks, K, normalize=False, lexical=LEXICAL_TRANSITIONS)

    keep = meta['condition'].notna() & meta['subject'].notna()
    dropped = int((~keep).sum())
    print(f'  with a parsed subject and condition  : {int(keep.sum())}'
          f'{f" ({dropped} dropped)" if dropped else ""}')

    kept_meta = meta[keep]
    print(f'  distinct animals                     : '
          f'{kept_meta["subject"].nunique()}')
    for cond in CONDITIONS:
        sub = kept_meta[kept_meta['condition'] == cond]
        print(f'    {cond:8}: {sub["subject"].nunique():3} animals, '
              f'{len(sub):4} recordings')

    per_animal = kept_meta.groupby('subject').size()
    if len(per_animal):
        print(f'  recordings per animal                : '
              f'min {per_animal.min()}, median '
              f'{int(per_animal.median())}, max {per_animal.max()}')
        odd = per_animal[per_animal != per_animal.median()]
        if len(odd):
            print(f'  animals with a non-median count      : '
                  f'{dict(odd)}')

    grid = kept_meta.groupby(['barrier', 'day']).size()
    if len(grid):
        print('  recordings per barrier x day:')
        for (barrier, day), n in grid.items():
            print(f'    barrier {barrier}cm  day {day}: {n}')
    return (TCMs[keep.values], RAW[keep.values],
            meta[keep].reset_index(drop=True), K)


def per_recording(TCMs, meta, raw=None):
    """Every recording is its own sample - no aggregation."""
    mats = np.asarray(TCMs)
    X = mats.reshape(len(mats), -1)
    Y = meta['condition'].values
    labels = meta['name'].tolist() if 'name' in meta.columns else \
        list(range(len(mats)))
    return X, Y, labels, [1] * len(mats), mats


def build_samples(TCMs, meta, level, raw=None):
    """Dispatch to the per-animal or per-recording view of the data."""
    if level == 'recording':
        return per_recording(TCMs, meta, raw)
    return aggregate_per_subject(TCMs, meta, raw)


def aggregate_per_subject(TCMs, meta, raw=None):
    """Average each animal's recordings into one transition matrix.

    Decoding operates on animals, not recordings: recordings from the same
    mouse are not independent, and leaving out one recording while its
    siblings stay in the training set leaks identity into the classifier.
    """
    subjects, mats, Y, rows = [], [], [], []
    for subject, idx in meta.groupby('subject').groups.items():
        idx = list(idx)
        cond = meta.loc[idx, 'condition'].iloc[0]
        subjects.append(subject)
        if ANIMAL_AGGREGATION == 'pooled_counts' and raw is not None:
            pooled = raw[idx].sum(axis=0)
            total = pooled.sum()
            mats.append(pooled / total if total > 0 else pooled)
        else:
            mats.append(TCMs[idx].mean(axis=0))
        Y.append(cond)
        rows.append(len(idx))

    # mats is (n_animals, K, K). Keep the square form for the heatmaps and a
    # flattened (n_animals, K*K) copy for the classifier, which needs 2-D.
    mats = np.array(mats)
    X = mats.reshape(len(mats), -1)
    return X, np.array(Y), subjects, rows, mats


# =============================================================================
#                                   FIGURES
# =============================================================================

def print_transition_matrices(X, Y, K, syllable_labels):
    """Full transition probabilities per condition, into the report."""
    print('\n' + '=' * 74)
    print('TRANSITION PROBABILITY MATRICES')
    print('=' * 74)
    print(f'{"lexical" if LEXICAL_TRANSITIONS else "full"} transitions, '
          f'{"row" if False else "matrix"}-normalised per recording, then '
          f'averaged per animal and per condition.')

    for cond in CONDITIONS:
        sub = X[Y == cond]
        if len(sub) == 0:
            continue
        mat = sub.mean(axis=0)
        print(f'\n--- {cond}  (n = {len(sub)} animals) ---')
        header = '  from\\to ' + ' '.join(f'{s:>8}' for s in syllable_labels)
        print(header)
        for i, s_from in enumerate(syllable_labels):
            row = ' '.join(f'{mat[s_from, s_to]:8.4f}'
                           for s_to in syllable_labels)
            print(f'  {s_from:>7} {row}')


def plot_delta_transition_matrix(mats, Y, syllable_labels, stem,
                                 tcm_suffix=''):
    """Three panels: Control TCM, CNSDS TCM, and their difference."""
    idx = np.array(syllable_labels)
    ctrl = mats[Y == 'Control'].mean(axis=0)[np.ix_(idx, idx)]
    cnsds = mats[Y == 'CNSDS'].mean(axis=0)[np.ix_(idx, idx)]
    delta = ctrl - cnsds

    print('\n' + '=' * 74)
    print('TRANSITION PROBABILITY MATRICES')
    print('=' * 74)
    for name, mat in [('Control', ctrl), ('CNSDS', cnsds),
                      ('Delta (Control - CNSDS)', delta)]:
        print(f'\n--- {name} ---')
        print('  from\\to ' + ' '.join(f'{s_:>8}' for s_ in syllable_labels))
        for i, s_from in enumerate(syllable_labels):
            print(f'  {s_from:>7} ' +
                  ' '.join(f'{mat[i, j]:8.4f}'
                           for j in range(len(syllable_labels))))
    print(f'\nlargest increase in Control: {delta.max():.4f}')
    print(f'largest increase in CNSDS  : {delta.min():.4f}')

    # The per-condition matrices and the difference go to separate files.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _matrix_panel(axes[0], ctrl, syllable_labels, 'Control TCM', TCM_CMAP,
                  cbar_label='Transition probability')
    _matrix_panel(axes[1], cnsds, syllable_labels, 'CNSDS TCM', TCM_CMAP,
                  cbar_label='Transition probability')
    axes[1].set_ylabel('')
    plt.tight_layout()
    save_figure(fig, f'transition_matrices{tcm_suffix}')

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    _matrix_panel(ax, delta, syllable_labels, r'$\Delta$Transition Matrix',
                  DELTA_CMAP, center=0,
                  cbar_label=r'$\Delta$ (Ctrl - CNSDS)')
    plt.tight_layout()
    save_figure(fig, stem)


def run_decoding(X, Y, label='All barriers', n_shuffles=N_SHUFFLES,
                 unit='animal'):
    """Leave-one-animal-out LDA with a shuffled null."""
    clabels = CONDITIONS
    k_fold, n_splits = make_folds(Y)
    results, weights_z, weights_raw = [], [], []

    for train_index, test_index in k_fold.split(X, Y):
        out = decode_labels(X, Y, train_index, test_index,
                            classifier='LDA', clabels=clabels,
                            shuffle=True, n_shuffles=n_shuffles)
        results.append(out)
        weights_z.append(out[4])     # z-scored vs shuffle
        weights_raw.append(out[2])   # raw LDA coefficients

    confusion_mat, confusion_shf, confusion_z, pvalues = calculate_accuracy(
        results, method=CV_METHOD, n_shuffles=n_shuffles)

    accuracy = np.trace(confusion_mat) / len(clabels)
    shf_accuracy = np.trace(confusion_shf) / len(clabels)

    print('\n' + '=' * 74)
    print(f'LDA DECODING - {label}')
    print('=' * 74)
    scheme = (f'Leave-one-{unit}-out' if CV_METHOD == 'L1O'
              else f'{n_splits}-fold stratified CV')
    print(f'{scheme}, {n_shuffles} label shuffles, '
          f'n = {len(Y)} {unit}s '
          f'({(Y == "Control").sum()} Control, {(Y == "CNSDS").sum()} CNSDS)')
    print(f'Balanced accuracy        : {accuracy:.4f}')
    print(f'Shuffled accuracy (mean) : {shf_accuracy:.4f}')
    print(f'Chance                   : {1 / len(clabels):.4f}')
    print('\nPer-class results:')
    print(f'{"class":>10} {"accuracy":>10} {"shuffled":>10} {"z":>8} {"p":>10}'
          f'  sig')
    for i, cls in enumerate(clabels):
        print(f'{cls:>10} {confusion_mat[i, i]:10.4f} '
              f'{confusion_shf[i, i]:10.4f} {confusion_z[i, i]:8.3f} '
              f'{pvalues[i, i]:10.4g}  '
              f'{"*" if pvalues[i, i] < 0.05 else ""}')

    print('\nFull confusion matrix (rows = actual, cols = decoded):')
    print(f'{"":>10} ' + ' '.join(f'{c:>10}' for c in clabels))
    for i, cls in enumerate(clabels):
        print(f'{cls:>10} ' +
              ' '.join(f'{confusion_mat[i, j]:10.4f}'
                       for j in range(len(clabels))))
    print('\nz-scores vs shuffle:')
    print(f'{"":>10} ' + ' '.join(f'{c:>10}' for c in clabels))
    for i, cls in enumerate(clabels):
        print(f'{cls:>10} ' +
              ' '.join(f'{confusion_z[i, j]:10.3f}'
                       for j in range(len(clabels))))

    return (confusion_mat, confusion_z, pvalues,
            np.array(weights_z), np.array(weights_raw))


def plot_confusion(confusion_mat, pvalues, stem):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _confusion_panel(ax, confusion_mat, pvalues, CONDITIONS,
                     title='Condition Decoding (LDA)',
                     cbar=True, cbar_label='Accuracy')
    plt.tight_layout()
    save_figure(fig, stem)


def plot_decoder_weights(weights_z, K, syllable_labels, stem):
    """Mean z-scored LDA weight per transition, K x K, black stars where
    |z| exceeds the two-sided 0.05 threshold."""
    mean_w = np.asarray(weights_z).mean(axis=0).reshape(-1)
    if mean_w.size != K * K:
        print(f'(weight vector is {mean_w.size}, expected {K * K}; skipped)')
        return
    idx = np.array(syllable_labels)
    mat = mean_w.reshape(K, K)[np.ix_(idx, idx)]
    star_mask = np.abs(mat) > 1.96

    print('\n' + '=' * 74)
    print('DECODER WEIGHTS (z-scored vs shuffle, averaged over folds)')
    print('=' * 74)
    print('  from\\to ' + ' '.join(f'{s_:>8}' for s_ in syllable_labels))
    for i, s_from in enumerate(syllable_labels):
        print(f'  {s_from:>7} ' +
              ' '.join(f'{mat[i, j]:8.3f}' for j in range(len(idx))))

    flat = [(abs(mat[i, j]), syllable_labels[i], syllable_labels[j], mat[i, j])
            for i in range(len(idx)) for j in range(len(idx))]
    flat.sort(reverse=True)
    print('\nStrongest transitions (|z| > 1.96 marked with a star):')
    for _, a_, b_, v in flat[:8]:
        print(f'  {a_} -> {b_:<3} z = {v:7.3f}  {"*" if abs(v) > 1.96 else ""}')

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    _matrix_panel(ax, mat, syllable_labels, 'LDA Decoder Weights (z-scored)',
                  WEIGHT_CMAP, center=0, cbar_label='Weight (z-score)',
                  star_mask=star_mask)
    plt.tight_layout()
    save_figure(fig, stem)


def plot_decoding_per_barrier(TCMs, meta, syllable_labels, stem,
                              n_shuffles=N_SHUFFLES, level=None, raw=None):
    """One page per barrier, each with three panels: confusion matrix,
    decoder weights, and the delta transition matrix for that barrier."""
    level = level or DECODE_PER
    barriers = sorted(meta['barrier'].dropna().unique())
    idx = np.array(syllable_labels)
    made_any = False

    for barrier in barriers:
        sel = meta['barrier'] == barrier
        raw_b = raw[sel.values] if raw is not None else None
        Xb, Yb, subjects, _, mats_b = build_samples(
            TCMs[sel.values], meta[sel].reset_index(drop=True), level, raw_b)

        if len(np.unique(Yb)) < 2 or len(Yb) < 4:
            print(f'\n(barrier {barrier}: only {len(Yb)} samples / '
                  f'{len(np.unique(Yb))} conditions - skipped)')
            continue

        cm, cz, pv, weights_z, weights_raw = run_decoding(
            Xb, Yb, label=f'Barrier {barrier}cm (per {level})',
            n_shuffles=n_shuffles, unit=level)

        ctrl = mats_b[Yb == 'Control'].mean(axis=0)[np.ix_(idx, idx)]
        cnsds = mats_b[Yb == 'CNSDS'].mean(axis=0)[np.ix_(idx, idx)]
        delta = ctrl - cnsds

        wmat = np.asarray(weights_raw).mean(axis=0).reshape(-1)
        wmat = wmat.reshape(int(np.sqrt(wmat.size)), -1)[np.ix_(idx, idx)]

        print(f'\n--- Barrier {barrier}cm: delta transition matrix ---')
        print('  from\\to ' + ' '.join(f'{s_:>8}' for s_ in syllable_labels))
        for i, s_from in enumerate(syllable_labels):
            print(f'  {s_from:>7} ' +
                  ' '.join(f'{delta[i, j]:8.4f}'
                           for j in range(len(syllable_labels))))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        fig.suptitle(f'Decoding: {barrier}cm barrier', y=1.02, fontsize=15)
        _confusion_panel(axes[0], cm, pv, CONDITIONS,
                         title='Confusion Matrix', cbar=True)
        _matrix_panel(axes[1], wmat, syllable_labels, 'Decoder Weights',
                      WEIGHT_CMAP, center=0)
        _matrix_panel(axes[2], delta, syllable_labels,
                      r'$\Delta$TCM (Ctrl - CNSDS)', DELTA_CMAP, center=0)
        for ax in axes[1:]:
            ax.set_ylabel('')
        plt.tight_layout()
        # One file per barrier: decoding_per_barrier_10cm.pdf, etc.
        save_figure(fig, f'{stem}_{barrier}cm')
        made_any = True

    if not made_any:
        print('\n(no barrier had enough animals for per-barrier decoding)')


# =============================================================================
#                                     MAIN
# =============================================================================

def main():
    start_run()
    np.random.seed(RANDOM_SEED)

    print('=' * 74)
    print('TRANSITION-MATRIX DECODING (LDA)')
    print('=' * 74)
    print(f'project : {PROJECT_DIR}')
    print(f'model   : {MODEL_NAME}   (read-only)')
    print(f'run     : {RUN_ID}')
    print(f'shuffles: {N_SHUFFLES}   seed: {RANDOM_SEED}')

    TCMs, RAW, meta, K = build_transition_features(
        PROJECT_DIR, MODEL_NAME)
    syllable_labels = [s for s in range(K) if s not in EXCLUDE_SYLLABLES]

    print(f'\n{len(TCMs)} recordings, {meta["subject"].nunique()} animals')
    for cond in CONDITIONS:
        n = meta.loc[meta['condition'] == cond, 'subject'].nunique()
        r = int((meta['condition'] == cond).sum())
        print(f'  {cond}: {n} animals, {r} recordings')

    per_animal = meta.groupby('subject').size()
    print(f'\nRecordings per animal: min {per_animal.min()}, '
          f'median {int(per_animal.median())}, max {per_animal.max()}')
    counts = (meta.groupby(['barrier', 'day']).size()
              .rename('recordings').reset_index())
    if not counts.empty:
        print('Recordings per barrier x day (all animals pooled):')
        for _, row in counts.iterrows():
            print(f'    barrier {row["barrier"]}cm  day {row["day"]}: '
                  f'{row["recordings"]}')

    levels = [DECODE_PER]
    if RUN_BOTH_LEVELS:
        levels.append('recording' if DECODE_PER == 'animal' else 'animal')

    for level in levels:
        suffix = f'_per_{level}'
        unit = 'animal' if level == 'animal' else 'recording'

        X, Y, sample_ids, rows, mats = build_samples(TCMs, meta, level, RAW)

        print('\n\n' + '#' * 74)
        print(f'# ANALYSIS LEVEL: PER {level.upper()}')
        print('#' * 74)
        print(f'{len(X)} samples ({X.shape[1]} features = {K}x{K} transitions)')
        for cond in CONDITIONS:
            print(f'  {cond}: {int((Y == cond).sum())} samples')
        if level == 'animal':
            print(f'Aggregation: {ANIMAL_AGGREGATION}')
            print(f'Recordings per animal: min {min(rows)}, max {max(rows)}')
        else:
            print('Each recording is one sample. Recordings from the same '
                  'animal appear\nin both the training and test sets, so part '
                  'of this accuracy reflects\nanimal identity rather than '
                  'condition.')
        if X.shape[1] > len(X):
            print(f'NOTE: {X.shape[1]} features vs {len(X)} samples. LDA is '
                  f'over-parameterised here, which is why the shuffle null '
                  f'matters - compare against it, not against chance.')

        plot_delta_transition_matrix(mats, Y, syllable_labels,
                                     f'delta_transition_matrix{suffix}',
                                     tcm_suffix=suffix)

        confusion_mat, confusion_z, pvalues, weights_z, _ = run_decoding(
            X, Y, label=f'All barriers, per {level}', unit=unit)
        plot_confusion(confusion_mat, pvalues, f'confusion_matrix{suffix}')
        plot_decoder_weights(weights_z, K, syllable_labels,
                             f'decoder_weights{suffix}')

        plot_decoding_per_barrier(TCMs, meta, syllable_labels,
                                  f'decoding_per_barrier{suffix}',
                                  level=level, raw=RAW)


if __name__ == '__main__':
    try:
        main()
    except Exception:
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
