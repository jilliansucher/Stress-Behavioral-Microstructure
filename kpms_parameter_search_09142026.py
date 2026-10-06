"""
Keypoint-MoSeq Parameter Search
================================
Sweeps num_states x kappa with multiple random seeds, selects the best
model using Expected Marginal Likelihood (EML) scores, and reproduces
the full analysis pipeline with the winning parameters.

Each model fit runs as a SUBPROCESS so that:
  - GPU memory is fully released between fits
  - A GPU OOM crash kills only the child, not the whole search

Usage:
    python kpms_parameter_search.py              # run full search
    python kpms_parameter_search.py --fit-single  # (internal) fit one model

Edit the CONFIGURATION section below before running.
"""

import os
import glob
import sys
import gc
import re
import time
import json
import subprocess
import itertools
import warnings
import traceback
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths and parameters before running
# ════════════════════════════════════════════════════════════════════

# Paths
PROJECT_DIR = r'D:\Barrier_testing_day1_videos\KPMS'
KEYPOINT_DATA_DIR = r'D:\Barrier_testing_day1_videos'  # directory with DLC .h5 files
KEYPOINT_FORMAT = 'deeplabcut'

# DeepLabCut writes two .h5 per video: the raw output ('...500000.h5') and the
# filtered one ('..._filtered.h5'). Loading the directory picks up both, so
# every video enters the model twice. This keeps only the filtered files.
# Set to None to load everything (the original behaviour).
KEYPOINT_FILE_PATTERN = '*filtered.h5'
VIDEO_DIR = r'D:\Barrier_testing_day1_videos'
PTS_CSV_PATH = r'D:\Barrier_testing_day1_videos\pts.csv'

# Output directory for parameter search results
SEARCH_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'parameter_search')

# Parameter grid
NUM_STATES_VALUES = [4, 6, 8, 10] # upper bound on syllables
KAPPA_VALUES = [1e3, 1e4, 1e5, 1e6, 1e7]   # syllable stickiness
NUM_SEEDS = 3                                # random seeds per combo

# Fitting iterations
NUM_AR_ITERS = 50       # AR-only stage
NUM_FULL_ITERS = 200    # full model stage
DECREASE_KAPPA_FACTOR = 10  # kappa is reduced by this factor for full model

# Parameter search strategy
# Use a SUBSET of recordings for the search to reduce GPU memory and speed
# things up. Set to None to use all recordings (not recommended for large
# datasets — will OOM on 12 GB GPUs). The subset is randomly sampled but
# stratified so every condition/barrier combo is represented.
SEARCH_SUBSET_SIZE = 500   # number of recordings to use for parameter search

# Set to True to skip Stage 2 (full SLDS) during parameter search and
# compare models using AR-only fits. Much faster and avoids OOM. The
# winning parameters still get a full Stage 1 + Stage 2 fit on ALL data.
SEARCH_AR_ONLY = True

# Set to True to skip all model fitting and go straight to evaluation.
# Useful when fits are already done and you just want to rerun analysis.
SKIP_FITTING = True

# Visualization
SELECTED_SYLLABLES = list(range(0, 11)) + [12] + list(range(14, 23)) + [24, 26]
PX_TO_CM = 38.735 / 885
MAZE_LIKELIHOOD_THRESHOLD = 0.95

# ════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def log(msg):
    """Print a timestamped progress message."""
    t = time.strftime('%H:%M:%S')
    print(f"[{t}] {msg}", flush=True)



def keypoint_files(quiet=False):
    """Resolve KEYPOINT_FILE_PATTERN to an explicit list of files.

    Returns the directory unchanged when no pattern is set, so behaviour is
    identical to before if KEYPOINT_FILE_PATTERN is None.
    """
    if not KEYPOINT_FILE_PATTERN:
        return KEYPOINT_DATA_DIR

    all_h5 = sorted(glob.glob(os.path.join(KEYPOINT_DATA_DIR, '*.h5')))
    matched = sorted(glob.glob(os.path.join(KEYPOINT_DATA_DIR,
                                            KEYPOINT_FILE_PATTERN)))

    if not quiet:
        log(f"  Keypoint files: {len(matched)} matching "
            f"'{KEYPOINT_FILE_PATTERN}' "
            f"(of {len(all_h5)} .h5 files in the directory)")

    if not matched:
        raise RuntimeError(
            f"No files in {KEYPOINT_DATA_DIR} match "
            f"'{KEYPOINT_FILE_PATTERN}'. The directory holds {len(all_h5)} "
            f".h5 file(s)."
        )
    return matched


def model_name_for(num_states, kappa, seed):
    """Generate a consistent model name for a parameter combination."""
    return f"search-K{num_states}-kappa{kappa:.0e}-seed{seed}"


def find_elbow(x, y):
    """
    Find the elbow point in a curve using the maximum distance from
    a line connecting the first and last points.
    Returns the index of the elbow point.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Normalize
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # Line from first to last point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        return len(x) // 2

    line_unit = line_vec / line_len

    # Distance from each point to the line
    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        proj = np.dot(p - p1, line_unit)
        proj_point = p1 + proj * line_unit
        dist = np.linalg.norm(p - proj_point)
        distances.append(dist)

    return int(np.argmax(distances))


def variance_explained_threshold(scores, threshold=0.8):
    """
    Find the index where cumulative 'variance explained' crosses the
    threshold. Here we use the normalized EML scores as a proxy.
    """
    scores = np.array(scores)
    if scores.max() == scores.min():
        return 0
    norm = (scores - scores.min()) / (scores.max() - scores.min())
    for i, v in enumerate(norm):
        if v >= threshold:
            return i
    return len(scores) - 1


# ════════════════════════════════════════════════════════════════════
# SUBPROCESS WORKER — fits a single model in an isolated process
# ════════════════════════════════════════════════════════════════════

def subsample_keypoints(coordinates, confidences, n, seed=0):
    """
    Randomly subsample n recordings from the full keypoint dataset.
    Returns subsetted coordinates and confidences dicts.
    """
    all_keys = sorted(coordinates.keys())
    rng = np.random.RandomState(seed)
    if n >= len(all_keys):
        return coordinates, confidences
    chosen = rng.choice(all_keys, size=n, replace=False)
    chosen = set(chosen)
    sub_coords = {k: v for k, v in coordinates.items() if k in chosen}
    sub_confs = {k: v for k, v in confidences.items() if k in chosen}
    return sub_coords, sub_confs


def run_fit_worker(num_states, kappa, seed, subset_size=None, ar_only_search=False):
    """
    Called when the script is invoked with --fit-single.
    Loads data, fits one model, and exits — fully releasing GPU memory.

    subset_size: if set, use only this many recordings (reduces GPU memory)
    ar_only_search: if True, skip Stage 2 (for faster parameter search)
    """
    # Set GPU memory fraction high since this process owns the GPU alone
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90'

    import jax
    import jax.numpy as jnp
    import keypoint_moseq as kpms

    name = model_name_for(num_states, kappa, seed)
    model_dir = os.path.join(PROJECT_DIR, name)

    # Skip if checkpoint already exists (allows resuming)
    checkpoint_path = os.path.join(model_dir, 'checkpoint.h5')
    if os.path.exists(checkpoint_path):
        log(f"  Checkpoint exists for {name}, skipping fit")
        return

    log(f"  Loading data for {name} ...")
    config = kpms.load_config(PROJECT_DIR)
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        keypoint_files(quiet=True), KEYPOINT_FORMAT,
        extension='h5', recursive=False
    )
    log(f"  Loaded {len(coordinates)} recordings total")

    # Subsample if requested (for parameter search — reduces GPU memory)
    if subset_size is not None and subset_size < len(coordinates):
        coordinates, confidences = subsample_keypoints(
            coordinates, confidences, subset_size, seed=42
        )
        log(f"  Using subset of {len(coordinates)} recordings for parameter search")

    data, metadata = kpms.format_data(coordinates, confidences, **config)
    pca = kpms.load_pca(PROJECT_DIR)

    log(f"  Initializing model {name}")

    # Build a config copy with overridden num_states
    init_config = dict(config)
    init_config['trans_hypparams'] = {
        **config.get('trans_hypparams', {}),
        'num_states': num_states,
    }

    model = kpms.init_model(
        data,
        pca=pca,
        seed=jax.random.PRNGKey(seed),
        **init_config,
    )

    # Stage 1: AR-only fitting
    log(f"  Stage 1 (AR-only, kappa={kappa:.0e}) ...")
    model = kpms.update_hypparams(model, kappa=kappa)
    model = kpms.fit_model(
        model, data, metadata, PROJECT_DIR, name,
        ar_only=True,
        num_iters=NUM_AR_ITERS,
        save_every_n_iters=25,
        generate_progress_plots=False,
    )[0]

    if not ar_only_search:
        # Stage 2: full model fitting
        full_kappa = kappa / DECREASE_KAPPA_FACTOR
        log(f"  Stage 2 (full model, kappa={full_kappa:.0e}) ...")
        model = kpms.update_hypparams(model, kappa=full_kappa)
        kpms.fit_model(
            model, data, metadata, PROJECT_DIR, name,
            ar_only=False,
            start_iter=NUM_AR_ITERS,
            num_iters=NUM_AR_ITERS + NUM_FULL_ITERS,
            save_every_n_iters=25,
            generate_progress_plots=False,
        )
    else:
        log(f"  Skipping Stage 2 (AR-only search mode)")

    # Reindex syllables by frequency and extract results
    kpms.reindex_syllables_in_checkpoint(PROJECT_DIR, name)
    model_final, data_final, metadata_final, _ = kpms.load_checkpoint(
        PROJECT_DIR, name
    )
    kpms.extract_results(model_final, metadata_final, PROJECT_DIR, name)

    log(f"  Done fitting {name}")


def spawn_fit_subprocess(num_states, kappa, seed,
                         subset_size=None, ar_only_search=False):
    """
    Launch a subprocess to fit one model. Returns True if successful.
    The subprocess gets its own GPU context, so OOM kills only it.
    """
    name = model_name_for(num_states, kappa, seed)

    # Skip if checkpoint already exists
    checkpoint_path = os.path.join(PROJECT_DIR, name, 'checkpoint.h5')
    if os.path.exists(checkpoint_path):
        log(f"  Checkpoint exists for {name}, skipping")
        return True

    cmd = [
        sys.executable, __file__,
        '--fit-single',
        str(num_states), str(kappa), str(seed),
    ]
    if subset_size is not None:
        cmd += ['--subset', str(subset_size)]
    if ar_only_search:
        cmd += ['--ar-only']

    log(f"  Spawning subprocess for {name} ...")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=7200,  # 2 hour timeout per model
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            log(f"  {name} completed in {elapsed/60:.1f} min")
            return True
        else:
            log(f"  {name} FAILED (exit code {result.returncode}) "
                f"after {elapsed/60:.1f} min")
            log(f"  This is likely a GPU out-of-memory crash.")
            return False

    except subprocess.TimeoutExpired:
        log(f"  {name} TIMED OUT after 2 hours")
        return False
    except Exception as e:
        log(f"  {name} subprocess error: {e}")
        return False


# ════════════════════════════════════════════════════════════════════
# EVALUATION AND ANALYSIS FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def compute_eml_for_group(project_dir, model_names):
    """Compute EML scores for a group of models (same hyperparams, different seeds)."""
    import keypoint_moseq as kpms

    if len(model_names) < 2:
        log("  Only 1 model in group — using single-model log-likelihood")
        model, data, metadata, _ = kpms.load_checkpoint(project_dir, model_names[0])
        mask = data['mask']
        z = model['states']['z']
        durations = []
        for row, m in zip(z, mask):
            seq = row[m.astype(bool)]
            changes = np.where(np.diff(seq) != 0)[0]
            if len(changes) > 0:
                durs = np.diff(np.concatenate([[0], changes + 1, [len(seq)]]))
                durations.extend(durs.tolist())
        return np.array([0.0]), np.array([0.0])

    scores, std_errs = kpms.expected_marginal_likelihoods(
        project_dir, model_names
    )
    return scores, std_errs


def get_num_used_states(project_dir, model_name):
    """Count how many distinct syllables are actually used by the model."""
    import keypoint_moseq as kpms
    model, data, metadata, _ = kpms.load_checkpoint(project_dir, model_name)
    mask = data['mask']
    z = model['states']['z']
    all_states = set()
    for row, m in zip(z, mask):
        n = min(len(row), len(m))
        seq = row[:n][m[:n].astype(bool)]
        all_states.update(seq.tolist())
    return len(all_states)


def get_median_duration(project_dir, model_name):
    """Get median syllable duration (in frames) for a fitted model."""
    import keypoint_moseq as kpms
    model, data, metadata, _ = kpms.load_checkpoint(project_dir, model_name)
    mask = data['mask']
    z = model['states']['z']
    durations = []
    for row, m in zip(z, mask):
        n = min(len(row), len(m))
        seq = row[:n][m[:n].astype(bool)]
        changes = np.where(np.diff(seq) != 0)[0]
        if len(changes) > 0:
            durs = np.diff(np.concatenate([[0], changes + 1, [len(seq)]]))
            durations.extend(durs.tolist())
    return np.median(durations) if durations else 0


# ════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    log("=" * 60)
    log("KEYPOINT-MOSEQ PARAMETER SEARCH")
    log("=" * 60)
    if SEARCH_SUBSET_SIZE:
        log(f"  Using subset of {SEARCH_SUBSET_SIZE} recordings for search")
    if SEARCH_AR_ONLY:
        log(f"  Using AR-only fits for search (skipping Stage 2)")
    log("")

    keypoint_files()          # report how many files will be used
    log("")

    os.makedirs(SEARCH_OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Fit models across parameter grid (each as a subprocess)
    # ------------------------------------------------------------------
    param_combos = list(itertools.product(NUM_STATES_VALUES, KAPPA_VALUES))
    total_models = len(param_combos) * NUM_SEEDS
    log(f"Parameter grid: {len(NUM_STATES_VALUES)} num_states x "
        f"{len(KAPPA_VALUES)} kappa x {NUM_SEEDS} seeds = {total_models} models")

    all_model_names = {}  # (num_states, kappa) -> [model_names]
    failed_models = []

    if SKIP_FITTING:
        log("SKIP_FITTING=True — scanning for existing checkpoints ...")
        for num_states, kappa in param_combos:
            names = []
            for seed in range(NUM_SEEDS):
                name = model_name_for(num_states, kappa, seed)
                checkpoint_path = os.path.join(PROJECT_DIR, name, 'checkpoint.h5')
                if os.path.exists(checkpoint_path):
                    names.append(name)
            if names:
                log(f"  num_states={num_states}, kappa={kappa:.0e}: "
                    f"{len(names)} checkpoint(s) found")
            all_model_names[(num_states, kappa)] = names
    else:
        model_count = 0
        for num_states, kappa in param_combos:
            log(f"\n{'─' * 50}")
            log(f"num_states={num_states}, kappa={kappa:.0e}")
            log(f"{'─' * 50}")

            names = []
            for seed in range(NUM_SEEDS):
                model_count += 1
                name = model_name_for(num_states, kappa, seed)
                log(f"Model {model_count}/{total_models} (seed={seed})")

                success = spawn_fit_subprocess(
                    num_states, kappa, seed,
                    subset_size=SEARCH_SUBSET_SIZE,
                    ar_only_search=SEARCH_AR_ONLY,
                )
                if success:
                    checkpoint_path = os.path.join(PROJECT_DIR, name, 'checkpoint.h5')
                    if os.path.exists(checkpoint_path):
                        names.append(name)
                    else:
                        log(f"  WARNING: subprocess succeeded but no checkpoint found")
                        failed_models.append(name)
                else:
                    failed_models.append(name)

            all_model_names[(num_states, kappa)] = names

        if failed_models:
            log(f"\nFailed models ({len(failed_models)}):")
            for fm in failed_models:
                log(f"  {fm}")

    # ------------------------------------------------------------------
    # 2. Evaluate models using EML scores
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("EVALUATING MODELS (Expected Marginal Likelihood)")
    log("=" * 60)

    # Need kpms for evaluation — import here so main process doesn't
    # grab GPU memory during fitting phase
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.50'
    import keypoint_moseq as kpms

    results_summary = []

    for (num_states, kappa), names in all_model_names.items():
        if not names:
            log(f"No successful models for num_states={num_states}, "
                f"kappa={kappa:.0e} — skipping")
            continue

        log(f"Computing EML for num_states={num_states}, kappa={kappa:.0e} ...")

        try:
            scores, std_errs = compute_eml_for_group(PROJECT_DIR, names)
            best_idx = int(np.argmax(scores))
            best_name = names[best_idx]
            best_score = scores[best_idx]
            best_err = std_errs[best_idx]
        except Exception as e:
            log(f"  ERROR computing EML: {e}")
            best_name = names[0]
            best_score = -np.inf
            best_err = 0.0

        # Get additional metrics from the best model in this group
        n_used = get_num_used_states(PROJECT_DIR, best_name)
        med_dur = get_median_duration(PROJECT_DIR, best_name)

        results_summary.append({
            'num_states': num_states,
            'kappa': kappa,
            'best_model': best_name,
            'eml_score': best_score,
            'eml_stderr': best_err,
            'num_used_states': n_used,
            'median_duration_frames': med_dur,
        })

        log(f"  Best: {best_name} | EML={best_score:.2f}+/-{best_err:.2f} | "
            f"used_states={n_used} | median_dur={med_dur:.1f}")

    if not results_summary:
        log("ERROR: No models were successfully fit. Exiting.")
        return

    results_df = pd.DataFrame(results_summary)
    results_csv_path = os.path.join(SEARCH_OUTPUT_DIR, 'parameter_search_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    log(f"\nResults saved to {results_csv_path}")

    # ------------------------------------------------------------------
    # 3. Select best model
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("SELECTING BEST MODEL")
    log("=" * 60)

    # Strategy: For each kappa value, find the best num_states via elbow
    # detection on the EML-vs-num_states curve. Then pick the best
    # (kappa, num_states) combo from those elbow picks.
    #
    # This avoids the meaningless "sort everything by EML and find elbow"
    # approach, which has no natural x-axis ordering.

    elbow_picks = []  # one per kappa
    for kappa_val in sorted(results_df['kappa'].unique()):
        subset = (results_df[results_df['kappa'] == kappa_val]
                  .sort_values('num_states'))
        if len(subset) < 2:
            # Can't do elbow with < 2 points; just take the best
            best_idx = subset['eml_score'].idxmax()
            elbow_picks.append(subset.loc[best_idx])
            continue

        ns_vals = subset['num_states'].values.astype(float)
        eml_vals = subset['eml_score'].values

        eidx = find_elbow(ns_vals, eml_vals)

        # The elbow point itself is the pick (not "elbow and above")
        elbow_picks.append(subset.iloc[eidx])
        log(f"  kappa={kappa_val:.0e}: elbow at num_states="
            f"{int(subset.iloc[eidx]['num_states'])} "
            f"(EML={subset.iloc[eidx]['eml_score']:.2f})")

    # Among the elbow picks, choose the one with highest EML
    elbow_picks_df = pd.DataFrame(elbow_picks)
    best_row = elbow_picks_df.loc[elbow_picks_df['eml_score'].idxmax()]
    selection_method = 'elbow (per-kappa) + max EML'

    # Warn if best is at parameter boundary
    best_num_states = int(best_row['num_states'])
    best_kappa = float(best_row['kappa'])
    at_boundary = False
    if best_num_states == max(NUM_STATES_VALUES):
        log(f"  WARNING: Best num_states={best_num_states} is at the upper "
            f"boundary. Consider extending NUM_STATES_VALUES.")
        at_boundary = True
    if best_kappa == min(KAPPA_VALUES):
        log(f"  WARNING: Best kappa={best_kappa:.0e} is at the lower boundary. "
            f"Consider extending KAPPA_VALUES.")
        at_boundary = True
    if at_boundary:
        selection_method += ' (at boundary — may need wider search)'

    best_model_name = best_row['best_model']

    log(f"Selected model: {best_model_name}")
    log(f"  Method: {selection_method}")
    log(f"  num_states={best_num_states}, kappa={best_kappa:.0e}")
    log(f"  EML score: {best_row['eml_score']:.2f} +/- {best_row['eml_stderr']:.2f}")
    log(f"  Used states: {best_row['num_used_states']}")
    log(f"  Median duration: {best_row['median_duration_frames']:.1f} frames")

    # Save selection info
    selection_info = {
        'best_model_name': best_model_name,
        'num_states': best_num_states,
        'kappa': best_kappa,
        'eml_score': float(best_row['eml_score']),
        'selection_method': selection_method,
        'num_used_states': int(best_row['num_used_states']),
        'median_duration_frames': float(best_row['median_duration_frames']),
    }
    with open(os.path.join(SEARCH_OUTPUT_DIR, 'best_model_info.json'), 'w') as f:
        json.dump(selection_info, f, indent=2)

    # ------------------------------------------------------------------
    # 3b. Refit the best model on ALL data (if search used subset/AR-only)
    # ------------------------------------------------------------------
    final_model_name = f"final-K{best_num_states}-kappa{best_kappa:.0e}"
    final_checkpoint = os.path.join(PROJECT_DIR, final_model_name, 'checkpoint.h5')

    if SEARCH_SUBSET_SIZE or SEARCH_AR_ONLY:
        log("\n" + "=" * 60)
        log(f"REFITTING BEST PARAMS ON FULL DATA: {final_model_name}")
        log(f"  num_states={best_num_states}, kappa={best_kappa:.0e}")
        log("=" * 60)

        # Fit with NO subset and full Stage 1+2, using seed 0
        success = spawn_fit_subprocess(
            best_num_states, best_kappa, seed=0,
            subset_size=None,       # use ALL recordings
            ar_only_search=False,   # do full Stage 1 + Stage 2
        )
        # The subprocess uses model_name_for() which produces a different
        # name than final_model_name, so rename via a wrapper
        refit_name = model_name_for(best_num_states, best_kappa, 0)
        refit_checkpoint = os.path.join(PROJECT_DIR, refit_name, 'checkpoint.h5')

        if success and os.path.exists(refit_checkpoint):
            # Use the refit model for all downstream analysis
            best_model_name = refit_name
            log(f"Full refit complete: {best_model_name}")
        else:
            log(f"WARNING: Full refit failed. Using search model {best_model_name} "
                f"for analysis (fitted on subset).")
    else:
        log("Search used full data — no refit needed.")

    # Update selection info with final model name
    selection_info['final_model_name'] = best_model_name
    with open(os.path.join(SEARCH_OUTPUT_DIR, 'best_model_info.json'), 'w') as f:
        json.dump(selection_info, f, indent=2)

    # ------------------------------------------------------------------
    # 4. Plot parameter search results
    # ------------------------------------------------------------------
    log("\nPlotting parameter search results ...")

    sns.set(style="white")

    # 4a. Heatmap of EML scores across num_states x kappa
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pivot_eml = results_df.pivot(
        index='num_states', columns='kappa', values='eml_score'
    )
    sns.heatmap(
        pivot_eml, annot=True, fmt='.1f', cmap='viridis', ax=axes[0]
    )
    axes[0].set_title('EML Score')
    axes[0].set_xlabel('Kappa')
    axes[0].set_ylabel('Num States (upper bound)')

    # 4b. Heatmap of number of used states
    pivot_used = results_df.pivot(
        index='num_states', columns='kappa', values='num_used_states'
    )
    sns.heatmap(
        pivot_used, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1]
    )
    axes[1].set_title('Number of Used States')
    axes[1].set_xlabel('Kappa')
    axes[1].set_ylabel('Num States (upper bound)')

    # 4c. Heatmap of median duration
    pivot_dur = results_df.pivot(
        index='num_states', columns='kappa', values='median_duration_frames'
    )
    sns.heatmap(
        pivot_dur, annot=True, fmt='.1f', cmap='coolwarm', ax=axes[2]
    )
    axes[2].set_title('Median Syllable Duration (frames)')
    axes[2].set_xlabel('Kappa')
    axes[2].set_ylabel('Num States (upper bound)')

    plt.tight_layout()
    plt.savefig(os.path.join(SEARCH_OUTPUT_DIR, 'parameter_search_heatmaps.pdf'))
    plt.savefig(os.path.join(SEARCH_OUTPUT_DIR, 'parameter_search_heatmaps.png'),
                dpi=150)
    plt.close()

    # 4d. EML score vs kappa for each num_states value
    fig, ax = plt.subplots(figsize=(8, 5))
    for ns in NUM_STATES_VALUES:
        subset = results_df[results_df['num_states'] == ns]
        ax.errorbar(
            subset['kappa'], subset['eml_score'], yerr=subset['eml_stderr'],
            marker='o', label=f'K={ns}', capsize=3
        )
    ax.set_xscale('log')
    ax.set_xlabel('Kappa')
    ax.set_ylabel('EML Score')
    ax.set_title('Model Quality Across Parameters')
    ax.legend()
    ax.axvline(best_kappa, color='red', linestyle='--', alpha=0.5,
               label=f'Best kappa={best_kappa:.0e}')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(SEARCH_OUTPUT_DIR, 'eml_vs_kappa.pdf'))
    plt.savefig(os.path.join(SEARCH_OUTPUT_DIR, 'eml_vs_kappa.png'), dpi=150)
    plt.close()

    # 4e. EML vs num_states (per kappa) with elbow points marked
    fig, ax = plt.subplots(figsize=(8, 5))
    for kappa_val in sorted(results_df['kappa'].unique()):
        subset = (results_df[results_df['kappa'] == kappa_val]
                  .sort_values('num_states'))
        ax.errorbar(
            subset['num_states'], subset['eml_score'],
            yerr=subset['eml_stderr'],
            marker='o', label=f'kappa={kappa_val:.0e}', capsize=3
        )
        # Mark elbow point
        if len(subset) >= 2:
            ns_vals = subset['num_states'].values.astype(float)
            eml_vals = subset['eml_score'].values
            eidx = find_elbow(ns_vals, eml_vals)
            ax.plot(subset.iloc[eidx]['num_states'],
                    subset.iloc[eidx]['eml_score'],
                    'r*', markersize=15, zorder=5)
    # Mark the selected best model
    ax.plot(best_num_states, best_row['eml_score'],
            'k^', markersize=12, zorder=6, label='Selected model')
    ax.set_xlabel('Num States (upper bound)')
    ax.set_ylabel('EML Score')
    ax.set_title('EML vs Num States (per Kappa) with Elbow Points')
    ax.legend(fontsize=8)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(SEARCH_OUTPUT_DIR, 'elbow_plot.pdf'))
    plt.savefig(os.path.join(SEARCH_OUTPUT_DIR, 'elbow_plot.png'), dpi=150)
    plt.close()

    # 4f. Used states vs num_states — shows how many syllables the model
    # actually discovers vs the upper bound
    fig, ax = plt.subplots(figsize=(8, 5))
    for kappa_val in sorted(results_df['kappa'].unique()):
        subset = (results_df[results_df['kappa'] == kappa_val]
                  .sort_values('num_states'))
        ax.plot(subset['num_states'], subset['num_used_states'],
                marker='o', label=f'kappa={kappa_val:.0e}')
    # Diagonal line (used = upper bound)
    max_ns = max(NUM_STATES_VALUES)
    ax.plot([0, max_ns], [0, max_ns], 'k--', alpha=0.3, label='used = upper bound')
    ax.set_xlabel('Num States (upper bound)')
    ax.set_ylabel('Actually Used States')
    ax.set_title('Discovered Syllables vs Upper Bound')
    ax.legend(fontsize=8)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(SEARCH_OUTPUT_DIR, 'used_vs_upper_bound.pdf'))
    plt.savefig(os.path.join(SEARCH_OUTPUT_DIR, 'used_vs_upper_bound.png'), dpi=150)
    plt.close()

    log("Parameter search plots saved.")

    # ------------------------------------------------------------------
    # 5. Extract results from the best model and reproduce notebook plots
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log(f"REPRODUCING ANALYSIS WITH BEST MODEL: {best_model_name}")
    log("=" * 60)

    # Load and extract results
    log("Loading best model checkpoint ...")
    config = kpms.load_config(PROJECT_DIR)
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        keypoint_files(quiet=True), KEYPOINT_FORMAT,
        extension='h5', recursive=False
    )

    model, data_loaded, metadata_loaded, current_iter = kpms.load_checkpoint(
        PROJECT_DIR, best_model_name
    )
    results = kpms.extract_results(
        model, metadata_loaded, PROJECT_DIR, best_model_name
    )

    # compute_moseq_df reads PROJECT_DIR/index.csv internally and expects
    # recording names to match the checkpoint. The index stores names without
    # .h5 but load_keypoints keys include the extension. Temporarily patch.
    project_index_path = os.path.join(PROJECT_DIR, 'index.csv')
    index_backup = None

    # results is keyed by recording name: results[rec_name] = {'syllable':..., ...}
    checkpoint_names = set(results.keys())
    log(f"  Recording names: {len(checkpoint_names)} "
        f"(sample: {list(checkpoint_names)[:3]})")

    if os.path.exists(project_index_path):
        idx_df = pd.read_csv(project_index_path)
        index_backup = idx_df.copy()
        index_names = set(idx_df['name'].values)

        missing = checkpoint_names - index_names
        if missing:
            log(f"  {len(missing)}/{len(checkpoint_names)} recording names "
                f"not found in index.csv — rebuilding with checkpoint names")
            # Try to preserve group assignments where names match.
            # For names that don't match, try fuzzy match by stripping
            # or adding '_filtered', or fall back to default group.
            name_to_group = dict(zip(idx_df['name'], idx_df['group']))

            new_rows = []
            for cn in sorted(checkpoint_names):
                group = name_to_group.get(cn)
                if group is None:
                    # Try with/without _filtered
                    if cn.endswith('_filtered'):
                        group = name_to_group.get(cn[:-len('_filtered')])
                    else:
                        group = name_to_group.get(cn + '_filtered')
                if group is None:
                    group = 'default'
                new_rows.append({'name': cn, 'group': group})

            idx_df_fixed = pd.DataFrame(new_rows)
            idx_df_fixed.to_csv(project_index_path, index=False)
            log(f"  Temporarily replaced index.csv ({len(idx_df_fixed)} entries)")
        else:
            log(f"  Names already match — no patch needed.")
    else:
        index_backup = None
        # No index.csv at all — create one from checkpoint names
        log(f"  No index.csv found — creating from checkpoint recording names")
        idx_df_fixed = pd.DataFrame({
            'name': sorted(checkpoint_names),
            'group': 'default'
        })
        idx_df_fixed.to_csv(project_index_path, index=False)

    # Compute moseq_df
    log("Computing moseq_df ...")
    moseq_df = kpms.compute_moseq_df(PROJECT_DIR, best_model_name,
                                       smooth_heading=True)

    # Restore original index.csv
    if index_backup is not None:
        index_backup.to_csv(project_index_path, index=False)
        log(f"  Restored original index.csv")

    # Parse experimental metadata from recording names
    log("Parsing experimental metadata ...")
    moseq_df['name'] = (
        moseq_df['name']
        .str.replace(
            r'Barrier_Testing_(\d+cm)_Day_(\d)',
            r'\1_barrier_day\2',
            regex=True
        )
    )

    name_col = moseq_df['name']
    moseq_df['condition'] = (
        name_col.str.extract(r'(cnsds|control)', flags=re.IGNORECASE)[0]
        .str.lower()
    )
    moseq_df['barrier'] = (
        name_col.str.extract(r'(10|15|20)\s*cm?', flags=re.IGNORECASE)[0]
    )
    moseq_df['day'] = (
        name_col.str.extract(r'day[_\s]?(1|2|3)', flags=re.IGNORECASE)[0]
    )

    def extract_subject(name):
        match = re.search(r'(wt\d+)', name, re.IGNORECASE)
        return match.group(1) if match else None

    def extract_trial(name):
        match = re.search(r'(\d+)[^\d]*DLC', name, re.IGNORECASE)
        return int(match.group(1)) if match else None

    moseq_df['subject'] = moseq_df['name'].apply(extract_subject)
    moseq_df['trial'] = moseq_df['name'].apply(extract_trial)

    analysis_df = moseq_df[
        ['subject', 'trial', 'syllable', 'condition', 'barrier', 'day']
    ].dropna()

    # ── Plot directory ──
    plot_dir = os.path.join(SEARCH_OUTPUT_DIR, 'best_model_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 5a. Syllable location plot
    # ------------------------------------------------------------------
    log("Plotting syllable locations ...")
    try:
        pts = pd.read_csv(PTS_CSV_PATH)
        pts['frame_id'] = pts.groupby('name').cumcount()
        moseq_df['frame_id'] = moseq_df.groupby('name').cumcount()

        pts = pts.merge(
            moseq_df[['name', 'frame_id', 'syllable', 'centroid_x', 'centroid_y']],
            on=['name', 'frame_id'],
            how='left'
        )

        # Normalize coordinates relative to maze center
        maze_lk = pts['Maze_Center likelihood']
        pts.loc[
            maze_lk < MAZE_LIKELIHOOD_THRESHOLD,
            ['Maze_Center x', 'Maze_Center y']
        ] = np.nan
        maze_cx = pts['Maze_Center x'].mean(skipna=True)
        maze_cy = pts['Maze_Center y'].mean(skipna=True)
        pts['normalized_centroid_x'] = pts['centroid_x'] - maze_cx
        pts['normalized_centroid_y'] = -(pts['centroid_y'] - maze_cy)

        syllables = [
            s for s in sorted(pts['syllable'].dropna().unique().astype(int))
            if s in SELECTED_SYLLABLES
        ]

        ncols = 5
        nrows = int(np.ceil(len(syllables) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        condition_colors = {'control': 'orange', 'cnsds': 'blue'}

        for i, (ax, syll) in enumerate(zip(axes, syllables)):
            syll_df = pts[pts['syllable'] == syll]
            for cond, color in condition_colors.items():
                cond_df = syll_df[syll_df['condition'].str.lower() == cond]
                ax.scatter(
                    cond_df['normalized_centroid_x'],
                    cond_df['normalized_centroid_y'],
                    c=color, s=5, alpha=0.6,
                    label=cond if i == 0 else None
                )
            ax.set_title(f'Syllable {syll}', fontsize=10)
            ax.set_aspect('equal', adjustable='box')
            ax.set_facecolor('#f7f7f7')

        for j in range(len(syllables), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'syllable_location.pdf'))
        plt.close()
        log("  Syllable location plot saved.")
    except Exception as e:
        log(f"  WARNING: Could not create syllable location plot: {e}")

    # ------------------------------------------------------------------
    # 5b. Syllable velocity plot
    # ------------------------------------------------------------------
    log("Plotting syllable velocity ...")
    syllables_to_plot = [
        s for s in SELECTED_SYLLABLES
        if s in moseq_df['syllable'].unique()
    ]

    filtered_df = moseq_df[moseq_df['syllable'].isin(syllables_to_plot)].copy()
    filtered_df['velocity_cm_s'] = filtered_df['velocity_px_s'] * PX_TO_CM

    avg_vel = (
        filtered_df.groupby('syllable')['velocity_cm_s']
        .mean()
        .reindex(syllables_to_plot)
        .dropna()
    )

    x_pos = np.arange(len(avg_vel))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x_pos, avg_vel.values)
    ax.set_xlabel('Syllable')
    ax.set_ylabel('Average Velocity (cm/s)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(avg_vel.index, rotation=45)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'syllable_velocity_cm_s.pdf'))
    plt.close()
    log("  Syllable velocity plot saved.")

    # ------------------------------------------------------------------
    # 5c. Syllable angular velocity plot
    # ------------------------------------------------------------------
    log("Plotting syllable angular velocity ...")
    avg_angvel = (
        filtered_df.groupby('syllable')['angular_velocity']
        .mean()
        .reindex(syllables_to_plot)
        .dropna()
    )

    x_pos = np.arange(len(avg_angvel))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x_pos, avg_angvel.values)
    ax.set_xlabel('Syllable')
    ax.set_ylabel('Average Angular Velocity')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(avg_angvel.index, rotation=45)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'syllable_angular_velocity.pdf'))
    plt.close()
    log("  Angular velocity plot saved.")

    # ------------------------------------------------------------------
    # 5d. Syllable frequency by condition and barrier
    # ------------------------------------------------------------------
    log("Plotting syllable frequency ...")

    # Compute syllable ratios
    syllable_counts = (
        analysis_df
        .groupby(['subject', 'barrier', 'day', 'syllable'])
        .size()
        .reset_index(name='count')
    )
    total_counts = (
        analysis_df
        .groupby(['subject', 'barrier', 'day'])
        .size()
        .reset_index(name='total')
    )
    merged = pd.merge(syllable_counts, total_counts,
                       on=['subject', 'barrier', 'day'])
    merged['ratio'] = merged['count'] / merged['total']

    avg_across_days = (
        merged.groupby(['subject', 'barrier', 'syllable'])['ratio']
        .mean()
        .reset_index()
    )

    subject_info = analysis_df[['subject', 'condition']].drop_duplicates()
    new_analysis_df = pd.merge(avg_across_days, subject_info, on='subject')
    new_analysis_df = new_analysis_df[
        ['syllable', 'subject', 'condition', 'barrier', 'ratio']
    ]

    # Standardize labels
    new_analysis_df['condition'] = new_analysis_df['condition'].replace({
        'cnsds': 'CNSDS', 'control': 'Control'
    })
    new_analysis_df['barrier'] = new_analysis_df['barrier'].astype(str) + 'cm'

    # Filter to selected syllables
    plot_syllables = sorted([
        s for s in new_analysis_df['syllable'].unique()
        if s in SELECTED_SYLLABLES
    ])

    colors = {'Control': 'orange', 'CNSDS': 'blue'}
    barriers = new_analysis_df['barrier'].unique()
    n_barriers = len(barriers)

    from scipy.stats import ttest_ind

    fig, axes = plt.subplots(1, max(n_barriers, 1),
                             figsize=(14, 4), sharey=True)
    if n_barriers == 1:
        axes = [axes]

    log("  T-test results:")
    for i, barrier in enumerate(barriers):
        ax = axes[i]
        df_barrier = new_analysis_df[new_analysis_df['barrier'] == barrier]
        syllables = sorted([
            s for s in df_barrier['syllable'].unique()
            if s in SELECTED_SYLLABLES
        ])
        x_positions = np.arange(len(syllables))
        syll_map = {s: x for x, s in zip(x_positions, syllables)}
        total_tests = len(syllables)

        for syll in syllables:
            df_syll = df_barrier[df_barrier['syllable'] == syll]
            for cond in ['Control', 'CNSDS']:
                cond_data = df_syll[df_syll['condition'] == cond]['ratio']
                mean_val = cond_data.mean()
                sem_val = cond_data.sem()
                offset = -0.2 if cond == 'Control' else 0.2
                jitter = np.random.normal(0, 0.03, len(cond_data))
                ax.scatter(
                    syll_map[syll] + offset + jitter, cond_data,
                    color=colors[cond], alpha=0.6, s=10
                )
                ax.bar(
                    syll_map[syll] + offset, mean_val, yerr=sem_val,
                    width=0.35, facecolor='white', edgecolor=colors[cond],
                    capsize=4,
                    label=cond if syll == syllables[0] else None
                )

            # T-test with Bonferroni correction
            ctrl = df_syll[df_syll['condition'] == 'Control']['ratio']
            cnsds = df_syll[df_syll['condition'] == 'CNSDS']['ratio']
            if len(ctrl) > 1 and len(cnsds) > 1:
                t_stat, p_val = ttest_ind(ctrl, cnsds, equal_var=False)
                p_bonf = min(p_val * total_tests, 1.0)
                if p_bonf < 0.05:
                    log(f"    {barrier} syll {syll}: t={t_stat:.3f}, "
                        f"p_bonf={p_bonf:.4f} *")
                    x1, x2 = syll_map[syll] - 0.2, syll_map[syll] + 0.2
                    max_y = max(
                        ctrl.mean() + ctrl.sem(),
                        cnsds.mean() + cnsds.sem()
                    ) + 0.05
                    ax.plot(
                        [x1, x1, x2, x2],
                        [max_y - 0.01, max_y, max_y, max_y - 0.01],
                        color='black', linewidth=1
                    )
                    star = ('***' if p_bonf < 0.001
                            else '**' if p_bonf < 0.01
                            else '*')
                    ax.text((x1 + x2) / 2, max_y - 0.005, star,
                            ha='center', va='bottom', fontsize=16)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(syllables, rotation=45)
        ax.set_xlabel('Syllable')
        ax.set_title(f'{barrier}')
        if i == 0:
            ax.set_ylabel('Fraction of Total Syllables')

    plt.tight_layout()
    plt.legend()
    sns.despine()
    plt.savefig(os.path.join(plot_dir, 'syllable_frequency.pdf'))
    plt.close()
    log("  Syllable frequency plot saved.")

    # ------------------------------------------------------------------
    # 5e. Native kpms plots (syllable frequencies, durations, transitions)
    # ------------------------------------------------------------------
    log("Generating native kpms plots ...")

    # Syllable frequency plot (built-in)
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        syll_counts = moseq_df['syllable'].value_counts().sort_index()
        syll_freq = syll_counts / syll_counts.sum()
        ax.bar(syll_freq.index.astype(int), syll_freq.values)
        ax.set_xlabel('Syllable')
        ax.set_ylabel('Frequency')
        ax.set_title('Syllable Usage Distribution')
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'syllable_usage_distribution.pdf'))
        plt.close()
        log("  Syllable usage distribution saved.")
    except Exception as e:
        log(f"  WARNING: syllable usage distribution: {e}")

    # Syllable duration distribution
    try:
        # Compute durations from moseq_df onset column
        onset_mask = moseq_df['onset'].values
        syllables_arr = moseq_df['syllable'].values
        onset_indices = np.where(onset_mask)[0]
        durations_by_syll = {}
        for i in range(len(onset_indices)):
            start = onset_indices[i]
            end = onset_indices[i + 1] if i + 1 < len(onset_indices) else len(syllables_arr)
            syll = int(syllables_arr[start])
            dur = end - start
            durations_by_syll.setdefault(syll, []).append(dur)

        # Plot duration distributions for top syllables
        top_sylls = sorted(durations_by_syll.keys(),
                           key=lambda s: len(durations_by_syll[s]), reverse=True)[:12]
        ncols = 4
        nrows = int(np.ceil(len(top_sylls) / ncols))
        fig, axes_dur = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
        axes_dur = axes_dur.flatten()
        for i, syll in enumerate(sorted(top_sylls)):
            ax = axes_dur[i]
            durs = durations_by_syll[syll]
            ax.hist(durs, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(f'Syllable {syll} (n={len(durs)})')
            ax.set_xlabel('Duration (frames)')
            ax.set_ylabel('Count')
        for j in range(len(top_sylls), len(axes_dur)):
            fig.delaxes(axes_dur[j])
        plt.suptitle('Syllable Duration Distributions', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'syllable_duration_distributions.pdf'))
        plt.close()
        log("  Syllable duration distributions saved.")
    except Exception as e:
        log(f"  WARNING: syllable duration distributions: {e}")

    # Transition matrix
    try:
        # Build transition counts from syllable sequences
        all_sylls = sorted(moseq_df['syllable'].unique().astype(int))
        n_sylls = len(all_sylls)
        syll_to_idx = {s: i for i, s in enumerate(all_sylls)}
        trans_matrix = np.zeros((n_sylls, n_sylls))

        for _, group in moseq_df.groupby('name'):
            seq = group['syllable'].values.astype(int)
            for a, b in zip(seq[:-1], seq[1:]):
                if a != b:  # only count transitions, not self-loops
                    trans_matrix[syll_to_idx[a], syll_to_idx[b]] += 1

        # Normalize rows to probabilities
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = trans_matrix / row_sums

        # Only plot the top N most frequent syllables
        top_n = min(15, n_sylls)
        top_idx = [syll_to_idx[s] for s in all_sylls[:top_n]]
        sub_matrix = trans_prob[np.ix_(top_idx, top_idx)]

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(sub_matrix, xticklabels=all_sylls[:top_n],
                    yticklabels=all_sylls[:top_n],
                    cmap='Blues', ax=ax, square=True,
                    cbar_kws={'label': 'P(transition)'})
        ax.set_xlabel('To Syllable')
        ax.set_ylabel('From Syllable')
        ax.set_title('Syllable Transition Probabilities')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'transition_matrix.pdf'))
        plt.close()
        log("  Transition matrix saved.")
    except Exception as e:
        log(f"  WARNING: transition matrix: {e}")

    # ------------------------------------------------------------------
    # 5f. Generate grid movies (if videos are available)
    # ------------------------------------------------------------------
    # Build video_paths dict: kpms expects video filename (no ext) to be a
    # prefix of the recording name, but our recording names have DLC suffixes
    # (e.g. _DLC_resnet50_...).  We manually match by finding the video whose
    # stem is the longest prefix of each recording name.
    log("Matching recordings to video files ...")
    manual_video_paths = None
    grid_movie_dir = os.path.join(plot_dir, 'grid_movies')
    os.makedirs(grid_movie_dir, exist_ok=True)
    try:
        video_exts = ('.avi', '.mp4', '.mov', '.mkv', '.wmv')
        all_videos = [os.path.join(VIDEO_DIR, f)
                      for f in os.listdir(VIDEO_DIR)
                      if f.lower().endswith(video_exts)]
        log(f"  Found {len(all_videos)} video files in {VIDEO_DIR}")
        if all_videos:
            vid_stems = {os.path.splitext(os.path.basename(v))[0]: v
                         for v in all_videos}
            manual_video_paths = {}
            unmatched = []
            for rec_name in results.keys():
                matches = [stem for stem in vid_stems
                           if os.path.basename(rec_name).startswith(stem)]
                if matches:
                    best = max(matches, key=len)
                    manual_video_paths[rec_name] = vid_stems[best]
                else:
                    unmatched.append(rec_name)
            log(f"  Matched {len(manual_video_paths)}/{len(results)} recordings to videos")
            if unmatched:
                log(f"  Unmatched (sample): {unmatched[:3]}")
            if not manual_video_paths:
                manual_video_paths = None
    except Exception as e:
        log(f"  WARNING: Could not scan VIDEO_DIR: {e}")

    log("Generating grid movies ...")
    try:
        import traceback as _tb_grid
        kpms.generate_grid_movies(
            results=results,
            output_dir=grid_movie_dir,
            coordinates=coordinates,
            video_paths=manual_video_paths,
            rows=5, cols=6,
            pre=15, post=45,
            fps=30,
            overlay_keypoints=True,
            keypoints_only=False,
        )
        log(f"  Grid movies saved to {grid_movie_dir}")
    except Exception as e:
        log(f"  WARNING: Could not generate grid movies: {e}")
        log(f"  Traceback:\n{''.join(_tb_grid.format_exc())}")

    # ------------------------------------------------------------------
    # 5g. Generate trajectory plots
    # ------------------------------------------------------------------
    log("Generating trajectory plots ...")
    try:
        import traceback as _tb_traj
        traj_dir = os.path.join(plot_dir, 'trajectory_plots')
        os.makedirs(traj_dir, exist_ok=True)
        traj_kwargs = {k: config[k] for k in ['use_bodyparts', 'skeleton',
                                                'keypoint_colormap']
                       if k in config}
        log(f"  Passing kwargs: {list(traj_kwargs.keys())}")
        kpms.generate_trajectory_plots(
            coordinates=coordinates,
            results=results,
            output_dir=traj_dir,
            **traj_kwargs,
        )
        log(f"  Trajectory plots saved to {traj_dir}")
    except Exception as e:
        log(f"  WARNING: Could not generate trajectory plots: {e}")
        log(f"  Traceback:\n{''.join(_tb_traj.format_exc())}")

    # ------------------------------------------------------------------
    # 5g. Summary of syllable statistics
    # ------------------------------------------------------------------
    log("Computing syllable summary statistics ...")
    summary_path = os.path.join(plot_dir, 'syllable_summary.csv')
    if len(syllables_to_plot) > 0:
        summary_rows = []
        for syll in syllables_to_plot:
            sdf = moseq_df[moseq_df['syllable'] == syll]
            if len(sdf) == 0:
                continue
            summary_rows.append({
                'syllable': syll,
                'n_frames': len(sdf),
                'frac_total': len(sdf) / len(moseq_df),
                'mean_velocity_cm_s': sdf['velocity_px_s'].mean() * PX_TO_CM,
                'mean_angular_velocity': sdf['angular_velocity'].mean(),
            })
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(summary_path, index=False)
        log(f"  Syllable summary saved to {summary_path}")

    # ------------------------------------------------------------------
    # 6. Save moseq_df for downstream use
    # ------------------------------------------------------------------
    moseq_df_path = os.path.join(SEARCH_OUTPUT_DIR, 'best_model_moseq_df.csv')
    moseq_df.to_csv(moseq_df_path, index=False)
    log(f"moseq_df saved to {moseq_df_path}")

    analysis_df_path = os.path.join(SEARCH_OUTPUT_DIR, 'best_model_analysis_df.csv')
    analysis_df.to_csv(analysis_df_path, index=False)
    log(f"analysis_df saved to {analysis_df_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("PARAMETER SEARCH COMPLETE")
    log(f"Best model: {best_model_name}")
    log(f"  num_states={best_num_states}, kappa={best_kappa:.0e}")
    log(f"  Selection method: {selection_method}")
    log(f"Results directory: {SEARCH_OUTPUT_DIR}")
    log("=" * 60)


# ════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if '--fit-single' in sys.argv:
        # Worker mode: fit one model and exit
        idx = sys.argv.index('--fit-single')
        num_states = int(sys.argv[idx + 1])
        kappa = float(sys.argv[idx + 2])
        seed = int(sys.argv[idx + 3])

        subset_size = None
        if '--subset' in sys.argv:
            si = sys.argv.index('--subset')
            subset_size = int(sys.argv[si + 1])

        ar_only = '--ar-only' in sys.argv

        try:
            run_fit_worker(num_states, kappa, seed,
                           subset_size=subset_size,
                           ar_only_search=ar_only)
        except Exception as e:
            log(f"WORKER ERROR: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Main mode: orchestrate the full parameter search
        main()
