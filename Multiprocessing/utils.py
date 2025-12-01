import numpy as np
import pandas as pd
import os
import time
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import hdbscan
import optuna
import umap
import matplotlib.pyplot as plt

# Set up basic logging for functions in this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def trustworthiness_low_memory(X, X_embedding, k=15):
    n_samples = X.shape[0]
    t_score = 1
    t_punishment = 0
    punishment_constant = (2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0)))
    logging.info("Finding NN in embedding space for trustworthiness...")
    ind_X_embedded = (
        NearestNeighbors(n_neighbors=k)
        .fit(X_embedding)
        .kneighbors(return_distance=False)
    )
    for i in range(n_samples):
        dist_X = np.sqrt(np.sum((X - X[i])**2, axis=1))
        dist_X[i] = np.inf
        ind_X = np.argsort(dist_X)
        ranks = np.where(np.isin(ind_X, ind_X_embedded[i]))[0]
        t_punishment += np.sum(np.maximum(0, ranks - k))
    t_score -= t_punishment * punishment_constant
    logging.info(f'Trustworthiness T(k={k}) = {t_score:.6f}')
    return t_score


# ---------------------------
# UMAP runner
# ---------------------------
def umap_testing(n_neighbors, min_dist, n_epochs, verbose, set_op_mix_ratio, local_connectivity, X, i, out_dir='umap_outputs'):
    """
    Runs UMAP and saves embedding to CSV. Also saves a small PNG scatter and params JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    hyperparameters = {
        'n_neighbors': int(n_neighbors),
        'min_dist': float(min_dist),
        'n_epochs': int(n_epochs),
        'verbose': bool(verbose),
        'set_op_mix_ratio': float(set_op_mix_ratio),
        'local_connectivity': float(local_connectivity)
    }

    logging.info(f"[UMAP {i}] running with params: {hyperparameters}")
    model = umap.UMAP(**hyperparameters)
    embedding = model.fit_transform(X)

    # Save CSV
    csv_path = os.path.join(out_dir, f'umap_embeddings_{i}.csv')
    df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    df.to_csv(csv_path, index=False)

    # Save small scatter PNG (no show)
    plt.figure(figsize=(5,5))
    plt.scatter(embedding[:,0], embedding[:,1], s=1)
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
    plt.title(f"UMAP {i}")
    png_path = os.path.join(out_dir, f'umap_scatter_{i}.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=100)
    plt.close()

    # Save params sidecar
    import json
    meta = {
        'index': i,
        'csv': csv_path,
        'png': png_path,
        'params': hyperparameters,
        'timestamp': time.time()
    }
    with open(os.path.join(out_dir, f'umap_meta_{i}.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logging.info(f"[UMAP {i}] saved {csv_path} and {png_path}")
    return csv_path, png_path, meta


def wrapper(idx, X, n_neighbors, min_dist, n_epochs, verbose, set_op_mix_ratio, local_connectivity, out_dir='umap_outputs'):
    """
    Safe wrapper to pin a process to a single core and run UMAP. idx -> mapped to a valid core.
    Call this as a target of multiprocessing.Process.
    """
    # Map run index to CPU core ID
    cpu_count = os.cpu_count() or 1
    core_id = int(idx) % cpu_count

    # Attempt to set affinity (Linux)
    try:
        os.sched_setaffinity(0, {core_id})
        logging.info(f"[Wrapper {idx}] pinned to core {core_id}")
    except AttributeError:
        logging.warning("[Wrapper] os.sched_setaffinity not available on this platform; skipping pin.")
    except OSError as e:
        logging.warning(f"[Wrapper {idx}] could not set affinity to core {core_id}: {e}")

    # Run UMAP and save results
    try:
        umap_testing(n_neighbors, min_dist, n_epochs, verbose, set_op_mix_ratio, local_connectivity, X, idx, out_dir=out_dir)
    except Exception as e:
        logging.exception(f"[Wrapper {idx}] UMAP failed: {e}")
        raise


# ---------------------------
# Optuna / HDBSCAN
# ---------------------------
def objective_for_embedding(trial, embedding):
    """
    Objective expects the embedding passed directly.
    """
    # sample hyperparams
    min_cluster_size = trial.suggest_int('min_cluster_size', 5, 200)
    min_samples = trial.suggest_int('min_samples', 1, 100)

    hyperparams = {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'cluster_selection_method': 'eom',
        'cluster_selection_epsilon': 0.0,
        'allow_single_cluster': False
    }

    try:
        clusterer = hdbscan.HDBSCAN(**hyperparams)
        clusterer.fit(embedding)
        # if no clusters (all noise), penalize
        labels = np.array(clusterer.labels_)
        n_unique = len(np.unique(labels))
        if n_unique <= 1:
            return -1.0
        # use mean cluster persistence as score
        persistence = np.mean(clusterer.cluster_persistence_)
        if not np.isfinite(persistence):
            return -1.0
        return float(persistence)
    except Exception as e:
        logging.exception(f"[Optuna objective] failure: {e}")
        return -1.0


def opt(embedding, trials=30, study_name=None, output_dir='hdbscan_outputs'):
    """
    Run Optuna to tune HDBSCAN hyperparams on a provided embedding. Saves best params and a plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.RandomSampler()
    study_name = study_name or f"HDBSCAN_study_{int(time.time())}"
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)

    # objective wrapper
    study.optimize(lambda t: objective_for_embedding(t, embedding), n_trials=trials, n_jobs=1, show_progress_bar=False)

    best_params = study.best_params
    best_val = study.best_value
    logging.info(f"[Optuna] Best score {best_val:.4f} with params: {best_params}")

    # Fit final clustering with best params
    clusterer = hdbscan.HDBSCAN(**best_params)
    res = clusterer.fit(embedding)
    labels = res.labels_
    persistence_mean = float(np.mean(res.cluster_persistence_)) if len(res.cluster_persistence_)>0 else None

    # Save results
    import json
    meta = {
        'best_score': best_val,
        'best_params': best_params,
        'persistence_mean': persistence_mean,
        'timestamp': time.time()
    }
    with open(os.path.join(output_dir, 'hdbscan_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save scatter with clustering
    plt.figure(figsize=(6,6))
    plt.title(f'Persistence: {persistence_mean}')
    plt.scatter(embedding[:,0], embedding[:,1], s=1, c=labels)
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
    out_png = os.path.join(output_dir, f'hdbscan_result_{int(time.time())}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    logging.info(f"[HDBSCAN] Saved results to {output_dir}")
    return {
        'labels': labels,
        'persistence_mean': persistence_mean,
        'best_params': best_params,
        'plot': out_png
    }
