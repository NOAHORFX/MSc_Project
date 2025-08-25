# position_bias_batch.py
from __future__ import annotations

from typing import Dict, Iterable, List
import numpy as np
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


# Register custom loss if your model needs it.
try:
    from losses import mse_with_similarity_and_confidence_dynamic
    _CUSTOM_OBJECTS = {"mse_with_sim_and_conf_dynamic": mse_with_similarity_and_confidence_dynamic}
except Exception:
    _CUSTOM_OBJECTS = {}


# ----------------------------- Utils -----------------------------

def _safe_savgol(y: np.ndarray, n: int) -> np.ndarray:
    if n < 13:
        return y
    win = min(11, max(5, (n // 2) * 2 - 1))
    return savgol_filter(y, window_length=win, polyorder=3)


def _empty_metrics() -> Dict[str, float]:
    return {"mean_abs_err_frames": 0.0,"median_abs_err_frames": 0.0,"p95_abs_err_frames": 0.0,"bias_frames": 0.0,"recall": 0.0,"precision": 0.0,"n_gt": 0,"n_pred": 0,"matched": 0,}


# ------------------------- Single sequence ------------------------

def position_bias_for_sequence(labels: np.ndarray,y_pred: np.ndarray,*,exclude_margin: int = 10,selection: str = "topk",score_threshold: float = 0.7,tolerance: int = 2,smooth_method: str = "gaussian") -> Dict[str, float]:

    labels = np.asarray(labels).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = labels.size
    if n <= 1:
        return _empty_metrics()

    # Smooth scores
    if smooth_method == "gaussian":
        y_pred = gaussian_filter1d(y_pred, sigma=2)
    elif smooth_method == "savgol":
        y_pred = _safe_savgol(y_pred, n)

    # Valid window to avoid boundary artifacts
    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[exclude_margin:n - exclude_margin] = True

    # Ground truth positions (≥ 0.5 considered keyframe)
    gt_idx = np.where((labels >= 0.5) & valid_mask)[0]
    if gt_idx.size == 0:
        return _empty_metrics()

    # Prediction positions
    if selection == "topk":
        k = int(gt_idx.size)
        scores = np.where(valid_mask, y_pred, -np.inf)
        if k >= scores.size:
            pred_idx = np.where(valid_mask)[0]
        else:
            topk_part = np.argpartition(-scores, k - 1)[:k]
            pred_idx = topk_part[np.argsort(-scores[topk_part])]
    elif selection == "threshold":
        pred_idx = np.where((y_pred >= score_threshold) & valid_mask)[0]
    else:
        raise ValueError("selection must be 'topk' or 'threshold'.")

    if pred_idx.size == 0:
        out = _empty_metrics()
        out["n_gt"] = int(gt_idx.size)
        return out

    # Vectorized nearest-GT computation
    diff_mat = pred_idx[:, None] - gt_idx[None, :]
    nearest = np.argmin(np.abs(diff_mat), axis=1)
    diffs = diff_mat[np.arange(pred_idx.size), nearest]  # signed bias (pred - gt)

    hit_mask = np.abs(diffs) <= tolerance
    abs_errs = np.where(hit_mask, 0, np.abs(diffs))
    matched = int(hit_mask.sum())

    return {
        "mean_abs_err_frames": float(np.mean(abs_errs)),
        "median_abs_err_frames": float(np.median(abs_errs)),
        "p95_abs_err_frames": float(np.percentile(abs_errs, 95)),
        "bias_frames": float(np.mean(diffs)),         # >0 late; <0 early
        "recall": matched / float(gt_idx.size),
        "precision": matched / float(pred_idx.size),
        "n_gt": int(gt_idx.size),
        "n_pred": int(pred_idx.size),
        "matched": matched,
    }


# ---------------------------- Batch eval --------------------------

def position_bias_batch(model_path: str,val_feature_files: Iterable[str],*,exclude_margin: int = 10,selection: str = "topk",score_threshold: float = 0.7,tolerance: int = 2,smooth_method: str = "gaussian") -> Dict[str, float]:

    model = load_model(model_path, custom_objects=_CUSTOM_OBJECTS)
    results: List[Dict[str, float]] = []
    n_files = 0

    for npy_path in val_feature_files:
        data = np.load(npy_path, allow_pickle=True).item()
        labels = np.asarray(data["labels"])
        feats = np.asarray(data["features"])

        if np.max(labels) < 0.5:   # skip sequences without any GT keyframe
            continue

        n_files += 1
        y_pred = model.predict(np.expand_dims(feats, 0), verbose=0)[0]
        metrics = position_bias_for_sequence(labels, y_pred, exclude_margin=exclude_margin, selection=selection, 
                                             score_threshold=score_threshold, tolerance=tolerance, smooth_method=smooth_method)
        results.append(metrics)

    if n_files == 0:
        print("[WARNING] No samples with keyframes found.")
        return {}

    def mean_of(key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    summary = {
        "n_files": n_files,
        "mean_abs_err_frames": mean_of("mean_abs_err_frames"),
        "median_abs_err_frames": mean_of("median_abs_err_frames"),
        "p95_abs_err_frames": mean_of("p95_abs_err_frames"),
        "bias_frames": mean_of("bias_frames"),
        "recall": mean_of("recall"),
        "precision": mean_of("precision"),
    }

    print("\n=== Keyframe Position Bias Evaluation (Batch) ===")
    print(f"Files evaluated     : {summary['n_files']}")
    print(f"Mean abs error      : {summary['mean_abs_err_frames']:.2f} frames")
    print(f"Median abs error    : {summary['median_abs_err_frames']:.2f} frames")
    print(f"95% abs error ≤     : {summary['p95_abs_err_frames']:.2f} frames")
    print(f"Mean bias direction : {summary['bias_frames']:.2f} frames (pos=late, neg=early)")
    print(f"Recall              : {summary['recall']:.2%}")
    print(f"Precision           : {summary['precision']:.2%}")

    return summary
