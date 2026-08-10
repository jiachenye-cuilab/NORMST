"""Optional, slice-wise diagnostics for standard Visium reconstruction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F

from training.engine import move_batch, visium_prediction


DIAGNOSTIC_CHOICES = (
    "pca_reconstruction",
    "residual_pca",
    "latent_rank",
    "loss_contribution",
)


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return value or "slice"


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"cannot write empty diagnostic table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def _as_matrix(values, name: str):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"{name} must have shape [observations, features]")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def _fit_spectrum(values, max_components: int):
    matrix = _as_matrix(values, "PCA input")
    if max_components < 1:
        raise ValueError("max_components must be positive")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    count = min(max_components, len(singular))
    singular = singular[:count]
    components = components[:count]
    squared = singular ** 2
    total = float(np.square(centered).sum())
    ratio = squared / total if total > 0 else np.zeros_like(squared)
    return matrix.mean(axis=0), components, singular, ratio


def _correlation(left, right, epsilon=1e-12):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.shape != left.shape:
        return np.nan
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.square(left).sum() * np.square(right).sum())
    if denominator <= epsilon:
        return np.nan
    return float(np.dot(left, right) / denominator)


def analyze_pca_reconstruction(
    x_true,
    x_idw,
    x_pred,
    max_components: int = 50,
):
    """Fit PCA only on truth, then project IDW and NORMST on that basis."""
    truth = _as_matrix(x_true, "x_true")
    idw = _as_matrix(x_idw, "x_idw")
    prediction = _as_matrix(x_pred, "x_pred")
    if idw.shape != truth.shape or prediction.shape != truth.shape:
        raise ValueError("truth, IDW, and prediction matrices must align")
    mean, components, _, ratio = _fit_spectrum(truth, max_components)
    true_score = (truth - mean) @ components.T
    idw_score = (idw - mean) @ components.T
    prediction_score = (prediction - mean) @ components.T
    cumulative = np.cumsum(ratio)
    return [
        {
            "pc": index + 1,
            "explained_variance_ratio": float(ratio[index]),
            "cumulative_explained_variance": float(cumulative[index]),
            "idw_corr": _correlation(true_score[:, index], idw_score[:, index]),
            "normst_corr": _correlation(
                true_score[:, index], prediction_score[:, index]
            ),
        }
        for index in range(len(ratio))
    ]


def _spectrum_rows(values, max_components: int):
    _, _, singular, ratio = _fit_spectrum(values, max_components)
    cumulative = np.cumsum(ratio)
    rows = [
        {
            "pc": index + 1,
            "explained_variance_ratio": float(ratio[index]),
            "cumulative_explained_variance": float(cumulative[index]),
        }
        for index in range(len(ratio))
    ]
    total = singular.sum()
    if total > 0:
        probability = singular / total
        effective_rank = float(
            np.exp(-(probability * np.log(probability + 1e-12)).sum())
        )
    else:
        effective_rank = 0.0
    return rows, effective_rank


def _spectrum_summary(rows, thresholds, effective_rank=None):
    ratio = np.asarray(
        [row["explained_variance_ratio"] for row in rows], dtype=np.float64
    )
    cumulative = np.cumsum(ratio)
    summary = {
        "available_components": len(rows),
        "pc1_explained_variance": float(ratio[0]) if len(ratio) else 0.0,
    }
    for threshold in thresholds:
        actual = min(threshold, len(cumulative))
        summary[f"pc{threshold}_actual_component"] = actual
        summary[f"pc{threshold}_cumulative_explained_variance"] = (
            float(cumulative[actual - 1]) if actual else 0.0
        )
    if effective_rank is not None:
        summary["effective_rank"] = effective_rank
    return summary


def analyze_residual_pca(x_true, x_idw, max_components: int = 256):
    truth = _as_matrix(x_true, "x_true")
    idw = _as_matrix(x_idw, "x_idw")
    if idw.shape != truth.shape:
        raise ValueError("truth and IDW matrices must align")
    rows, _ = _spectrum_rows(truth - idw, max_components)
    return rows, _spectrum_summary(rows, (10, 32, 64, 128, 256))


def analyze_latent_effective_rank(h0, hl, max_components: int = 128):
    """Use centered singular values and p_i=s_i/sum(s_i) effective rank."""
    result = {}
    combined_rows = []
    for name, values in (("h0", h0), ("hl", hl)):
        rows, effective_rank = _spectrum_rows(values, max_components)
        combined_rows.extend({"representation": name, **row} for row in rows)
        result[name] = _spectrum_summary(
            rows, (10, 32, 64, 128), effective_rank=effective_rank
        )
    result["definition"] = (
        "column-centered SVD; effective_rank=exp(-sum(p*log(p))), "
        "p=singular_value/sum(singular_values)"
    )
    return combined_rows, result


def analyze_gene_loss_contribution(x_true, x_pred, genes):
    truth = _as_matrix(x_true, "x_true")
    prediction = _as_matrix(x_pred, "x_pred")
    genes = np.asarray(genes).astype(str)
    if prediction.shape != truth.shape or len(genes) != truth.shape[1]:
        raise ValueError("truth, prediction, and genes must align")
    elementwise = F.smooth_l1_loss(
        torch.from_numpy(prediction),
        torch.from_numpy(truth),
        reduction="none",
    ).numpy()
    mean_loss = elementwise.mean(axis=0)
    loss_sum = elementwise.sum(axis=0)
    total = loss_sum.sum()
    fraction = loss_sum / total if total > 0 else np.zeros_like(loss_sum)
    rows = [
        {
            "gene": genes[index],
            "variance": float(np.var(truth[:, index])),
            "mean_expression": float(np.mean(truth[:, index])),
            "loss": float(mean_loss[index]),
            "loss_fraction": float(fraction[index]),
        }
        for index in np.argsort(-mean_loss)
    ]
    summary = {}
    for percentage in (1, 5, 10, 20):
        count = max(1, int(np.ceil(len(rows) * percentage / 100.0)))
        summary[f"top_{percentage}_percent_gene_count"] = count
        summary[f"top_{percentage}_percent_loss_fraction"] = float(
            sum(row["loss_fraction"] for row in rows[:count])
        )
    return rows, summary


def _mean_by_index(values, indices):
    matrix = _as_matrix(values, "indexed values")
    indices = np.asarray(indices).reshape(-1)
    if len(indices) != len(matrix):
        raise ValueError("indices and values must have the same length")
    unique, inverse = np.unique(indices, return_inverse=True)
    result = np.zeros((len(unique), matrix.shape[1]), dtype=np.float64)
    np.add.at(result, inverse, matrix)
    count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    return unique, result / count[:, None]


def _query_views(values):
    mask = np.asarray(values["mask"]).reshape(-1).astype(bool)
    target_spots = np.asarray(values["target_spots"]).reshape(-1)[mask]
    raw = {}
    collapsed = {}
    for key in ("truth", "baseline", "prediction"):
        matrix = np.asarray(values[key]).reshape(
            -1, np.asarray(values[key]).shape[-1]
        )[mask]
        raw[key] = matrix
        unique, collapsed[key] = _mean_by_index(matrix, target_spots)
    return raw, collapsed, unique


def _latent_views(values):
    h0_index, h0 = _mean_by_index(values["h0"], values["visible_spots"])
    hl_index, hl = _mean_by_index(values["hl"], values["visible_spots"])
    if not np.array_equal(h0_index, hl_index):
        raise ValueError("H0 and HL visible spot indices do not align")
    return h0, hl, h0_index


def _aggregate_pc_rows(records, columns):
    grouped = {}
    for record in records:
        grouped.setdefault(int(record["pc"]), []).append(record)
    output = []
    for pc in sorted(grouped):
        group = grouped[pc]
        row = {"pc": pc, "slice_count": len(group)}
        for column in columns:
            values = np.asarray([item[column] for item in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[column] = float(finite.mean()) if len(finite) else np.nan
        output.append(row)
    return output


@torch.no_grad()
def collect_visium_diagnostic_arrays(
    model,
    loader,
    device,
    use_amp,
    full_neighbors,
    full_xy,
):
    """Collect exact model IDW, prediction, query ids, and H0/HL latents."""
    model.eval()
    collected = {
        "prediction": [], "truth": [], "baseline": [], "mask": [],
        "target_spots": [], "h0": [], "hl": [], "visible_spots": [],
    }
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction, target, mask, baseline, auxiliary = visium_prediction(
                model,
                batch,
                full_neighbors,
                full_xy,
                return_latents=True,
            )
        for key, value in (
            ("prediction", prediction),
            ("truth", target),
            ("baseline", baseline),
            ("mask", mask),
            ("target_spots", batch["target_spots"]),
        ):
            collected[key].append(value.detach().float().cpu().numpy())
        visible_mask = auxiliary["visible_mask"].detach().cpu().numpy().astype(bool)
        for key in ("h0", "hl"):
            values = auxiliary[key].detach().float().cpu().numpy()
            collected[key].append(values[visible_mask])
        visible_spots = batch["visible_spots"].detach().cpu().numpy()
        collected["visible_spots"].append(visible_spots[visible_mask])
    if not collected["prediction"]:
        raise ValueError("cannot diagnose an empty loader")
    return {
        key: np.concatenate(value, axis=0)
        for key, value in collected.items()
    }


def write_visium_diagnostics(
    output_dir,
    split: str,
    per_slice: dict[str, dict[str, np.ndarray]],
    genes,
    requested,
    pca_components: int = 50,
):
    """Write per-slice diagnostics and macro summaries under one split."""
    requested = tuple(dict.fromkeys(requested))
    unknown = set(requested) - set(DIAGNOSTIC_CHOICES)
    if unknown:
        raise ValueError(f"unknown diagnostics: {sorted(unknown)}")
    root = Path(output_dir) / "diagnostics" / split
    root.mkdir(parents=True, exist_ok=True)
    pca_records = []
    residual_records = []
    latent_records = []
    loss_records = []
    manifest = {
        "split": split,
        "requested": list(requested),
        "pca_components": pca_components,
        "definitions": {
            "pca_inputs": (
                "repeated target occurrences are averaged by native spot id "
                "within each slice before truth-only PCA"
            ),
            "latent_inputs": (
                "H0 and HL occurrences are averaged by visible native spot id "
                "within each slice before centered SVD"
            ),
            "loss_inputs": (
                "gene SmoothL1 contribution uses every evaluated query occurrence"
            ),
        },
        "slices": {},
    }

    for name, values in per_slice.items():
        slice_dir = root / _safe_name(name)
        slice_dir.mkdir(parents=True, exist_ok=True)
        raw, collapsed, query_index = _query_views(values)
        manifest["slices"][name] = {
            "query_occurrences": len(raw["truth"]),
            "unique_query_spots": len(query_index),
        }
        if "pca_reconstruction" in requested:
            rows = analyze_pca_reconstruction(
                collapsed["truth"],
                collapsed["baseline"],
                collapsed["prediction"],
                max_components=max(50, pca_components),
            )
            _write_csv(slice_dir / "pca_reconstruction.csv", rows)
            pca_records.extend({"slice": name, **row} for row in rows)
        if "residual_pca" in requested:
            rows, summary = analyze_residual_pca(
                collapsed["truth"], collapsed["baseline"]
            )
            _write_csv(slice_dir / "residual_pca.csv", rows)
            _write_json(slice_dir / "residual_pca_summary.json", summary)
            residual_records.extend({"slice": name, **row} for row in rows)
        if "latent_rank" in requested:
            h0, hl, visible_index = _latent_views(values)
            rows, summary = analyze_latent_effective_rank(h0, hl)
            summary["unique_visible_spots"] = len(visible_index)
            _write_csv(slice_dir / "latent_rank.csv", rows)
            _write_json(slice_dir / "latent_rank_summary.json", summary)
            latent_records.extend({"slice": name, **row} for row in rows)
        if "loss_contribution" in requested:
            rows, summary = analyze_gene_loss_contribution(
                raw["truth"], raw["prediction"], genes
            )
            _write_csv(slice_dir / "gene_loss_contribution.csv", rows)
            _write_json(slice_dir / "gene_loss_contribution_summary.json", summary)
            loss_records.extend({"slice": name, **row} for row in rows)

    if pca_records:
        _write_csv(
            root / "pca_reconstruction_summary.csv",
            _aggregate_pc_rows(
                pca_records,
                (
                    "explained_variance_ratio",
                    "cumulative_explained_variance",
                    "idw_corr",
                    "normst_corr",
                ),
            ),
        )
    if residual_records:
        _write_csv(
            root / "residual_pca_summary.csv",
            _aggregate_pc_rows(
                residual_records,
                ("explained_variance_ratio", "cumulative_explained_variance"),
            ),
        )
    if latent_records:
        for representation in ("h0", "hl"):
            selected = [
                row for row in latent_records
                if row["representation"] == representation
            ]
            _write_csv(
                root / f"latent_rank_{representation}_summary.csv",
                _aggregate_pc_rows(
                    selected,
                    (
                        "explained_variance_ratio",
                        "cumulative_explained_variance",
                    ),
                ),
            )
    if loss_records:
        by_gene = {}
        for row in loss_records:
            by_gene.setdefault(row["gene"], []).append(row)
        rows = []
        for gene, group in by_gene.items():
            rows.append({
                "gene": gene,
                "variance": float(np.mean([item["variance"] for item in group])),
                "mean_expression": float(np.mean([
                    item["mean_expression"] for item in group
                ])),
                "loss": float(np.mean([item["loss"] for item in group])),
                "loss_fraction": 0.0,
            })
        total = sum(row["loss"] for row in rows)
        for row in rows:
            row["loss_fraction"] = row["loss"] / total if total > 0 else 0.0
        rows.sort(key=lambda row: row["loss"], reverse=True)
        _write_csv(root / "gene_loss_contribution_summary.csv", rows)
        summary = {}
        for percentage in (1, 5, 10, 20):
            count = max(1, int(np.ceil(len(rows) * percentage / 100.0)))
            summary[f"top_{percentage}_percent_gene_count"] = count
            summary[f"top_{percentage}_percent_loss_fraction"] = float(
                sum(row["loss_fraction"] for row in rows[:count])
            )
        _write_json(root / "gene_loss_contribution_summary.json", summary)

    _write_json(root / "diagnostics_manifest.json", manifest)
    return manifest
