#!/usr/bin/env python3
"""One-SR boosted VHbb parquet smoke test and weighted DNN training.

This script is intentionally self-contained for the first lxplus/local-agent test.
It does four things:

1. Find a small set/fraction of parquet files for one boosted SR.
2. Inspect available parquet branches/columns and select safe numeric features.
3. Make signal-vs-background feature plots before training.
4. Preprocess features and train a weighted plain DNN on GPU if available.

ChatGPT should write/review this file but should not execute lxplus/EOS/GPU work.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - lxplus env issue
    raise RuntimeError("PyYAML is required. Try: python3 -m pip install --user pyyaml") from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - lxplus env issue
    raise RuntimeError("pyarrow is required. Try: python3 -m pip install --user pyarrow") from exc

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - lxplus env issue
    raise RuntimeError("PyTorch is required. Load a CUDA/PyTorch LCG view or install torch.") from exc


@dataclass
class FileRecord:
    path: str
    sample_name: str
    region_name: str
    process_group: str
    label: int
    n_rows: int
    columns: List[str]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=str)


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def classify_sample(sample_name: str, sample_cfg: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """Return (process_group, label), or None if sample is not selected."""
    for group, cfg in sample_cfg.items():
        include_patterns = cfg.get("include_patterns", []) or []
        exclude_patterns = cfg.get("exclude_patterns", []) or []
        optional = bool(cfg.get("optional", False))
        label = cfg.get("label", None)

        if label is None:
            continue
        if include_patterns and not matches_any(sample_name, include_patterns):
            continue
        if exclude_patterns and matches_any(sample_name, exclude_patterns):
            continue
        if optional and not include_patterns:
            continue
        return group, int(label)
    return None


def arrow_type_is_numeric(dtype: pa.DataType) -> bool:
    return (
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_boolean(dtype)
    )


def parquet_schema(path: Path) -> Tuple[List[str], Dict[str, str], int]:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    columns = list(schema.names)
    type_map = {field.name: str(field.type) for field in schema}
    n_rows = int(pf.metadata.num_rows) if pf.metadata is not None else -1
    return columns, type_map, n_rows


def find_parquet_files(columns_root: Path, region: str, sample_cfg: Dict[str, Any]) -> List[FileRecord]:
    if not columns_root.exists():
        raise FileNotFoundError(f"columns_root does not exist: {columns_root}")

    records: List[FileRecord] = []
    for sample_dir in sorted(p for p in columns_root.iterdir() if p.is_dir()):
        sample_name = sample_dir.name
        cls = classify_sample(sample_name, sample_cfg)
        if cls is None:
            continue
        process_group, label = cls

        region_dirs = [p for p in sample_dir.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, region)]
        for region_dir in region_dirs:
            actual_region = region_dir.name
            files = sorted(region_dir.rglob("*.parquet"))
            for path in files:
                try:
                    columns, _, n_rows = parquet_schema(path)
                except Exception as exc:
                    print(f"[WARN] Failed to read parquet schema: {path}: {exc}")
                    continue
                records.append(
                    FileRecord(
                        path=str(path),
                        sample_name=sample_name,
                        region_name=actual_region,
                        process_group=process_group,
                        label=label,
                        n_rows=n_rows,
                        columns=columns,
                    )
                )
    return records


def limit_files_per_group(records: List[FileRecord], max_files_per_group: Optional[int]) -> List[FileRecord]:
    if max_files_per_group is None:
        return records
    out: List[FileRecord] = []
    counts: Dict[str, int] = {}
    for rec in records:
        counts.setdefault(rec.process_group, 0)
        if counts[rec.process_group] >= max_files_per_group:
            continue
        out.append(rec)
        counts[rec.process_group] += 1
    return out


def collect_column_report(records: Sequence[FileRecord]) -> Dict[str, Any]:
    column_counts: Dict[str, int] = {}
    column_types: Dict[str, List[str]] = {}
    per_file: List[Dict[str, Any]] = []

    for rec in records:
        path = Path(rec.path)
        columns, type_map, n_rows = parquet_schema(path)
        for col in columns:
            column_counts[col] = column_counts.get(col, 0) + 1
            column_types.setdefault(col, [])
            if type_map[col] not in column_types[col]:
                column_types[col].append(type_map[col])
        per_file.append(
            {
                "path": rec.path,
                "sample_name": rec.sample_name,
                "process_group": rec.process_group,
                "label": rec.label,
                "n_rows": n_rows,
                "n_columns": len(columns),
                "columns": columns,
                "types": type_map,
            }
        )

    return {
        "n_files": len(records),
        "total_rows_nominal_before_sampling": int(sum(max(r.n_rows, 0) for r in records)),
        "column_counts": column_counts,
        "column_types": column_types,
        "per_file": per_file,
    }


def pattern_filter(names: Iterable[str], patterns: Sequence[str]) -> List[str]:
    return [name for name in names if matches_any(name, patterns)]


def select_weight_column(all_columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_to_name = {c.lower(): c for c in all_columns}
    for cand in candidates:
        if cand.lower() in lower_to_name:
            return lower_to_name[cand.lower()]
    # fallback: prefer total nominal-looking weight before generic genWeight-like names
    weight_like = [c for c in all_columns if "weight" in c.lower()]
    preferred_tokens = ["nominal", "event", "total", "lumi", "xsec"]
    for token in preferred_tokens:
        for c in weight_like:
            if token in c.lower():
                return c
    return weight_like[0] if weight_like else None


def select_event_column(all_columns: Sequence[str]) -> Optional[str]:
    exact = ["event", "Event", "eventNumber", "event_number", "evt", "Evt"]
    lower_to_name = {c.lower(): c for c in all_columns}
    for cand in exact:
        if cand.lower() in lower_to_name:
            return lower_to_name[cand.lower()]
    for col in all_columns:
        low = col.lower()
        if "event" in low and "weight" not in low:
            return col
    return None


def select_numeric_features(
    column_report: Dict[str, Any], feature_cfg: Dict[str, Any]
) -> Tuple[List[str], Optional[str], Optional[str]]:
    all_columns = list(column_report["column_counts"].keys())
    types = column_report["column_types"]

    forbidden = feature_cfg.get("forbidden_patterns", []) or []
    preferred = feature_cfg.get("preferred_patterns", []) or []
    weight_col = select_weight_column(all_columns, feature_cfg.get("weight_candidates", []) or [])
    event_col = select_event_column(all_columns)

    safe_numeric: List[str] = []
    for col in all_columns:
        if matches_any(col, forbidden):
            continue
        if col == weight_col or col == event_col:
            continue
        type_strings = types.get(col, [])
        # Accept simple numeric Arrow type strings. This avoids needing the actual pa.DataType here.
        if any(
            token in t.lower()
            for t in type_strings
            for token in ["int", "float", "double", "bool"]
        ):
            safe_numeric.append(col)

    preferred_features = [c for c in safe_numeric if matches_any(c, preferred)]
    remaining = [c for c in safe_numeric if c not in set(preferred_features)]

    # Keep preferred features first, then safe extra features. Extra features are useful for first branch scan.
    features = sorted(preferred_features) + sorted(remaining)
    return features, weight_col, event_col


def stable_hash_to_fold(sample_name: str, region_name: str, row_index: int) -> int:
    payload = f"{sample_name}|{region_name}|{row_index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
    return int(digest, 16) % 2


def read_sampled_parquets(
    records: Sequence[FileRecord],
    feature_columns: Sequence[str],
    weight_col: Optional[str],
    event_col: Optional[str],
    fraction: float,
    seed: int,
    max_events_per_class: Optional[int],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    requested = list(dict.fromkeys(list(feature_columns) + [c for c in [weight_col, event_col] if c]))
    chunks: List[pd.DataFrame] = []

    for file_idx, rec in enumerate(records):
        path = Path(rec.path)
        try:
            file_columns, _, _ = parquet_schema(path)
        except Exception as exc:
            print(f"[WARN] Skip unreadable parquet: {path}: {exc}")
            continue

        available = [c for c in requested if c in file_columns]
        missing_features = [c for c in feature_columns if c not in file_columns]
        if not available:
            print(f"[WARN] Skip file with no requested columns: {path}")
            continue

        pf = pq.ParquetFile(path)
        row_offset = 0
        for batch in pf.iter_batches(batch_size=200_000, columns=available):
            df = batch.to_pandas()
            n = len(df)
            if n == 0:
                continue
            df["__local_row_index"] = np.arange(row_offset, row_offset + n, dtype=np.int64)
            row_offset += n

            if fraction < 1.0:
                mask = rng.random(n) < fraction
                df = df.loc[mask].copy()
            if df.empty:
                continue

            for col in missing_features:
                df[col] = np.nan

            df["__sample_name"] = rec.sample_name
            df["__region_name"] = rec.region_name
            df["__process_group"] = rec.process_group
            df["__label"] = rec.label
            df["__source_path"] = rec.path

            if weight_col and weight_col in df.columns:
                df["__signed_weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
            else:
                df["__signed_weight"] = 1.0

            if event_col and event_col in df.columns:
                event_values = pd.to_numeric(df[event_col], errors="coerce")
                fallback = df["__local_row_index"].astype(np.int64)
                event_int = event_values.fillna(fallback).astype(np.int64)
                df["__fold"] = (event_int % 2).astype(np.int8)
            else:
                df["__fold"] = [
                    stable_hash_to_fold(rec.sample_name, rec.region_name, int(i))
                    for i in df["__local_row_index"].to_numpy()
                ]

            chunks.append(df)
        print(f"[INFO] sampled {file_idx + 1}/{len(records)}: {rec.sample_name}")

    if not chunks:
        raise RuntimeError("No events were read. Check region, columns_root, and sample patterns.")

    data = pd.concat(chunks, ignore_index=True)

    if max_events_per_class is not None:
        limited_chunks = []
        for label, group in data.groupby("__label", sort=True):
            if len(group) > max_events_per_class:
                limited_chunks.append(group.sample(n=max_events_per_class, random_state=seed + int(label)))
            else:
                limited_chunks.append(group)
        data = pd.concat(limited_chunks, ignore_index=True)

    return data


def summarize_dataframe(df: pd.DataFrame, feature_columns: Sequence[str]) -> Dict[str, Any]:
    return {
        "n_events": int(len(df)),
        "n_features_requested": int(len(feature_columns)),
        "labels": df["__label"].value_counts(dropna=False).to_dict(),
        "process_groups": df["__process_group"].value_counts(dropna=False).to_dict(),
        "folds": df["__fold"].value_counts(dropna=False).to_dict(),
        "signed_weight_sum_by_label": df.groupby("__label")["__signed_weight"].sum().to_dict(),
        "abs_weight_sum_by_label": df.assign(__absw=df["__signed_weight"].abs()).groupby("__label")["__absw"].sum().to_dict(),
    }


def sanitize_filename(name: str) -> str:
    keep = []
    for char in name:
        if char.isalnum() or char in ("_", "-", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)[:180]


def plot_feature_svb(
    df: pd.DataFrame,
    feature: str,
    out_dir: Path,
    bins: int = 50,
) -> Optional[str]:
    x = pd.to_numeric(df[feature], errors="coerce")
    y = df["__label"].to_numpy()
    w = df["__signed_weight"].abs().to_numpy(dtype=float)
    finite = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(w) & (w >= 0)
    if finite.sum() < 10:
        return None

    x_arr = x.to_numpy(dtype=float)[finite]
    y_arr = y[finite]
    w_arr = w[finite]

    if len(np.unique(y_arr)) < 2 or np.nanstd(x_arr) == 0:
        return None

    q_low, q_high = np.quantile(x_arr, [0.001, 0.999])
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low == q_high:
        return None
    x_arr = np.clip(x_arr, q_low, q_high)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, name in [(1, "signal"), (0, "background")]:
        mask = y_arr == label
        if mask.sum() == 0:
            continue
        ax.hist(
            x_arr[mask],
            bins=bins,
            weights=w_arr[mask],
            histtype="step",
            density=True,
            label=name,
        )
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("weighted density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"{sanitize_filename(feature)}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def make_feature_plots(df: pd.DataFrame, features: Sequence[str], out_dir: Path, bins: int) -> List[str]:
    safe_mkdir(out_dir)
    outputs: List[str] = []
    for i, feature in enumerate(features, start=1):
        try:
            out = plot_feature_svb(df, feature, out_dir, bins=bins)
            if out:
                outputs.append(out)
        except Exception as exc:
            print(f"[WARN] Failed feature plot for {feature}: {exc}")
        if i % 25 == 0:
            print(f"[INFO] feature plots: {i}/{len(features)}")
    return outputs


@dataclass
class PreprocessStats:
    kept_features: List[str]
    dropped_features: Dict[str, str]
    medians: Dict[str, float]
    clip_low: Dict[str, float]
    clip_high: Dict[str, float]
    means: Dict[str, float]
    stds: Dict[str, float]


def fit_preprocessor(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    clip_quantiles: Tuple[float, float],
    min_finite_fraction: float,
    remove_constant: bool,
) -> PreprocessStats:
    kept: List[str] = []
    dropped: Dict[str, str] = {}
    medians: Dict[str, float] = {}
    lows: Dict[str, float] = {}
    highs: Dict[str, float] = {}
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    for feature in feature_columns:
        values = pd.to_numeric(train_df[feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        finite_fraction = float(finite.mean()) if len(values) else 0.0
        if finite_fraction < min_finite_fraction:
            dropped[feature] = f"finite_fraction {finite_fraction:.3f} < {min_finite_fraction:.3f}"
            continue
        finite_values = values[finite]
        if finite_values.size == 0:
            dropped[feature] = "no finite values"
            continue
        lo, hi = np.quantile(finite_values, clip_quantiles)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            if remove_constant:
                dropped[feature] = "constant or invalid quantile range"
                continue
        clipped = np.clip(finite_values, lo, hi)
        median = float(np.median(clipped))
        mean = float(np.mean(clipped))
        std = float(np.std(clipped))
        if remove_constant and (not np.isfinite(std) or std <= 0.0):
            dropped[feature] = "zero std"
            continue
        kept.append(feature)
        medians[feature] = median
        lows[feature] = float(lo)
        highs[feature] = float(hi)
        means[feature] = mean
        stds[feature] = std if std > 0 else 1.0

    if not kept:
        raise RuntimeError("No features survived preprocessing. Check feature selection and parquet columns.")

    return PreprocessStats(kept, dropped, medians, lows, highs, means, stds)


def transform_features(df: pd.DataFrame, stats: PreprocessStats) -> np.ndarray:
    arrays: List[np.ndarray] = []
    for feature in stats.kept_features:
        x = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, stats.medians[feature])
        x = np.clip(x, stats.clip_low[feature], stats.clip_high[feature])
        x = (x - stats.means[feature]) / stats.stds[feature]
        x = np.where(np.isfinite(x), x, 0.0)
        arrays.append(x.astype(np.float32))
    return np.stack(arrays, axis=1)


def balanced_training_weights(labels: np.ndarray, abs_weights: np.ndarray) -> np.ndarray:
    labels = labels.astype(int)
    w = np.asarray(abs_weights, dtype=float)
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    out = w.copy()
    total = 0.0
    for label in [0, 1]:
        total += float(w[labels == label].sum())
    target = total / 2.0 if total > 0 else 1.0
    for label in [0, 1]:
        mask = labels == label
        denom = float(w[mask].sum())
        factor = target / denom if denom > 0 else 1.0
        out[mask] *= factor
    mean = float(out.mean()) if len(out) else 1.0
    if mean > 0:
        out /= mean
    return out.astype(np.float32)


class DenseDNN(nn.Module):
    def __init__(self, n_inputs: int, hidden_layers: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = n_inputs
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, int(hidden)))
            layers.append(nn.LayerNorm(int(hidden)))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(hidden)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def weighted_auc(y_true: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    y_true = y_true.astype(int)
    score = np.asarray(score, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mask = np.isfinite(score) & np.isfinite(weight) & (weight >= 0)
    y_true = y_true[mask]
    score = score[mask]
    weight = weight[mask]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(-score)
    y = y_true[order]
    w = weight[order]
    pos = y == 1
    neg = ~pos
    sum_pos = float(w[pos].sum())
    sum_neg = float(w[neg].sum())
    if sum_pos <= 0 or sum_neg <= 0:
        return float("nan")
    tpr = np.r_[0.0, np.cumsum(w * pos) / sum_pos]
    fpr = np.r_[0.0, np.cumsum(w * neg) / sum_neg]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, x=fpr))
    return float(np.trapz(tpr, x=fpr))


def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    w_abs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    scores: List[np.ndarray] = []
    losses: List[float] = []
    weights_seen: List[float] = []
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            yb = torch.from_numpy(y[start : start + batch_size].astype(np.float32)).to(device)
            wb = torch.from_numpy(w_abs[start : start + batch_size].astype(np.float32)).to(device)
            logits = model(xb)
            loss_vec = criterion(logits, yb)
            loss = (loss_vec * wb).sum() / torch.clamp(wb.sum(), min=1e-12)
            losses.append(float(loss.detach().cpu()))
            weights_seen.append(float(wb.sum().detach().cpu()))
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    score = np.concatenate(scores) if scores else np.array([], dtype=float)
    loss = float(np.average(losses, weights=weights_seen)) if losses else float("nan")
    auc = weighted_auc(y, score, w_abs)
    return {"loss": loss, "auc": auc, "score": score}


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val_abs: np.ndarray,
    cfg: Dict[str, Any],
    out_dir: Path,
) -> Tuple[nn.Module, Dict[str, Any]]:
    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    model = DenseDNN(
        n_inputs=x_train.shape[1],
        hidden_layers=cfg.get("hidden_layers", [256, 256, 128, 64, 32]),
        dropout=float(cfg.get("dropout", 0.05)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    batch_size = int(cfg.get("batch_size", 8192))
    epochs = int(cfg.get("epochs", 20))
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(w_train.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc = -float("inf")
    best_state = None

    print(f"[INFO] training on device={device}, amp={use_amp}, n_train={len(x_train)}, n_val={len(x_val)}")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_weight = 0.0
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss_vec = criterion(logits, yb)
                loss = (loss_vec * wb).sum() / torch.clamp(wb.sum(), min=1e-12)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            weight_sum = float(wb.sum().detach().cpu())
            running_loss += float(loss.detach().cpu()) * weight_sum
            running_weight += weight_sum

        train_loss = running_loss / running_weight if running_weight > 0 else float("nan")
        val = evaluate_model(model, x_val, y_val, w_val_abs, batch_size, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val["loss"]))
        history["val_auc"].append(float(val["auc"]))
        print(
            f"[INFO] epoch {epoch:03d}/{epochs}: "
            f"train_loss={train_loss:.5f} val_loss={val['loss']:.5f} val_auc={val['auc']:.5f}"
        )
        if math.isfinite(float(val["auc"])) and float(val["auc"]) > best_auc:
            best_auc = float(val["auc"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    safe_mkdir(out_dir / "checkpoints")
    torch.save({"model_state_dict": model.state_dict(), "history": history}, out_dir / "checkpoints" / "model.pt")
    return model, history


def plot_training_history(history: Dict[str, List[float]], out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(history.get("train_loss", []), label="train loss")
    ax1.plot(history.get("val_loss", []), label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("weighted BCE loss")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(history.get("val_auc", []), linestyle="--", label="val AUC")
    ax2.set_ylabel("weighted AUC")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def roc_curve_arrays(y_true: np.ndarray, score: np.ndarray, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y = y_true.astype(int)[order]
    w = weight[order]
    pos = y == 1
    neg = ~pos
    sum_pos = max(float(w[pos].sum()), 1e-12)
    sum_neg = max(float(w[neg].sum()), 1e-12)
    tpr = np.r_[0.0, np.cumsum(w * pos) / sum_pos]
    fpr = np.r_[0.0, np.cumsum(w * neg) / sum_neg]
    return fpr, tpr


def plot_score_outputs(y_true: np.ndarray, score: np.ndarray, weight: np.ndarray, out_dir: Path) -> None:
    safe_mkdir(out_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, name in [(1, "signal"), (0, "background")]:
        mask = y_true == label
        ax.hist(score[mask], bins=50, weights=weight[mask], histtype="step", density=True, label=name)
    ax.set_xlabel("DNN score")
    ax.set_ylabel("weighted density")
    ax.set_title("Validation score distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution_validation.png")
    plt.close(fig)

    fpr, tpr = roc_curve_arrays(y_true, score, weight)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr)
    ax.set_xlabel("background efficiency")
    ax.set_ylabel("signal efficiency")
    ax.set_title("Weighted ROC")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_validation.png")
    plt.close(fig)

    sic = tpr / np.sqrt(np.maximum(fpr, 1e-6))
    finite = np.isfinite(sic)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tpr[finite], sic[finite])
    ax.set_xlabel("signal efficiency")
    ax.set_ylabel("SIC = sig eff / sqrt(bkg eff)")
    ax.set_title("Significance improvement curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sic_validation.png")
    plt.close(fig)


def choose_columns_root(args: argparse.Namespace, cfg: Dict[str, Any]) -> Path:
    if args.columns_root:
        return Path(args.columns_root)
    eos = Path(cfg["paths"]["eos_columns_root"])
    mounted = Path(cfg["paths"]["mounted_columns_root"])
    if eos.exists():
        return eos
    return mounted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke_one_sr_dnn.yaml")
    parser.add_argument("--columns-root", default=None, help="Override columns root path.")
    parser.add_argument("--region", default=None, help="Override one SR region name.")
    parser.add_argument("--fraction", type=float, default=None, help="Override per-file random sampling fraction.")
    parser.add_argument("--max-events-per-class", type=int, default=None)
    parser.add_argument("--max-files-per-group", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--inspect-only", action="store_true", help="Only write manifest/column report and feature plots; skip training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))

    seed = int(cfg["training"].get("seed", 20260701))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    columns_root = choose_columns_root(args, cfg)
    region = args.region or cfg["selection"]["region"]
    fraction = float(args.fraction if args.fraction is not None else cfg["selection"].get("fraction", 0.02))
    max_events_per_class = args.max_events_per_class
    if max_events_per_class is None:
        max_events_per_class = cfg["selection"].get("max_events_per_class")
    max_files_per_group = args.max_files_per_group
    if max_files_per_group is None:
        max_files_per_group = cfg["selection"].get("max_files_per_group")

    output_base = Path(args.output_dir or cfg["paths"].get("output_dir", "outputs/smoke_one_sr_dnn"))
    run_tag = time.strftime("%Y%m%d_%H%M%S") + f"_{sanitize_filename(region)}"
    out_dir = safe_mkdir(output_base / run_tag)
    plot_dir = safe_mkdir(out_dir / "plots")

    print(f"[INFO] columns_root = {columns_root}")
    print(f"[INFO] region       = {region}")
    print(f"[INFO] fraction     = {fraction}")
    print(f"[INFO] output       = {out_dir}")

    records = find_parquet_files(columns_root, region, cfg["samples"])
    records = limit_files_per_group(records, max_files_per_group)
    if not records:
        raise RuntimeError(f"No parquet files found for region={region} under {columns_root}")

    manifest = [asdict(r) for r in records]
    write_json(manifest, out_dir / "manifest.json")
    print(f"[INFO] found {len(records)} parquet files")

    column_report = collect_column_report(records)
    write_json(column_report, out_dir / "branch_report.json")

    features, weight_col, event_col = select_numeric_features(column_report, cfg["features"])
    feature_info = {
        "n_selected_features_before_preprocessing": len(features),
        "selected_features_before_preprocessing": features,
        "weight_column": weight_col,
        "event_column": event_col,
    }
    write_json(feature_info, out_dir / "selected_branches.json")
    print(f"[INFO] selected {len(features)} numeric feature candidates")
    print(f"[INFO] weight column = {weight_col or 'UNIT WEIGHTS'}")
    print(f"[INFO] event column  = {event_col or 'stable hash fallback'}")

    data = read_sampled_parquets(
        records=records,
        feature_columns=features,
        weight_col=weight_col,
        event_col=event_col,
        fraction=fraction,
        seed=seed,
        max_events_per_class=max_events_per_class,
    )
    write_json(summarize_dataframe(data, features), out_dir / "sample_summary.json")

    if cfg.get("plots", {}).get("make_feature_svb_plots", True):
        feature_plot_paths = make_feature_plots(
            data,
            features,
            plot_dir / "feature_svb_raw",
            bins=int(cfg.get("plots", {}).get("bins", 50)),
        )
        write_json({"feature_plots": feature_plot_paths}, out_dir / "feature_plot_index.json")

    if args.inspect_only:
        print("[INFO] --inspect-only set; skipping preprocessing/training")
        return

    train_fold_name = cfg["folding"].get("smoke_train_on_fold", "odd")
    train_fold = 1 if train_fold_name == "odd" else 0
    val_fold = 1 - train_fold
    train_df = data[data["__fold"] == train_fold].copy()
    val_df = data[data["__fold"] == val_fold].copy()
    if train_df.empty or val_df.empty:
        raise RuntimeError("Train or validation fold is empty. Check fold definition and event column.")
    if train_df["__label"].nunique() < 2 or val_df["__label"].nunique() < 2:
        raise RuntimeError("Train/validation fold does not contain both signal and background.")

    pp_cfg = cfg["preprocessing"]
    stats = fit_preprocessor(
        train_df=train_df,
        feature_columns=features,
        clip_quantiles=tuple(pp_cfg.get("clip_quantiles", [0.001, 0.999])),
        min_finite_fraction=float(pp_cfg.get("min_finite_fraction", 0.50)),
        remove_constant=bool(pp_cfg.get("remove_constant_features", True)),
    )
    write_json(asdict(stats), out_dir / "preprocess_stats.json")
    print(f"[INFO] kept {len(stats.kept_features)} features after preprocessing")

    x_train = transform_features(train_df, stats)
    x_val = transform_features(val_df, stats)
    y_train = train_df["__label"].to_numpy(dtype=np.int64)
    y_val = val_df["__label"].to_numpy(dtype=np.int64)
    w_train_abs_raw = train_df["__signed_weight"].abs().to_numpy(dtype=np.float32)
    w_val_abs = val_df["__signed_weight"].abs().to_numpy(dtype=np.float32)
    w_train = balanced_training_weights(y_train, w_train_abs_raw)

    train_cfg = dict(cfg["training"])
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.device is not None:
        train_cfg["device"] = args.device

    model, history = train_model(
        x_train=x_train,
        y_train=y_train,
        w_train=w_train,
        x_val=x_val,
        y_val=y_val,
        w_val_abs=w_val_abs,
        cfg=train_cfg,
        out_dir=out_dir,
    )
    write_json(history, out_dir / "training_history.json")
    plot_training_history(history, plot_dir / "training_curves.png")

    device = torch.device("cuda" if (train_cfg.get("device", "auto") == "auto" and torch.cuda.is_available()) else train_cfg.get("device", "cpu"))
    if str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    val_eval = evaluate_model(model, x_val, y_val, w_val_abs, int(train_cfg.get("batch_size", 8192)), device)
    val_score = val_eval.pop("score")
    metrics = {
        "region": region,
        "columns_root": str(columns_root),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_features": int(len(stats.kept_features)),
        "validation": val_eval,
        "train_fold": int(train_fold),
        "val_fold": int(val_fold),
        "weight_column": weight_col,
        "event_column": event_col,
        "device": str(device),
    }
    write_json(metrics, out_dir / "metrics.json")
    plot_score_outputs(y_val, val_score, w_val_abs, plot_dir / "scores")

    scored = val_df[["__sample_name", "__region_name", "__process_group", "__label", "__signed_weight", "__fold"]].copy()
    scored["score_dnn_smoke"] = val_score
    scored.to_parquet(out_dir / "scored_validation.parquet", index=False)

    print("[INFO] smoke training complete")
    print(f"[INFO] metrics: {out_dir / 'metrics.json'}")
    print(f"[INFO] plots:   {plot_dir}")


if __name__ == "__main__":
    main()
