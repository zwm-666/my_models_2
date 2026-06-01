"""Plot raw/baseline/proposed feature embeddings with t-SNE, UMAP or PCA."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["正常", "过湿", "过干"]
SPLIT_TO_VALUE = {"train": 0, "val": 1, "test": 2}
BASELINE_DISPLAY_NAMES = {
    "mlp": "MLP",
    "cnn1d": "CNN1D",
    "tcn": "TCN",
    "cnn_bilstm_attention": "CNN-BiLSTM-Attention",
    "lstm": "LSTM",
    "autoformer": "Autoformer",
    "transformer": "Transformer",
    "itransformer": "iTransformer",
}


def _as_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def split_indices_from_npz(data: Any, split: str) -> np.ndarray[Any, np.dtype[np.int64]]:
    normalized = split.lower()
    if normalized == "all":
        if "labels" in data or "y" in data:
            label_key = "labels" if "labels" in data else "y"
            length = np.asarray(data[label_key]).shape[0]
        elif "split" in data:
            length = np.asarray(data["split"]).shape[0]
        else:
            raise ValueError("NPZ data must contain labels/y or split to select all samples.")
        return np.arange(length, dtype=np.int64)
    if normalized not in SPLIT_TO_VALUE:
        raise ValueError(f"Unsupported split={split!r}. Use train, val, test or all.")
    if "split" not in data:
        raise ValueError("NPZ data must contain a split array for named split selection.")
    split_values = np.asarray(data["split"], dtype=np.int64)
    return np.where(split_values == SPLIT_TO_VALUE[normalized])[0].astype(np.int64)


def label_array(data: Any) -> np.ndarray[Any, np.dtype[np.int64]]:
    label_key = "labels" if "labels" in data else "y"
    return np.asarray(data[label_key], dtype=np.int64)


def condition_array(data: Any, indices: np.ndarray[Any, np.dtype[np.int64]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    cond_key = "x_cond" if "x_cond" in data else "cond"
    return np.asarray(data[cond_key][indices], dtype=np.float32)


def raw_feature_matrix(data: Any, indices: np.ndarray[Any, np.dtype[np.int64]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    x_op = np.asarray(data["x_op"][indices], dtype=np.float32).reshape(len(indices), -1)
    x_eis = np.asarray(data["x_eis"][indices], dtype=np.float32).reshape(len(indices), -1)
    x_cond = condition_array(data, indices).reshape(len(indices), -1)
    return np.concatenate([x_op, x_eis, x_cond], axis=1)


def embedding_axis_labels(method: str) -> tuple[str, str]:
    normalized = method.lower()
    if normalized == "tsne":
        return "t-SNE 1", "t-SNE 2"
    if normalized == "umap":
        return "UMAP 1", "UMAP 2"
    if normalized == "pca":
        return "PC1", "PC2"
    return "Embedding 1", "Embedding 2"


def panel_title(kind: str, detail: str | None = None) -> str:
    normalized = kind.lower()
    if normalized == "raw":
        return "(a) Raw input features"
    if normalized == "raw_source":
        return "(a) Raw source features"
    if normalized == "baseline":
        return "(b) Baseline model features"
    if normalized == "unishape_before":
        return "(c) UniShape features before shape-aware adapter"
    if normalized == "unishape_after":
        return "(d) UniShape final features after shape-aware adapter"
    if normalized == "proposed":
        return f"Proposed model features ({detail or 'h'})"
    return str(detail or kind)


def display_panel_title(title: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(title, width=max(10, int(width)), break_long_words=False))


def compact_panel_label(title: str) -> str:
    match = re.match(r"^\(([a-zA-Z])\)", str(title).strip())
    return f"({match.group(1)})" if match else str(title)


def panel_caption_text(panel_titles: list[str], width: int = 94) -> str:
    caption = "; ".join(str(title) for title in panel_titles)
    return "\n".join(textwrap.wrap(caption, width=max(40, int(width)), break_long_words=False))


def parse_feature_keys(feature_spec: str) -> list[str]:
    keys = [item.strip() for item in feature_spec.split("+") if item.strip()]
    if not keys:
        raise ValueError("feature spec must contain at least one aux feature key")
    return keys


def stratified_sample_indices(
    labels: np.ndarray[Any, np.dtype[np.int64]],
    indices: np.ndarray[Any, np.dtype[np.int64]],
    max_samples: int,
    seed: int,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    if max_samples <= 0 or len(indices) <= max_samples:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    per_class = max(1, int(np.ceil(max_samples / max(1, len(np.unique(labels[indices]))))))
    for label in sorted(np.unique(labels[indices]).tolist()):
        class_indices = indices[labels[indices] == label].copy()
        rng.shuffle(class_indices)
        selected.extend(class_indices[:per_class].tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    if len(selected_array) > max_samples:
        selected_array = rng.choice(selected_array, size=max_samples, replace=False)
    return np.asarray(sorted(selected_array.tolist()), dtype=np.int64)


def _standardize(features: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    x = np.asarray(features, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def raw_feature_columns_for_mode(frame: Any, *, mode: str, label_col: str) -> list[str]:
    normalized = str(mode).lower()
    if normalized == "all_numeric":
        return [
            str(col)
            for col in frame.select_dtypes(include=[np.number]).columns
            if str(col) not in {str(label_col), "__group_key__"}
        ]
    if normalized == "model_input":
        from experiments.build_official_npz_from_self_excel import COND_COLS, STACK_COLS

        return list(STACK_COLS) + list(COND_COLS)
    raise ValueError(f"Unsupported raw feature mode: {mode!r}")


def build_raw_self_window_feature_arrays(
    frame: Any,
    *,
    group_sets: list[set[object]],
    group_to_int: dict[object, int],
    group_column: str,
    stack_cols: list[str],
    cond_cols: list[str],
    label_col: str,
    window_size: int,
    strides: list[int],
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.int64]]]:
    """Build unstandardized self-measured Excel window features.

    The feature vector keeps physical Excel values: raw stack variables are
    flattened per time row, then raw condition/EIS statistic columns are averaged
    over the same window. No z-score normalization or constructed EIS sequence is
    applied here.
    """
    if group_column not in frame:
        if len(group_to_int) != 1:
            raise ValueError(f"Missing group column {group_column!r} in raw source frame.")
        frame = frame.copy()
        frame[group_column] = next(iter(group_to_int.keys()))
    features: list[np.ndarray[Any, Any]] = []
    labels: list[int] = []
    for group_set, stride in zip(group_sets, strides):
        for group in sorted(group_set, key=str):
            group_df = frame.loc[frame[group_column] == group].sort_index()
            if group_df.empty:
                continue
            label = int(group_df[label_col].iloc[0])
            stack_values = group_df[stack_cols].fillna(0).to_numpy(dtype=np.float32)
            cond_values = group_df[cond_cols].fillna(0).to_numpy(dtype=np.float32)
            row_count = int(len(group_df))
            starts = [0] if row_count < int(window_size) else list(range(0, row_count - int(window_size) + 1, max(1, int(stride))))
            for start in starts:
                if row_count < int(window_size):
                    stack_window = np.pad(stack_values, ((0, int(window_size) - row_count), (0, 0)), mode="edge")
                    cond_window = cond_values
                else:
                    stack_window = stack_values[start : start + int(window_size)]
                    cond_window = cond_values[start : start + int(window_size)]
                cond_summary = cond_window.mean(axis=0) if cond_window.shape[1] else np.empty((0,), dtype=np.float32)
                features.append(np.concatenate([stack_window.reshape(-1), cond_summary], axis=0).astype(np.float32))
                labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def raw_self_excel_feature_matrix(
    excel_path: Path,
    indices: np.ndarray[Any, np.dtype[np.int64]],
    *,
    sheet_name: str,
    window_size: int,
    stride_train: int,
    stride_val: int,
    stride_eval: int,
    split_mode: str,
    segment_gap_seconds: float,
    segment_block_seconds: float,
    segment_label_boundary: bool,
    random_state: int,
    test_size: float,
    val_size: float,
    group_split_strategy: str,
    split_retries: int,
    min_eval_class_windows: int,
    min_eval_class_groups: int,
    min_train_class_windows: int,
    min_train_class_groups: int,
    min_val_class_groups: int,
    min_test_class_groups: int,
    prefer_balanced_train_groups: bool,
    use_first_split_candidate: bool,
    raw_feature_mode: str,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.int64]], dict[str, Any]]:
    import pandas as pd

    from experiments.build_official_npz_from_self_excel import (
        COND_COLS,
        LABEL_COL,
        STACK_COLS,
        TIME_COL,
        _choose_group_split,
        _derive_group_keys,
        _derive_segment_group_keys,
        _group_split,
        _prepare_label_column,
        _split_quality,
    )

    frame = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    frame, label_source_col, label_value_map = _prepare_label_column(frame)
    expected = set(STACK_COLS + COND_COLS + [TIME_COL, LABEL_COL])
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected raw Excel columns: {sorted(missing)}")
    frame = frame.copy()
    if split_mode == "segment":
        frame["__group_key__"] = _derive_segment_group_keys(
            frame,
            gap_seconds=float(segment_gap_seconds),
            block_seconds=float(segment_block_seconds),
            label_boundary=bool(segment_label_boundary),
        )
    else:
        frame["__group_key__"] = _derive_group_keys(pd.Series(frame[TIME_COL].to_numpy()), split_mode)
    groups = np.asarray(frame["__group_key__"].unique())
    group_label_map = frame.groupby("__group_key__")[LABEL_COL].first()
    group_labels = np.asarray([group_label_map[group] for group in groups])
    row_counts = {group: int(count) for group, count in frame.groupby("__group_key__").size().items()}
    split_labels = [str(int(label)) for label in sorted(np.unique(group_labels).tolist())]
    if use_first_split_candidate:
        g_tr, g_va, g_te = _group_split(
            groups,
            group_labels,
            float(test_size),
            float(val_size),
            int(random_state),
            strategy=str(group_split_strategy),
        )
        split_quality = _split_quality(
            row_counts=row_counts,
            group_label_map=group_label_map,
            g_tr=g_tr,
            g_va=g_va,
            g_te=g_te,
            window_size=int(window_size),
            stride_train=int(stride_train),
            stride_val=int(stride_val),
            stride_eval=int(stride_eval),
            labels=split_labels,
            min_eval_class_windows=int(min_eval_class_windows),
            min_eval_class_groups=int(min_eval_class_groups),
            min_train_class_windows=int(min_train_class_windows),
            min_train_class_groups=int(min_train_class_groups),
            min_val_class_groups=int(min_val_class_groups),
            min_test_class_groups=int(min_test_class_groups),
            prefer_balanced_train_groups=bool(prefer_balanced_train_groups),
        )
        split_quality["chosen_seed"] = int(random_state)
        split_quality["attempt_index"] = 0
        split_quality["split_source"] = f"first_candidate:{group_split_strategy}"
    else:
        g_tr, g_va, g_te, split_quality = _choose_group_split(
            groups=groups,
            group_labels=group_labels,
            row_counts=row_counts,
            group_label_map=group_label_map,
            test_size=float(test_size),
            val_size=float(val_size),
            random_state=int(random_state),
            group_split_strategy=str(group_split_strategy),
            window_size=int(window_size),
            stride_train=int(stride_train),
            stride_val=int(stride_val),
            stride_eval=int(stride_eval),
            split_retries=int(split_retries),
            min_eval_class_windows=int(min_eval_class_windows),
            min_eval_class_groups=int(min_eval_class_groups),
            min_train_class_windows=int(min_train_class_windows),
            min_train_class_groups=int(min_train_class_groups),
            min_val_class_groups=int(min_val_class_groups),
            min_test_class_groups=int(min_test_class_groups),
            prefer_balanced_train_groups=bool(prefer_balanced_train_groups),
        )
    group_to_int = {group: idx for idx, group in enumerate(sorted(groups, key=str))}
    raw_feature_cols = raw_feature_columns_for_mode(frame, mode=raw_feature_mode, label_col=LABEL_COL)
    features_all, labels_all = build_raw_self_window_feature_arrays(
        frame,
        group_sets=[set(g_tr.tolist()), set(g_va.tolist()), set(g_te.tolist())],
        group_to_int=group_to_int,
        group_column="__group_key__",
        stack_cols=raw_feature_cols,
        cond_cols=[],
        label_col=str(LABEL_COL),
        window_size=int(window_size),
        strides=[int(stride_train), int(stride_val), int(stride_eval)],
    )
    return features_all[indices], labels_all[indices], {
        "path": str(excel_path),
        "sheet_name": sheet_name,
        "feature_source": f"unstandardized Excel {raw_feature_mode} window",
        "raw_feature_mode": str(raw_feature_mode),
        "raw_feature_columns": raw_feature_cols,
        "label_source_col": label_source_col,
        "label_value_map": label_value_map,
        "feature_shape_all": list(features_all.shape),
        "split_quality": split_quality,
    }


def _pca_2d(features: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    x = _standardize(features)
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[: min(2, vt.shape[0])].T
    reduced = x @ components
    if reduced.shape[1] == 1:
        reduced = np.concatenate([reduced, np.zeros((reduced.shape[0], 1), dtype=reduced.dtype)], axis=1)
    return np.asarray(reduced[:, :2], dtype=np.float32)


def reduce_features(
    features: np.ndarray[Any, Any],
    *,
    method: str = "tsne",
    seed: int = 42,
    perplexity: float = 30.0,
    umap_neighbors: int = 15,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    x = _standardize(features)
    normalized = method.lower()
    if normalized == "pca":
        return _pca_2d(x)
    if x.shape[0] < 3:
        return _pca_2d(x)
    if normalized == "tsne":
        from sklearn.manifold import TSNE

        effective_perplexity = min(float(perplexity), max(1.0, float(x.shape[0] - 1) / 3.0))
        reducer = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=int(seed),
        )
        return np.asarray(reducer.fit_transform(x), dtype=np.float32)
    if normalized == "umap":
        try:
            import umap  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("UMAP requires installing umap-learn: pip install umap-learn") from exc
        reducer = umap.UMAP(n_components=2, n_neighbors=int(umap_neighbors), random_state=int(seed))
        return np.asarray(reducer.fit_transform(x), dtype=np.float32)
    raise ValueError(f"Unsupported reduction method: {method!r}")


def _torch_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _checkpoint_state_dict(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict-like payload: {checkpoint_path}")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    return checkpoint, state_dict


def extract_torch_baseline_features(model: Any, model_key: str, x_op: Any, x_eis: Any, x_cond: Any) -> Any:
    import torch
    import torch.nn.functional as F

    normalized = model_key.lower()
    if normalized == "mlp":
        features = torch.cat([x_op.flatten(1), x_eis.flatten(1), x_cond], dim=1)
        return model.net[:-1](features)
    if normalized == "cnn1d":
        from experiments.run_official_baseline_experiments import _combined_sequence

        return model.net[:-1](_combined_sequence(x_op, x_eis, x_cond))
    if normalized == "cnn_bilstm_attention":
        from experiments.run_official_baseline_experiments import _combined_sequence

        sequence = model.conv(_combined_sequence(x_op, x_eis, x_cond)).transpose(1, 2)
        output, _ = model.lstm(sequence)
        weights = torch.softmax(model.attention(output).squeeze(-1), dim=1).unsqueeze(-1)
        return torch.sum(output * weights, dim=1)
    if normalized == "tcn":
        from experiments.run_official_baseline_experiments import _combined_sequence

        return model.net(_combined_sequence(x_op, x_eis, x_cond)).mean(dim=2)
    if normalized == "lstm":
        from experiments.run_official_baseline_experiments import _combined_sequence

        sequence = _combined_sequence(x_op, x_eis, x_cond).transpose(1, 2)
        output, _ = model.lstm(sequence)
        return output.mean(dim=1)
    if normalized == "autoformer":
        from experiments.run_official_baseline_experiments import _combined_sequence

        seasonal, trend = model._decompose(_combined_sequence(x_op, x_eis, x_cond))
        encoded_input = model.seasonal_proj(seasonal) + model.trend_proj(trend) + model.pos_embed[:, : seasonal.shape[1]]
        return model.encoder(encoded_input).mean(dim=1)
    if normalized == "transformer":
        from experiments.run_official_baseline_experiments import _combined_sequence

        sequence = _combined_sequence(x_op, x_eis, x_cond).transpose(1, 2)
        encoded = model.encoder(model.input_proj(sequence) + model.pos_embed[:, : sequence.shape[1]])
        return encoded.mean(dim=1)
    if normalized == "itransformer":
        op_tokens = model.op_encoder(model.op_proj(x_op) + model.op_variable_embed[:, : x_op.shape[1]])
        eis_tokens = model.eis_encoder(model.eis_proj(x_eis) + model.eis_variable_embed[:, : x_eis.shape[1]])
        z_cond = model.cond_encoder(x_cond)
        return torch.cat([op_tokens.mean(dim=1), eis_tokens.mean(dim=1), z_cond], dim=1)
    raise KeyError(f"Unsupported torch baseline model: {model_key}")


def _iter_npz_batches(data: Any, indices: np.ndarray[Any, np.dtype[np.int64]], batch_size: int, device: Any):
    import torch

    cond_key = "x_cond" if "x_cond" in data else "cond"
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        yield (
            torch.as_tensor(data["x_op"][batch_indices], dtype=torch.float32, device=device),
            torch.as_tensor(data["x_eis"][batch_indices], dtype=torch.float32, device=device),
            torch.as_tensor(data[cond_key][batch_indices], dtype=torch.float32, device=device),
        )


def baseline_shape_dataset(data: Any) -> Any:
    cond_key = "x_cond" if "x_cond" in data else "cond"
    return SimpleNamespace(
        x_op=np.zeros((1, int(data["x_op"].shape[1]), int(data["x_op"].shape[2])), dtype=np.float32),
        x_eis=np.zeros((1, int(data["x_eis"].shape[1]), int(data["x_eis"].shape[2])), dtype=np.float32),
        x_cond=np.zeros((1, int(data[cond_key].shape[1])), dtype=np.float32),
    )


def extract_baseline_feature_matrix(
    npz_path: Path,
    checkpoint_path: Path,
    indices: np.ndarray[Any, np.dtype[np.int64]],
    *,
    model_key: str | None,
    hidden_dim: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    device_name: str,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str]:
    import torch

    from experiments.run_official_baseline_experiments import _build_torch_model

    payload, state_dict = _checkpoint_state_dict(checkpoint_path)
    resolved_model_key = model_key or str(payload.get("model_key") or checkpoint_path.parent.name)
    data = np.load(npz_path)
    shape_ds = baseline_shape_dataset(data)
    num_classes = int(np.max(label_array(data)) + 1)
    model = _build_torch_model(resolved_model_key, shape_ds, num_classes, hidden_dim, d_model, num_layers, dropout)
    model.load_state_dict(state_dict, strict=True)
    device = _torch_device(device_name)
    model.to(device)
    model.eval()

    features: list[np.ndarray[Any, Any]] = []
    with torch.no_grad():
        for x_op, x_eis, x_cond in _iter_npz_batches(data, indices, batch_size, device):
            batch_features = extract_torch_baseline_features(model, resolved_model_key, x_op, x_eis, x_cond)
            features.append(batch_features.detach().cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32), resolved_model_key


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def extract_proposed_feature_matrix(
    npz_path: Path,
    checkpoint_path: Path,
    indices: np.ndarray[Any, np.dtype[np.int64]],
    *,
    config_path: Path | None,
    feature_key: str,
    batch_size: int,
    device_name: str,
    allow_partial_load: bool,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    import torch

    from models import build_model_from_config
    from train import FuelCellNPZDataset, sync_config_with_dataset

    payload, state_dict = _checkpoint_state_dict(checkpoint_path)
    config = payload.get("config")
    if not isinstance(config, dict):
        if config_path is None:
            raise ValueError("Proposed checkpoint has no config; pass --proposed-config.")
        config = _load_yaml(config_path)
    dataset = FuelCellNPZDataset(npz_path)
    model_config = sync_config_with_dataset(config, dataset)
    model = build_model_from_config(model_config)
    load_result = model.load_state_dict(state_dict, strict=not allow_partial_load)
    if allow_partial_load and (load_result.missing_keys or load_result.unexpected_keys):
        print(
            f"Partial proposed load: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}",
            flush=True,
        )
    device = _torch_device(device_name)
    model.to(device)
    model.eval()

    data = np.load(npz_path)
    features: list[np.ndarray[Any, Any]] = []
    feature_keys = parse_feature_keys(feature_key)
    with torch.no_grad():
        for x_op, x_eis, x_cond in _iter_npz_batches(data, indices, batch_size, device):
            _logits, aux = model(x_op, x_eis, x_cond)
            tensors = []
            for key in feature_keys:
                if key not in aux or aux[key] is None:
                    raise KeyError(f"Proposed model aux output does not contain feature_key={key!r}")
                tensors.append(aux[key])
            batch_features = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=1)
            features.append(batch_features.detach().cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32)


def plot_embedding_panels(
    embeddings: dict[str, np.ndarray[Any, Any]],
    labels: np.ndarray[Any, np.dtype[np.int64]],
    output_path: Path,
    class_names: list[str],
    dpi: int,
    method: str,
    panel_label_style: str,
    legend_position: str,
    include_panel_caption: bool,
) -> Path:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2"]
    x_label, y_label = embedding_axis_labels(method)
    if len(embeddings) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), squeeze=False)
        flat_axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, len(embeddings), figsize=(5.2 * len(embeddings), 4.8), squeeze=False)
        flat_axes = axes.ravel()
    handles = None
    labels_for_legend = None
    for ax, (title, coords) in zip(flat_axes, embeddings.items()):
        for offset, label in enumerate(sorted(np.unique(labels).tolist())):
            mask = labels == label
            name = class_names[int(label)] if int(label) < len(class_names) else str(label)
            ax.scatter(coords[mask, 0], coords[mask, 1], s=24, alpha=0.78, label=name, color=colors[offset % len(colors)], edgecolors="none")
        shown_title = compact_panel_label(title) if panel_label_style == "compact" else display_panel_title(title)
        ax.set_title(shown_title, fontsize=12 if panel_label_style != "compact" else 13)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.25)
        handles, labels_for_legend = ax.get_legend_handles_labels()
    for ax in flat_axes[len(embeddings) :]:
        ax.axis("off")
    top_margin = 0.94
    bottom_margin = 0.06
    if handles and labels_for_legend:
        if legend_position == "top":
            fig.legend(handles, labels_for_legend, loc="upper center", ncol=len(labels_for_legend), frameon=False, bbox_to_anchor=(0.5, 0.995))
            top_margin = 0.90
        else:
            fig.legend(handles, labels_for_legend, loc="lower center", ncol=len(labels_for_legend), frameon=False)
            bottom_margin = 0.08
    if include_panel_caption:
        fig.text(
            0.5,
            0.018,
            panel_caption_text(list(embeddings.keys())),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        bottom_margin = max(bottom_margin, 0.12)
    fig.tight_layout(rect=(0.0, bottom_margin, 1.0, top_margin))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_embedding_csv(
    path: Path,
    embeddings: dict[str, np.ndarray[Any, Any]],
    labels: np.ndarray[Any, np.dtype[np.int64]],
    indices: np.ndarray[Any, np.dtype[np.int64]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "sample_index", "label", "x", "y"])
        writer.writeheader()
        for panel, coords in embeddings.items():
            for row, sample_index in enumerate(indices.tolist()):
                writer.writerow(
                    {
                        "panel": panel,
                        "sample_index": int(sample_index),
                        "label": int(labels[row]),
                        "x": float(coords[row, 0]),
                        "y": float(coords[row, 1]),
                    }
                )
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="绘制 raw / baseline / UniShape staged features 的 t-SNE、UMAP 或 PCA 可视化图")
    parser.add_argument("--data", required=True, help="包含 x_op/x_eis/x_cond/labels/split 的 NPZ 数据")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--raw-self-excel", default=None, help="自测数据原始 Excel；提供后 (a) 使用未标准化 Excel 原始物理值")
    parser.add_argument("--raw-self-sheet", default="Sheet1")
    parser.add_argument("--raw-feature-mode", choices=["model_input", "all_numeric"], default="model_input")
    parser.add_argument("--raw-window-size", type=int, default=64)
    parser.add_argument("--raw-stride-train", type=int, default=16)
    parser.add_argument("--raw-stride-val", type=int, default=32)
    parser.add_argument("--raw-stride-eval", type=int, default=32)
    parser.add_argument("--raw-split-mode", default="segment")
    parser.add_argument("--raw-segment-gap-seconds", type=float, default=600.0)
    parser.add_argument("--raw-segment-block-seconds", type=float, default=300.0)
    parser.add_argument("--raw-segment-label-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-test-size", type=float, default=0.20)
    parser.add_argument("--raw-val-size", type=float, default=0.25)
    parser.add_argument("--raw-group-split-strategy", default="holdout_first")
    parser.add_argument("--raw-use-first-split-candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-split-retries", type=int, default=50)
    parser.add_argument("--raw-min-eval-class-windows", type=int, default=5)
    parser.add_argument("--raw-min-eval-class-groups", type=int, default=1)
    parser.add_argument("--raw-min-train-class-windows", type=int, default=1)
    parser.add_argument("--raw-min-train-class-groups", type=int, default=1)
    parser.add_argument("--raw-min-val-class-groups", type=int, default=1)
    parser.add_argument("--raw-min-test-class-groups", type=int, default=1)
    parser.add_argument("--raw-prefer-balanced-train-groups", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/paper_figures/feature_embeddings")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--class-names", default=",".join(CLASS_NAMES))
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--panel-label-style", choices=["full", "compact"], default="full")
    parser.add_argument("--legend-position", choices=["bottom", "top"], default="bottom")
    parser.add_argument("--panel-caption", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--baseline-checkpoint", default=None, help="Torch 基线 best.ckpt；不提供则跳过基线面板")
    parser.add_argument("--baseline-model-key", default=None, help="mlp/cnn1d/tcn/cnn_bilstm_attention/lstm/autoformer/transformer/itransformer；默认从 checkpoint 或目录名推断")
    parser.add_argument("--baseline-hidden-dim", type=int, default=64)
    parser.add_argument("--baseline-d-model", type=int, default=64)
    parser.add_argument("--baseline-num-layers", type=int, default=2)
    parser.add_argument("--baseline-dropout", type=float, default=0.1)

    parser.add_argument("--proposed-checkpoint", default=None, help="本文模型 checkpoint，如 best.ckpt 或 best_val.ckpt；不提供则跳过本文模型面板")
    parser.add_argument("--proposed-config", default="configs/rbf_kanfusion.yaml")
    parser.add_argument("--figure-layout", choices=["four_panel", "three_panel"], default="four_panel")
    parser.add_argument("--proposed-feature-key", default="h", help="three_panel 布局下本文模型 aux 中的特征键，常用 h 或 z_fused")
    parser.add_argument("--unishape-before-feature-keys", default="z_op+z_eis", help="four_panel 中 (c) 面板使用的 aux 特征，可用 + 拼接")
    parser.add_argument("--unishape-after-feature-keys", default="h", help="four_panel 中 (d) 面板使用的 aux 特征，可用 + 拼接")
    parser.add_argument("--allow-partial-load", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    data_path = _as_project_path(args.data)
    if data_path is None or not data_path.is_file():
        raise FileNotFoundError(f"Missing NPZ data: {data_path}")
    data = np.load(data_path)
    labels_all = label_array(data)
    indices = split_indices_from_npz(data, args.split)
    indices = stratified_sample_indices(labels_all, indices, max_samples=int(args.max_samples), seed=int(args.seed))
    labels = labels_all[indices]

    raw_source_meta: dict[str, Any] | None = None
    raw_self_excel = _as_project_path(args.raw_self_excel)
    if raw_self_excel is not None:
        if not raw_self_excel.is_file():
            raise FileNotFoundError(f"Missing raw self Excel: {raw_self_excel}")
        raw_features, raw_labels, raw_source_meta = raw_self_excel_feature_matrix(
            raw_self_excel,
            indices,
            sheet_name=str(args.raw_self_sheet),
            window_size=int(args.raw_window_size),
            stride_train=int(args.raw_stride_train),
            stride_val=int(args.raw_stride_val),
            stride_eval=int(args.raw_stride_eval),
            split_mode=str(args.raw_split_mode),
            segment_gap_seconds=float(args.raw_segment_gap_seconds),
            segment_block_seconds=float(args.raw_segment_block_seconds),
            segment_label_boundary=bool(args.raw_segment_label_boundary),
            random_state=int(args.seed),
            test_size=float(args.raw_test_size),
            val_size=float(args.raw_val_size),
            group_split_strategy=str(args.raw_group_split_strategy),
            split_retries=int(args.raw_split_retries),
            min_eval_class_windows=int(args.raw_min_eval_class_windows),
            min_eval_class_groups=int(args.raw_min_eval_class_groups),
            min_train_class_windows=int(args.raw_min_train_class_windows),
            min_train_class_groups=int(args.raw_min_train_class_groups),
            min_val_class_groups=int(args.raw_min_val_class_groups),
            min_test_class_groups=int(args.raw_min_test_class_groups),
            prefer_balanced_train_groups=bool(args.raw_prefer_balanced_train_groups),
            use_first_split_candidate=bool(args.raw_use_first_split_candidate),
            raw_feature_mode=str(args.raw_feature_mode),
        )
        if not np.array_equal(raw_labels, labels):
            raise ValueError("Raw Excel labels do not align with selected NPZ labels; check raw split/window parameters.")
        feature_panels: dict[str, np.ndarray[Any, Any]] = {panel_title("raw_source"): raw_features}
    else:
        feature_panels = {panel_title("raw"): raw_feature_matrix(data, indices)}
    baseline_checkpoint = _as_project_path(args.baseline_checkpoint)
    if baseline_checkpoint is not None:
        baseline_features, baseline_key = extract_baseline_feature_matrix(
            data_path,
            baseline_checkpoint,
            indices,
            model_key=args.baseline_model_key,
            hidden_dim=int(args.baseline_hidden_dim),
            d_model=int(args.baseline_d_model),
            num_layers=int(args.baseline_num_layers),
            dropout=float(args.baseline_dropout),
            batch_size=int(args.batch_size),
            device_name=str(args.device),
        )
        feature_panels[panel_title("baseline", baseline_key)] = baseline_features

    proposed_checkpoint = _as_project_path(args.proposed_checkpoint)
    if proposed_checkpoint is not None:
        if args.figure_layout == "four_panel":
            before_features = extract_proposed_feature_matrix(
                data_path,
                proposed_checkpoint,
                indices,
                config_path=_as_project_path(args.proposed_config),
                feature_key=str(args.unishape_before_feature_keys),
                batch_size=int(args.batch_size),
                device_name=str(args.device),
                allow_partial_load=bool(args.allow_partial_load),
            )
            after_features = extract_proposed_feature_matrix(
                data_path,
                proposed_checkpoint,
                indices,
                config_path=_as_project_path(args.proposed_config),
                feature_key=str(args.unishape_after_feature_keys),
                batch_size=int(args.batch_size),
                device_name=str(args.device),
                allow_partial_load=bool(args.allow_partial_load),
            )
            feature_panels[panel_title("unishape_before")] = before_features
            feature_panels[panel_title("unishape_after")] = after_features
        else:
            proposed_features = extract_proposed_feature_matrix(
                data_path,
                proposed_checkpoint,
                indices,
                config_path=_as_project_path(args.proposed_config),
                feature_key=str(args.proposed_feature_key),
                batch_size=int(args.batch_size),
                device_name=str(args.device),
                allow_partial_load=bool(args.allow_partial_load),
            )
            feature_panels[panel_title("proposed", str(args.proposed_feature_key))] = proposed_features

    embeddings = {
        panel: reduce_features(
            features,
            method=str(args.method),
            seed=int(args.seed),
            perplexity=float(args.perplexity),
            umap_neighbors=int(args.umap_neighbors),
        )
        for panel, features in feature_panels.items()
    }

    output_dir = _as_project_path(args.output_dir)
    if output_dir is None:
        raise ValueError("output-dir must not be empty")
    output_name = args.output_name or f"feature_embeddings_{args.split}_{args.method}.png"
    figure_path = plot_embedding_panels(
        embeddings,
        labels,
        output_dir / output_name,
        class_names=[item.strip() for item in str(args.class_names).split(",") if item.strip()],
        dpi=int(args.dpi),
        method=str(args.method),
        panel_label_style=str(args.panel_label_style),
        legend_position=str(args.legend_position),
        include_panel_caption=bool(args.panel_caption),
    )
    csv_path = write_embedding_csv(output_dir / f"{Path(output_name).stem}.csv", embeddings, labels, indices)
    meta_path = output_dir / f"{Path(output_name).stem}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "data": str(data_path),
                "split": args.split,
                "method": args.method,
                "seed": int(args.seed),
                "sample_count": int(len(indices)),
                "panels": list(embeddings.keys()),
                "figure_layout": str(args.figure_layout),
                "panel_label_style": str(args.panel_label_style),
                "legend_position": str(args.legend_position),
                "panel_caption": bool(args.panel_caption),
                "raw_source": raw_source_meta,
                "unishape_before_feature_keys": str(args.unishape_before_feature_keys),
                "unishape_after_feature_keys": str(args.unishape_after_feature_keys),
                "figure": str(figure_path),
                "csv": str(csv_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(figure_path)
    print(csv_path)
    print(meta_path)


if __name__ == "__main__":
    main()

