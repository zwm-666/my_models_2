"""
PEMFC Fault Classification Dataset.

Loads processed PEMFC data (operational time series, EIS spectra,
operating conditions) and returns samples in a standardised dictionary
format consumed by the CAPT-UniShape model.

Expected on-disk layout
-----------------------
data/processed/
    {domain_name}/
        {sample_id}.npz          # keys: x_op, x_eis (optional), cond, label
data/splits/
    {split_name}.json            # list of {"sample_id", "domain", "path"}
data/metadata/
    label_map.json               # {"coarse": {...}, "fine": {...}}
    norm_stats.json              # per-channel mean / std for normalisation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default label mappings (used when no external label_map.json is provided)
# ---------------------------------------------------------------------------
DEFAULT_COARSE_MAP: Dict[int, int] = {
    0: 0,  # Normal
    1: 1,  # Flooding
    2: 1,  # Drying  -> membrane degradation group
    3: 2,  # CO poisoning
    4: 2,  # Catalyst degradation -> poisoning group
}

DEFAULT_FINE_MAP: Dict[int, int] = {i: i for i in range(10)}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    """Load a JSON file and return its content."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_sample(path: Path) -> Dict[str, np.ndarray]:
    """Load a single ``.npz`` or ``.pkl`` sample from *path*.

    Supported formats
    -----------------
    * ``.npz`` -- loaded with :func:`numpy.load`
    * ``.pkl`` / ``.pickle`` -- loaded with :mod:`pickle`

    Returns a dict whose keys are at least ``x_op``, ``cond``, ``label``.
    ``x_eis`` is optional.
    """
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = dict(np.load(path, allow_pickle=True))
    elif suffix in (".pkl", ".pickle"):
        import pickle
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    else:
        raise ValueError(f"Unsupported file format: {suffix} ({path})")
    return data


class PEMFCDataset(Dataset):
    """PyTorch Dataset for PEMFC fault classification.

    Each sample is a dictionary::

        {
            "x_op":           Tensor[C_o, T],   # operational time series
            "x_eis":          Tensor[C_e, F] | None,
            "cond":           Tensor[d_u],       # operating condition vector
            "label":          int,               # fault class label
            "domain":         int,               # data-source domain ID
            "sample_id":      str,               # unique sample identifier
            "modality_mask":  Tensor[2],          # [op_avail, eis_avail]
        }

    Parameters
    ----------
    data_root : str or Path
        Root directory that contains ``processed/`` and ``splits/``.
    split : {"train", "val", "test"} or path to a JSON manifest.
        Which split to load.  If a path is given it is used directly;
        otherwise ``data_root / splits / {split}.json`` is read.
    label_granularity : {"fine", "coarse"}
        Which label mapping to apply (default ``"fine"``).
    domain_map : dict, optional
        Mapping from domain *name* (str) to domain *id* (int).
        If ``None``, domains are assigned integer IDs in alphabetical order
        of first appearance.
    transform : callable, optional
        A transform (or ``Compose``) applied to the sample dict **after**
        tensor conversion.
    eis_channels : int
        Expected number of EIS channels (real + imaginary = 2 by default).
    eis_freq_bins : int
        Expected number of EIS frequency bins (default 50).
    max_op_length : int or None
        If given, operational time series longer than this are truncated
        from the right.
    label_map_path : str or Path, optional
        Override path to ``label_map.json``.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        label_granularity: Literal["fine", "coarse"] = "fine",
        domain_map: Optional[Dict[str, int]] = None,
        transform: Optional[Any] = None,
        eis_channels: int = 2,
        eis_freq_bins: int = 50,
        max_op_length: Optional[int] = None,
        label_map_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.processed_dir = self.data_root / "processed"
        self.splits_dir = self.data_root / "splits"

        self.label_granularity = label_granularity
        self.transform = transform
        self.eis_channels = eis_channels
        self.eis_freq_bins = eis_freq_bins
        self.max_op_length = max_op_length

        # ---- load split manifest -------------------------------------------
        self.manifest: List[Dict[str, Any]] = self._load_manifest(split)
        logger.info(
            "PEMFCDataset: loaded %d samples for split='%s'",
            len(self.manifest),
            split,
        )

        # ---- label mapping -------------------------------------------------
        self.label_map = self._load_label_map(label_map_path)

        # ---- domain mapping ------------------------------------------------
        if domain_map is not None:
            self.domain_map: Dict[str, int] = domain_map
        else:
            unique_domains = sorted(
                {entry["domain"] for entry in self.manifest}
            )
            self.domain_map = {name: idx for idx, name in enumerate(unique_domains)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self, split: str) -> List[Dict[str, Any]]:
        """Return a list of sample descriptors for *split*.

        Each descriptor is a dict with at least:
        ``{"sample_id": str, "domain": str, "path": str}``.

        If *split* looks like an existing file path it is loaded directly;
        otherwise we look for ``{splits_dir}/{split}.json``.
        """
        split_path = Path(split)
        if split_path.is_file():
            return _load_json(split_path)

        candidate = self.splits_dir / f"{split}.json"
        if candidate.is_file():
            return _load_json(candidate)

        # Fallback: scan processed directory and build manifest on the fly
        logger.warning(
            "Split file '%s' not found -- scanning processed/ directory.",
            candidate,
        )
        return self._scan_processed_dir()

    def _scan_processed_dir(self) -> List[Dict[str, Any]]:
        """Build a manifest by scanning ``data/processed/``."""
        manifest: List[Dict[str, Any]] = []
        if not self.processed_dir.is_dir():
            logger.warning("processed directory does not exist: %s", self.processed_dir)
            return manifest

        for domain_dir in sorted(self.processed_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain_name = domain_dir.name
            for sample_file in sorted(domain_dir.iterdir()):
                if sample_file.suffix.lower() in (".npz", ".pkl", ".pickle"):
                    manifest.append(
                        {
                            "sample_id": sample_file.stem,
                            "domain": domain_name,
                            "path": str(sample_file.relative_to(self.data_root)),
                        }
                    )
        return manifest

    def _load_label_map(
        self, label_map_path: Optional[Union[str, Path]]
    ) -> Dict[int, int]:
        """Return the integer -> integer label mapping."""
        # Try user-supplied or default location
        if label_map_path is not None:
            path = Path(label_map_path)
        else:
            path = self.data_root / "metadata" / "label_map.json"

        if path.is_file():
            raw = _load_json(path)
            mapping = raw.get(self.label_granularity, {})
            # JSON keys are strings; convert to int->int
            return {int(k): int(v) for k, v in mapping.items()}

        # Fallback to built-in defaults
        if self.label_granularity == "coarse":
            return dict(DEFAULT_COARSE_MAP)
        return dict(DEFAULT_FINE_MAP)

    def _resolve_path(self, entry: Dict[str, Any]) -> Path:
        """Resolve a manifest entry to an absolute file path."""
        rel = entry.get("path", "")
        if rel:
            candidate = self.data_root / rel
            if candidate.is_file():
                return candidate

        # Fallback: domain / sample_id.npz
        domain = entry["domain"]
        sid = entry["sample_id"]
        for ext in (".npz", ".pkl", ".pickle"):
            candidate = self.processed_dir / domain / f"{sid}{ext}"
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"Cannot find data file for sample_id='{sid}', domain='{domain}'"
        )

    def _map_label(self, raw_label: int) -> int:
        """Map a raw integer label through the current label map."""
        return self.label_map.get(raw_label, raw_label)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.manifest[index]
        sample_id: str = entry["sample_id"]
        domain_name: str = entry["domain"]
        domain_id: int = self.domain_map.get(domain_name, 0)

        # ---- load raw arrays -------------------------------------------
        file_path = self._resolve_path(entry)
        raw = _load_sample(file_path)

        # -- operational time series (required): shape [C_o, T] ----------
        x_op = torch.as_tensor(raw["x_op"], dtype=torch.float32)
        if x_op.ndim == 1:
            x_op = x_op.unsqueeze(0)                       # [1, T]
        if self.max_op_length is not None:
            x_op = x_op[:, : self.max_op_length]

        # -- EIS spectrum (optional): shape [C_e, F] ---------------------
        has_eis = "x_eis" in raw and raw["x_eis"] is not None
        if has_eis:
            x_eis_np = raw["x_eis"]
            if x_eis_np.size == 0:
                has_eis = False

        if has_eis:
            x_eis = torch.as_tensor(raw["x_eis"], dtype=torch.float32)
            if x_eis.ndim == 1:
                x_eis = x_eis.unsqueeze(0)                 # [1, F]
        else:
            x_eis = torch.zeros(
                self.eis_channels, self.eis_freq_bins, dtype=torch.float32
            )

        # -- operating conditions: shape [d_u] ---------------------------
        cond = torch.as_tensor(raw["cond"], dtype=torch.float32).squeeze()
        if cond.ndim == 0:
            cond = cond.unsqueeze(0)

        # -- label -------------------------------------------------------
        raw_label = int(raw["label"]) if np.ndim(raw["label"]) == 0 else int(raw["label"].item())
        label = self._map_label(raw_label)

        # -- modality mask [op_available, eis_available] -----------------
        modality_mask = torch.tensor(
            [1.0, 1.0 if has_eis else 0.0], dtype=torch.float32
        )

        sample: Dict[str, Any] = {
            "x_op": x_op,
            "x_eis": x_eis if has_eis else None,
            "cond": cond,
            "label": label,
            "domain": domain_id,
            "sample_id": sample_id,
            "modality_mask": modality_mask,
        }

        # ---- apply transforms ------------------------------------------
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Number of unique target labels in the current mapping."""
        return len(set(self.label_map.values()))

    @property
    def num_domains(self) -> int:
        """Number of distinct domains."""
        return len(self.domain_map)

    def get_labels(self) -> List[int]:
        """Return a list of mapped labels for every sample (useful for
        weighted samplers).

        Note: this requires loading every sample's label, so results are
        cached after the first call.
        """
        if not hasattr(self, "_cached_labels"):
            labels: List[int] = []
            for entry in self.manifest:
                file_path = self._resolve_path(entry)
                raw = _load_sample(file_path)
                raw_label = int(raw["label"]) if np.ndim(raw["label"]) == 0 else int(raw["label"].item())
                labels.append(self._map_label(raw_label))
            self._cached_labels = labels
        return self._cached_labels

    def __repr__(self) -> str:
        return (
            f"PEMFCDataset(samples={len(self)}, "
            f"classes={self.num_classes}, "
            f"domains={self.num_domains}, "
            f"granularity='{self.label_granularity}')"
        )
