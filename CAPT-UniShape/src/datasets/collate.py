"""
Custom collate function for the PEMFC fault-classification DataLoader.

Handles:
* **Variable-length** operational time series -- pads to the maximum
  length in the batch and produces a boolean ``op_padding_mask``.
* **Missing EIS** -- creates zero tensors for samples whose ``x_eis``
  is ``None`` and marks them via ``modality_mask``.
* Returns a flat dictionary of properly batched / stacked tensors that
  is ready to be consumed by the model's ``forward()`` method.

Usage
-----
>>> from torch.utils.data import DataLoader
>>> from src.datasets.collate import pemfc_collate_fn
>>> loader = DataLoader(dataset, batch_size=32, collate_fn=pemfc_collate_fn)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def pemfc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of sample dicts into a single batched dict.

    Parameters
    ----------
    batch : list of dict
        Each element is a sample dict returned by
        :class:`~src.datasets.pemfc_dataset.PEMFCDataset`.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``x_op``            : ``Tensor[B, C_o, T_max]``  -- padded op series
        ``op_padding_mask`` : ``BoolTensor[B, T_max]``   -- True = padded
        ``op_lengths``      : ``LongTensor[B]``          -- original lengths
        ``x_eis``           : ``Tensor[B, C_e, F]``      -- EIS (zero-filled
                              for missing samples)
        ``cond``            : ``Tensor[B, d_u]``
        ``label``           : ``LongTensor[B]``
        ``domain``          : ``LongTensor[B]``
        ``sample_id``       : ``list[str]``  (length B)
        ``modality_mask``   : ``Tensor[B, 2]``
    """
    B = len(batch)

    # ------------------------------------------------------------------
    # 1.  Operational time series -- variable length padding
    # ------------------------------------------------------------------
    op_list: List[torch.Tensor] = [s["x_op"] for s in batch]     # each [C_o, T_i]
    C_o = op_list[0].shape[0]
    lengths = [x.shape[-1] for x in op_list]
    T_max = max(lengths)

    padded_ops: List[torch.Tensor] = []
    padding_mask_list: List[torch.Tensor] = []

    for x, L in zip(op_list, lengths):
        pad_len = T_max - L
        if pad_len > 0:
            x_padded = F.pad(x, (0, pad_len), mode="constant", value=0.0)
        else:
            x_padded = x
        padded_ops.append(x_padded)

        # True where the position is padding (convention: True = ignore)
        mask = torch.zeros(T_max, dtype=torch.bool)
        mask[L:] = True
        padding_mask_list.append(mask)

    x_op_batch = torch.stack(padded_ops, dim=0)            # [B, C_o, T_max]
    op_padding_mask = torch.stack(padding_mask_list, dim=0) # [B, T_max]
    op_lengths = torch.tensor(lengths, dtype=torch.long)    # [B]

    # ------------------------------------------------------------------
    # 2.  EIS spectra -- handle None by filling with zeros
    # ------------------------------------------------------------------
    # Determine EIS shape from the first available sample.
    eis_shape: Optional[tuple] = None
    for s in batch:
        if s["x_eis"] is not None:
            eis_shape = s["x_eis"].shape          # (C_e, F)
            break

    modality_masks: List[torch.Tensor] = []
    eis_list: List[torch.Tensor] = []

    for s in batch:
        mm = s["modality_mask"].clone()

        if s["x_eis"] is not None:
            eis_list.append(s["x_eis"])
        else:
            # Create a zero placeholder -- use known shape or a fallback
            if eis_shape is not None:
                eis_list.append(torch.zeros(eis_shape, dtype=torch.float32))
            else:
                # If *no* sample in the batch has EIS we still need a tensor.
                # Use a minimal placeholder; the model should check modality_mask.
                eis_list.append(torch.zeros(2, 1, dtype=torch.float32))
            mm[1] = 0.0

        modality_masks.append(mm)

    # If shapes are heterogeneous because of the fallback, unify them
    # to the maximum along each dimension.
    if eis_shape is None:
        # Entire batch has no EIS -- pick dimensions from the placeholders
        eis_shape = eis_list[0].shape

    # Ensure all EIS tensors have the same shape (pad frequency axis if needed)
    F_max = max(e.shape[-1] for e in eis_list)
    C_e = eis_list[0].shape[0]
    unified_eis: List[torch.Tensor] = []
    for e in eis_list:
        if e.shape[-1] < F_max:
            e = F.pad(e, (0, F_max - e.shape[-1]), mode="constant", value=0.0)
        if e.shape[0] < C_e:
            e = F.pad(e, (0, 0, 0, C_e - e.shape[0]), mode="constant", value=0.0)
        unified_eis.append(e)

    x_eis_batch = torch.stack(unified_eis, dim=0)           # [B, C_e, F_max]
    modality_mask_batch = torch.stack(modality_masks, dim=0) # [B, 2]

    # ------------------------------------------------------------------
    # 3.  Condition vectors, labels, domains
    # ------------------------------------------------------------------
    cond_batch = torch.stack([s["cond"] for s in batch], dim=0)   # [B, d_u]
    label_batch = torch.tensor(
        [s["label"] for s in batch], dtype=torch.long
    )                                                              # [B]
    domain_batch = torch.tensor(
        [s["domain"] for s in batch], dtype=torch.long
    )                                                              # [B]

    # ------------------------------------------------------------------
    # 4.  Sample IDs (kept as plain list)
    # ------------------------------------------------------------------
    sample_ids: List[str] = [s["sample_id"] for s in batch]

    return {
        "x_op": x_op_batch,
        "op_padding_mask": op_padding_mask,
        "op_lengths": op_lengths,
        "x_eis": x_eis_batch,
        "cond": cond_batch,
        "label": label_batch,
        "domain": domain_batch,
        "sample_id": sample_ids,
        "modality_mask": modality_mask_batch,
    }
