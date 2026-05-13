"""
Data augmentation and preprocessing transforms for PEMFC fault classification.

All transforms operate on a *sample dict* (the format returned by
:class:`PEMFCDataset`) and return a modified copy of that dict.  Tensor
shapes follow the convention:

* ``x_op``  -- ``[C_o, T]`` operational time series
* ``x_eis`` -- ``[C_e, F]`` EIS spectrum  (may be ``None``)
* ``cond``  -- ``[d_u]``    operating condition vector

Usage
-----
>>> from src.datasets.transforms import Compose, Normalize, GaussianNoise
>>> tfm = Compose([Normalize(mean, std), GaussianNoise(snr_db=20)])
>>> sample = tfm(sample)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

# Type alias for the sample dictionary flowing through the pipeline.
Sample = Dict[str, Any]


# =========================================================================
# Compose
# =========================================================================

class Compose:
    """Sequentially apply a list of transforms to a sample dict.

    Parameters
    ----------
    transforms : sequence of callables
        Each callable receives and returns a ``Sample`` dict.
    """

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + ",\n".join(lines) + "\n])"


# =========================================================================
# Normalisation
# =========================================================================

class Normalize:
    """Per-channel z-score normalisation for operational time series.

    .. math::
        x_{\\text{norm}} = \\frac{x - \\mu}{\\sigma + \\epsilon}

    Parameters
    ----------
    mean : Tensor or list of float
        Per-channel mean, shape ``[C_o]`` or ``[C_o, 1]``.
    std : Tensor or list of float
        Per-channel standard deviation, same shape as *mean*.
    eps : float
        Small constant to avoid division by zero.
    key : str
        Which sample key to normalise (default ``"x_op"``).
    """

    def __init__(
        self,
        mean: Union[torch.Tensor, List[float]],
        std: Union[torch.Tensor, List[float]],
        eps: float = 1e-8,
        key: str = "x_op",
    ) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.eps = eps
        self.key = key

        # Ensure shape is [C, 1] so broadcasting works with [C, T]
        if self.mean.ndim == 1:
            self.mean = self.mean.unsqueeze(-1)
        if self.std.ndim == 1:
            self.std = self.std.unsqueeze(-1)

    def __call__(self, sample: Sample) -> Sample:
        x = sample[self.key]
        if x is not None:
            device = x.device
            mean = self.mean.to(device)
            std = self.std.to(device)
            sample[self.key] = (x - mean) / (std + self.eps)
        return sample

    def __repr__(self) -> str:
        return (
            f"Normalize(key='{self.key}', "
            f"channels={self.mean.shape[0]}, eps={self.eps})"
        )


# =========================================================================
# Window Crop
# =========================================================================

class WindowCrop:
    """Crop a contiguous window from the time dimension of ``x_op``.

    During training a random start index is chosen; during evaluation the
    crop is taken from a fixed position (default: centre).

    Parameters
    ----------
    window_size : int
        Length of the crop along the time axis.
    mode : {"random", "center", "start", "end"}
        Where to place the window when not random.
    random : bool
        If ``True`` override *mode* with a uniformly random start (training).
    """

    def __init__(
        self,
        window_size: int,
        mode: str = "center",
        random: bool = True,
    ) -> None:
        self.window_size = window_size
        self.mode = mode
        self.random = random

    def __call__(self, sample: Sample) -> Sample:
        x = sample["x_op"]                               # [C, T]
        T = x.shape[-1]

        if T <= self.window_size:
            # Pad on the right if the signal is shorter than the window
            pad_len = self.window_size - T
            x = F.pad(x, (0, pad_len), mode="constant", value=0.0)
            sample["x_op"] = x
            return sample

        if self.random:
            start = torch.randint(0, T - self.window_size, (1,)).item()
        elif self.mode == "center":
            start = (T - self.window_size) // 2
        elif self.mode == "start":
            start = 0
        elif self.mode == "end":
            start = T - self.window_size
        else:
            raise ValueError(f"Unknown crop mode: {self.mode}")

        sample["x_op"] = x[:, start : start + self.window_size]
        return sample

    def __repr__(self) -> str:
        return (
            f"WindowCrop(window_size={self.window_size}, "
            f"mode='{self.mode}', random={self.random})"
        )


# =========================================================================
# Gaussian Noise
# =========================================================================

class GaussianNoise:
    """Add Gaussian noise to the operational time series.

    The noise standard deviation is derived from a target signal-to-noise
    ratio (SNR) measured in decibels.  If ``sigma`` is given directly it
    takes precedence over *snr_db*.

    Parameters
    ----------
    snr_db : float
        Target signal-to-noise ratio in dB.
    sigma : float or None
        Explicit noise std; overrides *snr_db* when set.
    key : str
        Sample key to augment.
    """

    def __init__(
        self,
        snr_db: float = 20.0,
        sigma: Optional[float] = None,
        key: str = "x_op",
    ) -> None:
        self.snr_db = snr_db
        self.sigma = sigma
        self.key = key

    def __call__(self, sample: Sample) -> Sample:
        x = sample[self.key]
        if x is None:
            return sample

        if self.sigma is not None:
            noise_std = self.sigma
        else:
            # Compute noise std from target SNR
            signal_power = x.pow(2).mean()
            noise_power = signal_power / (10 ** (self.snr_db / 10))
            noise_std = noise_power.sqrt().item()

        noise = torch.randn_like(x) * noise_std
        sample[self.key] = x + noise
        return sample

    def __repr__(self) -> str:
        if self.sigma is not None:
            return f"GaussianNoise(sigma={self.sigma}, key='{self.key}')"
        return f"GaussianNoise(snr_db={self.snr_db}, key='{self.key}')"


# =========================================================================
# Time Warp
# =========================================================================

class TimeWarp:
    """Random time warping for the operational time series.

    A smooth, monotonically increasing warping path is generated by
    accumulating a perturbed uniform spacing.  The signal is then
    resampled along this path via linear interpolation.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian perturbation applied to the
        uniform time-step spacing (controls warp intensity).
    num_knots : int
        Number of control knots used to build the warp path.  More knots
        give higher-frequency distortions.
    key : str
        Sample key to warp (default ``"x_op"``).
    """

    def __init__(
        self,
        sigma: float = 0.2,
        num_knots: int = 4,
        key: str = "x_op",
    ) -> None:
        self.sigma = sigma
        self.num_knots = max(num_knots, 2)
        self.key = key

    def __call__(self, sample: Sample) -> Sample:
        x = sample[self.key]                              # [C, T]
        if x is None:
            return sample

        C, T = x.shape
        if T < 2:
            return sample

        # Build a monotonically-increasing warp path of length T.
        # 1. Create sparse knots with perturbed spacing.
        orig_knots = torch.linspace(0, 1, self.num_knots)
        warp_knots = orig_knots + torch.randn(self.num_knots) * self.sigma / self.num_knots
        # Ensure monotonicity and endpoints
        warp_knots = warp_knots.sort().values
        warp_knots[0] = 0.0
        warp_knots[-1] = 1.0

        # 2. Interpolate to T points to get the full warp path.
        orig_steps = torch.linspace(0, 1, T)
        warp_path = _interp_1d(orig_knots, warp_knots, orig_steps)  # [T]

        # 3. Scale warp_path from [0, 1] to index coordinates [0, T-1].
        warp_indices = warp_path * (T - 1)

        # 4. Resample each channel via linear interpolation.
        warped = _resample_linear(x, warp_indices)        # [C, T]
        sample[self.key] = warped
        return sample

    def __repr__(self) -> str:
        return (
            f"TimeWarp(sigma={self.sigma}, "
            f"num_knots={self.num_knots}, key='{self.key}')"
        )


def _interp_1d(
    xp: torch.Tensor,
    fp: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Piece-wise linear 1-D interpolation (mimics :func:`numpy.interp`).

    Parameters
    ----------
    xp : Tensor[K]   -- sorted x-coordinates of data points
    fp : Tensor[K]   -- y-coordinates of data points
    x  : Tensor[N]   -- query x-coordinates

    Returns
    -------
    Tensor[N] -- interpolated y-values
    """
    indices = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0 = xp[indices - 1]
    x1 = xp[indices]
    f0 = fp[indices - 1]
    f1 = fp[indices]
    slope = (f1 - f0) / (x1 - x0 + 1e-12)
    return f0 + slope * (x - x0)


def _resample_linear(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Resample ``x`` (shape ``[C, T]``) at fractional *indices* (shape ``[T]``)
    using linear interpolation along the time axis.
    """
    T = x.shape[-1]
    idx0 = indices.long().clamp(0, T - 2)
    idx1 = (idx0 + 1).clamp(0, T - 1)
    frac = (indices - idx0.float()).unsqueeze(0)          # [1, T]
    return x[:, idx0] * (1 - frac) + x[:, idx1] * frac


# =========================================================================
# Random Missing
# =========================================================================

class RandomMissing:
    """Randomly mask contiguous or scattered portions of the signal.

    Two strategies are supported:

    * ``"contiguous"`` -- masks a single contiguous block.
    * ``"scattered"`` -- independently masks each time step with
      probability *ratio*.

    Parameters
    ----------
    ratio : float
        Fraction of the time axis to mask (0, 1).
    strategy : {"contiguous", "scattered"}
        Masking strategy.
    fill : float
        Value used to replace masked positions.
    key : str
        Sample key to augment.
    """

    def __init__(
        self,
        ratio: float = 0.15,
        strategy: str = "contiguous",
        fill: float = 0.0,
        key: str = "x_op",
    ) -> None:
        assert 0.0 < ratio < 1.0, "ratio must be in (0, 1)"
        self.ratio = ratio
        self.strategy = strategy
        self.fill = fill
        self.key = key

    def __call__(self, sample: Sample) -> Sample:
        x = sample[self.key]
        if x is None:
            return sample

        C, T = x.shape
        mask_len = max(1, int(T * self.ratio))

        if self.strategy == "contiguous":
            start = torch.randint(0, max(T - mask_len, 1), (1,)).item()
            x = x.clone()
            x[:, start : start + mask_len] = self.fill
        elif self.strategy == "scattered":
            mask = torch.rand(T) < self.ratio
            x = x.clone()
            x[:, mask] = self.fill
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        sample[self.key] = x
        return sample

    def __repr__(self) -> str:
        return (
            f"RandomMissing(ratio={self.ratio}, "
            f"strategy='{self.strategy}', key='{self.key}')"
        )


# =========================================================================
# EIS Frequency Mask
# =========================================================================

class EISFreqMask:
    """Mask random frequency bands in the EIS spectrum.

    Zeros out a randomly selected contiguous band of *num_bins* frequency
    bins.  Applied only when EIS data is available (``modality_mask[1]==1``).

    Parameters
    ----------
    num_bins : int
        Width of each masked band.
    num_masks : int
        Number of independent masks to apply.
    """

    def __init__(self, num_bins: int = 5, num_masks: int = 1) -> None:
        self.num_bins = num_bins
        self.num_masks = num_masks

    def __call__(self, sample: Sample) -> Sample:
        x_eis = sample.get("x_eis")
        modality_mask = sample.get("modality_mask")

        # Skip if EIS not present
        if x_eis is None:
            return sample
        if modality_mask is not None and modality_mask[1].item() < 0.5:
            return sample

        C, F = x_eis.shape
        x_eis = x_eis.clone()

        for _ in range(self.num_masks):
            band_width = min(self.num_bins, F)
            start = torch.randint(0, max(F - band_width, 1), (1,)).item()
            x_eis[:, start : start + band_width] = 0.0

        sample["x_eis"] = x_eis
        return sample

    def __repr__(self) -> str:
        return (
            f"EISFreqMask(num_bins={self.num_bins}, "
            f"num_masks={self.num_masks})"
        )


# =========================================================================
# Modality Dropout
# =========================================================================

class ModalityDropout:
    """Randomly zero out an entire modality during training.

    With probability *p_op* the operational time series is zeroed; with
    probability *p_eis* the EIS spectrum is zeroed.  The corresponding
    entry in ``modality_mask`` is set to 0 so the model can adapt.

    Parameters
    ----------
    p_op : float
        Probability of dropping the operational modality.
    p_eis : float
        Probability of dropping the EIS modality.
    """

    def __init__(self, p_op: float = 0.0, p_eis: float = 0.15) -> None:
        self.p_op = p_op
        self.p_eis = p_eis

    def __call__(self, sample: Sample) -> Sample:
        modality_mask = sample["modality_mask"].clone()

        # Drop operational time series
        if self.p_op > 0.0 and torch.rand(1).item() < self.p_op:
            sample["x_op"] = torch.zeros_like(sample["x_op"])
            modality_mask[0] = 0.0

        # Drop EIS spectrum
        x_eis = sample.get("x_eis")
        if (
            x_eis is not None
            and modality_mask[1].item() > 0.5
            and self.p_eis > 0.0
            and torch.rand(1).item() < self.p_eis
        ):
            sample["x_eis"] = torch.zeros_like(x_eis)
            modality_mask[1] = 0.0

        sample["modality_mask"] = modality_mask
        return sample

    def __repr__(self) -> str:
        return f"ModalityDropout(p_op={self.p_op}, p_eis={self.p_eis})"
