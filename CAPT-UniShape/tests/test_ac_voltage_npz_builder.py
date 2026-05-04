from pathlib import Path

import numpy as np

from scripts.build_ac_voltage_npz import (
    build_condition_features,
    build_eis_like_sequence,
    build_split_array,
    resample_curve,
)


def test_resample_curve_maps_curve_to_requested_length() -> None:
    curve = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    out = resample_curve(curve, 3)

    assert out.dtype == np.float32
    assert np.allclose(out, np.array([0.0, 1.5, 3.0], dtype=np.float32))


def test_condition_features_are_fixed_width_and_capture_basic_shape() -> None:
    curves = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )

    features = build_condition_features(curves)

    assert features.shape == (2, 12)
    assert np.isclose(features[0, 0], 2.5)
    assert np.isclose(features[0, 8], 4.0)
    assert np.isclose(features[0, 9], 3.0)
    assert np.isclose(features[1, 9], 0.0)


def test_eis_like_sequence_has_curve_diff_cumulative_and_axis() -> None:
    curves = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    seq = build_eis_like_sequence(curves, seq_len=5)

    assert seq.shape == (1, 4, 5)
    assert np.allclose(seq[0, 3], np.linspace(0.0, 1.0, 5, dtype=np.float32))


def test_old_to_new_split_uses_old_for_train_val_and_new_for_test() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2], dtype=np.int64)
    domains = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int64)

    split = build_split_array(labels, domains, protocol="old_to_new", val_fraction=0.5, seed=7)

    assert set(split[:6].tolist()) == {0, 1}
    assert np.all(split[6:] == 2)
    assert set(labels[split == 1].tolist()) == {0, 1, 2}
