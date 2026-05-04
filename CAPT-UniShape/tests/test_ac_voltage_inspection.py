from pathlib import Path

import pytest

from scripts.inspect_ac_voltage_dataset import (
    infer_ac_voltage_metadata,
    summarize_processed_csv,
)


def test_infers_domain_and_fault_label_from_processed_csv_name() -> None:
    meta = infer_ac_voltage_metadata(
        Path("Processed_Data/New_MEA_Starvation_Voltage_Response.csv")
    )

    assert meta.domain == "new_mea"
    assert meta.label_name == "starvation"
    assert meta.label_id == 2


def test_infers_old_mea_from_original_dta_directory() -> None:
    meta = infer_ac_voltage_metadata(
        Path("Original MEA Data/Old_MEA_drying/Normal_PWRDISCHARGEPROFILE_drying_#1.DTA")
    )

    assert meta.domain == "old_mea"
    assert meta.label_name == "drying"
    assert meta.label_id == 1


def test_summarize_processed_csv_counts_rows_columns_and_numeric_quality(tmp_path: Path) -> None:
    csv_path = tmp_path / "Normal_Voltage_Response.csv"
    csv_path.write_text(
        "0.1,0.2,1\n"
        "0.2,0.4,1\n",
        encoding="utf-8",
    )

    summary = summarize_processed_csv(csv_path, sample_rows=10)

    assert summary["row_count"] == 2
    assert summary["column_count"] == 3
    assert summary["curve_column_count"] == 2
    assert summary["trailing_label_values"] == [1]
    assert summary["sampled_rows"] == 2
    assert summary["finite_ratio"] == pytest.approx(1.0)
    assert summary["value_min"] == pytest.approx(0.1)
    assert summary["value_max"] == pytest.approx(0.4)
