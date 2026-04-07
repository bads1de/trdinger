import pandas as pd

from app.services.auto_strategy.core.evaluation.time_alignment import (
    align_timestamp_to_index,
    align_timestamp_to_reference,
)


def test_align_timestamp_to_index_localizes_naive_timestamp_to_aware_index():
    index = pd.date_range("2024-01-01 00:00:00", periods=2, freq="h", tz="UTC")
    aligned = align_timestamp_to_index("2024-01-01 01:00:00", index)

    assert aligned == pd.Timestamp("2024-01-01 01:00:00+00:00")


def test_align_timestamp_to_index_strips_timezone_for_naive_index():
    index = pd.date_range("2024-01-01 00:00:00", periods=2, freq="h")
    aligned = align_timestamp_to_index(
        pd.Timestamp("2024-01-01 01:00:00", tz="UTC"),
        index,
    )

    assert aligned == pd.Timestamp("2024-01-01 01:00:00")
    assert aligned.tzinfo is None


def test_align_timestamp_to_reference_uses_reference_timezone():
    reference = pd.Timestamp("2024-01-01 01:00:00", tz="Asia/Tokyo")
    aligned = align_timestamp_to_reference("2024-01-01 01:00:00", reference)

    assert aligned == pd.Timestamp("2024-01-01 01:00:00", tz="Asia/Tokyo")
