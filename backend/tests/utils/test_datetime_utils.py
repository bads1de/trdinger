from datetime import datetime, timezone

import pandas as pd

from app.utils.datetime_utils import (
    current_datetime_like,
    normalize_datetimes_for_comparison,
    parse_datetime_optional,
    parse_datetime_value,
    parse_timestamp_safe,
)


class TestDatetimeUtils:
    def test_parse_datetime_value_supports_z_suffix(self):
        result = parse_datetime_value("2024-01-01T00:00:00Z")
        assert result == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

    def test_parse_datetime_value_supports_timestamp(self):
        timestamp = pd.Timestamp("2024-01-01T09:00:00+09:00")

        result = parse_datetime_value(timestamp)

        assert result == timestamp.to_pydatetime()

    def test_parse_datetime_optional_returns_none_for_invalid_values(self):
        assert parse_datetime_optional("") is None
        assert parse_datetime_optional("invalid-date") is None
        assert parse_datetime_optional(pd.NaT) is None

    def test_parse_timestamp_safe_supports_milliseconds_and_invalid_values(self):
        expected = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

        assert parse_timestamp_safe(1704067200000) == expected
        assert parse_timestamp_safe("2024-01-01T00:00:00Z") == expected
        assert parse_timestamp_safe(float("nan")) is None
        assert parse_timestamp_safe(pd.NaT) is None
        assert parse_timestamp_safe(None) is None
        assert parse_timestamp_safe("invalid-date") is None

    def test_normalize_datetimes_for_comparison_handles_aware_and_naive_values(self):
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

        normalized_start, normalized_end = normalize_datetimes_for_comparison(
            start_date,
            end_date,
        )

        assert normalized_start.tzinfo is not None
        assert normalized_end.tzinfo is not None
        assert normalized_start < normalized_end

    def test_current_datetime_like_matches_reference_timezone(self):
        reference = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

        result = current_datetime_like(reference)

        assert result.tzinfo == timezone.utc
