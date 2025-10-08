import numpy as np
import pandas as pd
import pytest

from app.services.indicators import TechnicalIndicatorService


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    periods = 200
    index = pd.date_range("2022-01-01", periods=periods, freq="H")
    base = np.linspace(100, 200, periods)
    noise = np.sin(np.linspace(0, 8 * np.pi, periods))
    close = base + noise

    df = pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.linspace(1000, 2000, periods),
        },
        index=index,
    )
    return df


@pytest.fixture
def indicator_service() -> TechnicalIndicatorService:
    return TechnicalIndicatorService()


def test_hma_outputs_series(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv, "HMA", {"length": 30}
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_alma_outputs_series(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "ALMA",
        {"length": 10, "sigma": 6.0, "distribution_offset": 0.85, "offset": 0},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_vwma_outputs_series(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv, "VWMA", {"length": 24}
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_ppo_returns_three_arrays(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv, "PPO", {"fast": 12, "slow": 26, "signal": 9}
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    for series in result:
        assert isinstance(series, np.ndarray)
        assert series.shape[0] == len(sample_ohlcv)
        assert np.isfinite(series[-1])


def test_trix_returns_three_arrays(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv, "TRIX", {"length": 15, "signal": 9}
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    for series in result:
        assert isinstance(series, np.ndarray)
        assert series.shape[0] == len(sample_ohlcv)
        assert np.isfinite(series[-1])


def test_ultimate_oscillator_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "UO",
        {"fast": 7, "medium": 14, "slow": 28},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_trima_outputs_series(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "TRIMA",
        {"length": 20},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_zlma_outputs_series(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "ZLMA",
        {"length": 18, "mamode": "ema"},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_cmo_outputs_series(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "CMO",
        {"length": 14},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_fisher_returns_main_and_signal(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "FISHER",
        {"length": 9, "signal": 3},
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    for series in result:
        assert isinstance(series, np.ndarray)
        assert series.shape[0] == len(sample_ohlcv)
        assert np.isfinite(series[-1])


def test_kst_returns_line_and_signal(indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "KST",
        {"roc1": 8, "roc2": 12, "roc3": 20, "roc4": 28, "signal": 9},
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    for series in result:
        assert isinstance(series, np.ndarray)
        assert series.shape[0] == len(sample_ohlcv)
        assert np.isfinite(series[-1])
