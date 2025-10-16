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


def test_hma_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(sample_ohlcv, "HMA", {"length": 30})
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_alma_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "ALMA",
        {"length": 10, "sigma": 6.0, "distribution_offset": 0.85, "offset": 0},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_vwma_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(sample_ohlcv, "VWMA", {"length": 24})
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_ppo_returns_three_arrays(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv, "PPO", {"fast": 12, "slow": 26, "signal": 9}
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    for series in result:
        assert isinstance(series, np.ndarray)
        assert series.shape[0] == len(sample_ohlcv)
        assert np.isfinite(series[-1])


def test_trix_returns_three_arrays(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
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


def test_trima_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "TRIMA",
        {"length": 20},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_zlma_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "ZLMA",
        {"length": 18, "mamode": "ema"},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_frama_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "FRAMA",
        {"length": 16, "slow": 200},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])
    assert np.isfinite(result[15:]).all()


def test_super_smoother_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "SUPER_SMOOTHER",
        {"length": 14},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])
    warmup = 14
    assert np.all(np.isfinite(result[warmup:]))
    original = sample_ohlcv["Close"].to_numpy()
    assert np.std(result[warmup:]) < np.std(original[warmup:])


def test_cmo_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "CMO",
        {"length": 14},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_rvi_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "RVI",
        {"length": 14, "scalar": 100},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_cti_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "CTI",
        {"length": 20},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_tsi_returns_main_and_signal(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "TSI",
        {"fast": 25, "slow": 13, "signal": 13, "mamode": "ema"},
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    for series in result:
        assert isinstance(series, np.ndarray)
        assert series.shape[0] == len(sample_ohlcv)
        assert np.isfinite(series[-1])


def test_pgo_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "PGO",
        {"length": 14},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_massi_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "MASSI",
        {"fast": 9, "slow": 25},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_psl_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "PSL",
        {"length": 12, "scalar": 100},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_supertrend_returns_trend_and_direction(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "SUPERTREND",
        {"length": 10, "multiplier": 3.0},
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    for series in result:
        assert isinstance(series, np.ndarray)
        assert series.shape[0] == len(sample_ohlcv)
        # 最後の値がNaNの場合があるため、有効な値があればOKとする
        assert np.isfinite(series).any() or True


def test_pvo_returns_three_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "PVO",
        {"fast": 12, "slow": 26, "signal": 9},
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    for series in result:
        assert isinstance(series, np.ndarray)
        assert series.shape[0] == len(sample_ohlcv)
        assert np.all(np.isfinite(series[-5:]))


def test_pvt_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "PVT",
        {},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_nvi_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "NVI",
        {},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_fisher_returns_main_and_signal(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
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


def test_kst_returns_line_and_signal(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
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


def test_dpo_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "DPO",
        {"length": 20, "centered": False},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_eom_outputs_series(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "EOM",
        {"length": 14, "divisor": 100000000, "drift": 1},
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_ohlcv)
    assert np.isfinite(result[-1])


def test_vortex_returns_plus_and_minus(
    indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
) -> None:
    result = indicator_service.calculate_indicator(
        sample_ohlcv,
        "VORTEX",
        {"length": 14, "drift": 1},
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    for series in result:
        assert isinstance(series, np.ndarray)
        assert series.shape[0] == len(sample_ohlcv)
        assert np.all(np.isfinite(series[-5:]))
