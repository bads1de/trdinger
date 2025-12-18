"""
ラベル生成プリセット関数

よく使うラベル生成パターンをプリセット関数として提供します。
既存のLabelGeneratorクラスをラップして、より直感的なインターフェースを実現します。
"""

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..common.volatility_utils import calculate_volatility_atr, calculate_volatility_std
from .trend_scanning import TrendScanning
from .triple_barrier import TripleBarrier

logger = logging.getLogger(__name__)

# サポートする時間足の定義
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


def _log_distribution(name: str, labels: pd.Series) -> None:
    """ラベルの分布をログ出力"""
    counts = labels.value_counts()
    total = len(labels.dropna())
    if total > 0:
        v_pct, i_pct = (counts.get(1, 0) / total) * 100, (counts.get(0, 0) / total) * 100
        logger.info(f"{name}完了: Valid={counts.get(1, 0)}({v_pct:.1f}%), Invalid={counts.get(0, 0)}({i_pct:.1f}%)")


def triple_barrier_method_preset(
    df: pd.DataFrame,
    timeframe: str = "4h",
    horizon_n: int = 4,
    pt: float = 1.0,
    sl: float = 1.0,
    min_ret: float = 0.001,
    price_column: str = "close",
    volatility_window: int = 20,
    use_atr: bool = False,
    atr_period: int = 14,
    t_events: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """TBM ラベル生成プリセット"""
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    logger.info(f"TBMラベル生成開始: {timeframe}, horizon={horizon_n}")
    try:
        close = df[price_column]
        if use_atr:
            vol = calculate_volatility_atr(df["high"], df["low"], close, atr_period, True)
        else:
            vol = calculate_volatility_std(close.pct_change(), volatility_window)

        t_ev = t_events if t_events is not None else close.index
        v_bar = pd.Series(close.index, index=close.index).shift(-horizon_n)

        tb = TripleBarrier(pt=pt, sl=sl, min_ret=min_ret)
        events = tb.get_events(close, t_ev, [pt, sl], vol, min_ret, v_bar)
        labels = tb.get_bins(events, close, binary_label=True)["bin"]

        _log_distribution("TBMラベル生成", labels)
        return labels
    except Exception as e:
        logger.error(f"TBM error: {e}")
        raise


def trend_scanning_preset(
    df: pd.DataFrame,
    timeframe: str = "4h",
    horizon_n: int = 100,
    threshold: float = 5.0,
    min_window: int = 10,
    window_step: int = 1,
    price_column: str = "close",
    t_events: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """Trend Scanning ラベル生成プリセット"""
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    logger.info(f"TSラベル生成開始: {timeframe}, threshold={threshold}")
    try:
        close = df[price_column]
        ts = TrendScanning(min_window, horizon_n, window_step, threshold)
        labels = ts.get_labels(close, t_events)["bin"].abs().astype(int)

        _log_distribution("TSラベル生成", labels)
        return labels
    except Exception as e:
        logger.error(f"TS error: {e}")
        raise


def get_common_presets() -> Dict[str, Dict[str, Any]]:
    """
    よく使うラベル生成パラメータのプリセットを取得。

    各プリセットには、forward_classification_preset関数に渡すための
    パラメータセットが含まれています。

    Returns:
        Dict[str, Dict]: プリセット名をキーとする辞書
            各値は forward_classification_preset に渡せるパラメータ辞書

    Example:
        >>> from app.services.ml.label_generation import get_common_presets
        >>> presets = get_common_presets()
        >>> print(presets['tbm_4h_1.0_1.0'])
    """
    presets = {
        # TBMプリセット
        "tbm_4h_1.0_1.0": {
            "timeframe": "4h",
            "horizon_n": 12,  # 48時間
            "pt": 1.0,
            "sl": 1.0,
            "min_ret": 0.001,
            "description": "TBM (4h): PT=1.0σ, SL=1.0σ, Horizon=12bars",
        },
        "tbm_4h_0.5_1.0": {
            "timeframe": "4h",
            "horizon_n": 12,  # 48時間
            "pt": 0.5,
            "sl": 1.0,
            "min_ret": 0.001,
            "description": "TBM (4h): PT=0.5σ, SL=1.0σ, Horizon=12bars",
        },
        "tbm_4h_2.0_2.0": {
            "timeframe": "4h",
            "horizon_n": 12,
            "pt": 2.0,
            "sl": 2.0,
            "min_ret": 0.001,
            "description": "TBM (4h): PT=2.0σ, SL=2.0σ, Horizon=12bars",
        },
        "tbm_15m_1.0_1.0": {
            "timeframe": "15m",
            "horizon_n": 24,  # 6時間
            "pt": 1.0,
            "sl": 1.0,
            "min_ret": 0.0005,
            "description": "TBM (15m): PT=1.0σ, SL=1.0σ, Horizon=24bars",
        },
        # Trend Scanning プリセット
        "trend_scanning_strong": {
            "timeframe": "4h",
            "horizon_n": 100,  # 約2.5週間 (400時間)
            "min_window": 20,
            "window_step": 2,
            "threshold": 5.0,  # 強いトレンドのみ (t-value >= 5.0)
            "threshold_method": "TREND_SCANNING",
            "description": "Trend Scanning: Strong Trends (High T-value, Long Horizon)",
        },
        "trend_scanning_medium": {
            "timeframe": "4h",
            "horizon_n": 48,  # 約1週間
            "min_window": 12,
            "window_step": 1,
            "threshold": 3.0,  # 中程度のトレンド
            "threshold_method": "TREND_SCANNING",
            "description": "Trend Scanning: Medium Trends (Moderate T-value)",
        },
        # Trend Scanning 1h Preset
        "trend_scanning_1h": {
            "timeframe": "1h",
            "horizon_n": 72,  # Max 3 days
            "min_window": 12,  # Min 12 hours
            "window_step": 1,
            "threshold": 3.0,
            "threshold_method": "TREND_SCANNING",
            "description": "Trend Scanning (1h): Day Trading (Max 3 days)",
        },
        # Trend Scanning 15m Preset
        "trend_scanning_15m": {
            "timeframe": "15m",
            "horizon_n": 96,  # Max 24 hours
            "min_window": 30,  # Min 7.5 hours
            "window_step": 2,
            "threshold": 2.5,
            "threshold_method": "TREND_SCANNING",
            "description": "Trend Scanning (15m): Short-term (Max 24 hours)",
        },
    }

    return presets


def apply_preset_by_name(
    df: pd.DataFrame,
    preset_name: str,
    price_column: str = "close",
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    プリセット名でラベル生成を実行。

    Args:
        df: OHLCV データフレーム
        preset_name: プリセット名
        price_column: 価格カラム名

    Returns:
        Tuple[pd.Series, Dict]: ラベルSeriesとプリセット情報の辞書
    """
    # プリセットを取得
    presets = get_common_presets()

    if preset_name not in presets:
        available_presets = ", ".join(sorted(presets.keys()))
        raise ValueError(
            f"プリセット '{preset_name}' が見つかりません。"
            f"利用可能なプリセット: {available_presets}"
        )

    preset_params = presets[preset_name].copy()
    description = preset_params.pop("description")

    logger.info(f"プリセット '{preset_name}' を使用: {description}")

    # threshold_method を確認
    method_str = preset_params.get("threshold_method", "TRIPLE_BARRIER")
    # キーから削除しておく（引数として渡さないため、必要ならここでpop）
    preset_params.pop("threshold_method", None)

    if method_str == "TREND_SCANNING":
        labels = trend_scanning_preset(
            df=df, price_column=price_column, **preset_params
        )
    else:
        # Triple Barrier Method (デフォルト)
        labels = triple_barrier_method_preset(
            df=df, price_column=price_column, **preset_params
        )

    # プリセット情報を返却
    preset_info = {
        "preset_name": preset_name,
        "description": description,
        **preset_params,
    }

    return labels, preset_info
