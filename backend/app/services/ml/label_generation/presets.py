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
    binary_label: bool = False,
    t_events: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Triple Barrier Method (TBM) ラベル生成プリセット。

    Args:
        df: OHLCV データフレーム
        timeframe: 時間足
        horizon_n: 垂直バリア（時間切れ）までのバー数
        pt: 利食い（Profit Taking）乗数 (ボラティリティ * pt)
        sl: 損切り（Stop Loss）乗数 (ボラティリティ * sl)
        min_ret: 最小リターン閾値
        price_column: 価格カラム名
        volatility_window: ボラティリティ計算ウィンドウ（use_atr=Falseの場合）
        use_atr: ATRをボラティリティとして使用するか
        atr_period: ATR計算期間
        binary_label: 0/1のバイナリラベルを生成するか（メタラベリング用）
        t_events: イベント時刻（指定された場合、この時刻のみ計算）

    Returns:
        pd.Series: "UP"/"RANGE"/"DOWN" (binary_label=False) または 0/1 (binary_label=True)
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"未サポートの時間足です: {timeframe}")

    logger.info(
        f"TBMラベル生成開始: timeframe={timeframe}, horizon={horizon_n}, "
        f"pt={pt}, sl={sl}, min_ret={min_ret}, use_atr={use_atr}"
    )

    try:
        # 1. 準備
        close = df[price_column]

        # ボラティリティ計算
        if use_atr:
            # ATR計算 (共通ユーティリティを使用)
            volatility = calculate_volatility_atr(
                high=df["high"],
                low=df["low"],
                close=close,
                window=atr_period,
                as_percentage=True,
            )

        else:
            # 従来: リターンの標準偏差 (共通ユーティリティを使用)
            returns = close.pct_change()
            volatility = calculate_volatility_std(
                returns=returns, window=volatility_window
            )

        # t_events: 指定がなければ全てのバーをイベント候補とする
        if t_events is None:
            t_events = close.index

        # vertical_barrier_times: horizon_n本後の時刻
        vertical_barrier_times = pd.Series(close.index, index=close.index).shift(
            -horizon_n
        )

        # 2. Triple Barrier 実行
        tb = TripleBarrier(pt=pt, sl=sl, min_ret=min_ret, num_threads=1)

        events = tb.get_events(
            close=close,
            t_events=t_events,
            pt_sl=[pt, sl],
            target=volatility,
            min_ret=min_ret,
            vertical_barrier_times=vertical_barrier_times,
        )

        # 3. ラベル生成 (get_bins)
        bins = tb.get_bins(events, close, binary_label=binary_label)

        # 4. ラベル変換
        if binary_label:
            return bins["bin"]
        else:
            # (-1, 0, 1 -> DOWN, RANGE, UP)
            label_map = {-1.0: "DOWN", 0.0: "RANGE", 1.0: "UP"}
            string_labels = bins["bin"].map(label_map)

            # 元のインデックスに合わせる
            string_labels = string_labels.reindex(df.index)

            # 分布ログ
            counts = string_labels.value_counts()
            total = len(string_labels.dropna())
            if total > 0:
                up_pct = (counts.get("UP", 0) / total) * 100
                range_pct = (counts.get("RANGE", 0) / total) * 100
                down_pct = (counts.get("DOWN", 0) / total) * 100
                logger.info(
                    f"TBMラベル生成完了: "
                    f"UP={counts.get('UP', 0)}({up_pct:.1f}%), "
                    f"RANGE={counts.get('RANGE', 0)}({range_pct:.1f}%), "
                    f"DOWN={counts.get('DOWN', 0)}({down_pct:.1f}%)"
                )

            return string_labels

    except Exception as e:
        logger.error(f"TBMラベル生成エラー: {e}")
        raise


def trend_scanning_preset(
    df: pd.DataFrame,
    timeframe: str = "4h",
    horizon_n: int = 100,
    threshold: float = 5.0,
    min_window: int = 10,
    window_step: int = 1,
    price_column: str = "close",
    binary_label: bool = False,
    t_events: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Trend Scanning ラベル生成プリセット。

    Args:
        df: OHLCV データフレーム
        timeframe: 時間足
        horizon_n: 最大ウィンドウサイズ (max_window)
        threshold: t値の閾値 (min_t_value)
        min_window: 最小ウィンドウサイズ
        window_step: ウィンドウステップサイズ
        price_column: 価格カラム名
        binary_label: 0/1のバイナリラベルを生成するか
        t_events: イベント時刻（指定された場合、この時刻のみ計算）

    Returns:
        pd.Series: "UP"/"RANGE"/"DOWN" (binary_label=False) または 0/1 (binary_label=True)
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"未サポートの時間足です: {timeframe}")

    logger.info(
        f"TrendScanningラベル生成開始: timeframe={timeframe}, horizon(max)={horizon_n}, "
        f"threshold(t)={threshold}, min_window={min_window}"
    )

    try:
        # 1. 準備
        close = df[price_column]
        if t_events is None:
            t_events = close.index  # 全ての足を対象

        # 2. Trend Scanning 実行
        ts = TrendScanning(
            min_window=min_window,
            max_window=horizon_n,
            step=window_step,
            min_t_value=threshold,
        )

        labels_df = ts.get_labels(close=close, t_events=t_events)

        # 3. ラベル変換
        if binary_label:
            # 絶対値をとってバイナリ化 (0 or 1)
            return labels_df["bin"].abs().astype(int)
        else:
            # (-1, 0, 1 -> DOWN, RANGE, UP)
            label_map = {-1.0: "DOWN", 0.0: "RANGE", 1.0: "UP"}
            string_labels = labels_df["bin"].map(label_map)

            # 元のインデックスに合わせる
            string_labels = string_labels.reindex(df.index)

            # 分布ログ
            counts = string_labels.value_counts()
            total = len(string_labels.dropna())
            if total > 0:
                up_pct = (counts.get("UP", 0) / total) * 100
                range_pct = (counts.get("RANGE", 0) / total) * 100
                down_pct = (counts.get("DOWN", 0) / total) * 100
                logger.info(
                    f"TSラベル生成完了: "
                    f"UP={counts.get('UP', 0)}({up_pct:.1f}%), "
                    f"RANGE={counts.get('RANGE', 0)}({range_pct:.1f}%), "
                    f"DOWN={counts.get('DOWN', 0)}({down_pct:.1f}%)"
                )

            return string_labels

    except Exception as e:
        logger.error(f"TrendScanningラベル生成エラー: {e}")
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
