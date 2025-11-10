"""
ラベル生成プリセット関数

よく使うラベル生成パターンをプリセット関数として提供します。
既存のLabelGeneratorクラスをラップして、より直感的なインターフェースを実現します。
"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd

from .enums import ThresholdMethod
from .main import LabelGenerator

logger = logging.getLogger(__name__)

# サポートする時間足の定義
SUPPORTED_TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d"]


def forward_classification_preset(
    df: pd.DataFrame,
    timeframe: str = "4h",
    horizon_n: int = 4,
    threshold: float = 0.002,
    price_column: str = "close",
    threshold_method: ThresholdMethod = ThresholdMethod.FIXED,
) -> pd.Series:
    """
    汎用的なforward分類ラベル生成のプリセット関数。

    N本先の価格変化率に基づいて、UP/RANGE/DOWNの3値分類ラベルを生成します。
    既存のLabelGeneratorクラスをラップして、より直感的に使用できるようにしています。

    Args:
        df: OHLCV データフレーム（close, open, high, low, volume カラムを含む）
        timeframe: 時間足 (例: "15m", "30m", "1h", "4h", "1d")
        horizon_n: N本先を見る（例: 4本先）
        threshold: 対称閾値（例: 0.002 = 0.2%）
            - FIXED メソッドの場合: この値がそのまま閾値として使用される
            - 他のメソッドの場合: 目安値として内部計算に使用される
        price_column: 価格カラム名（通常は "close"）
        threshold_method: 閾値計算方法
            - FIXED: 固定閾値（threshold値をそのまま使用）
            - STD_DEVIATION: 標準偏差ベース
            - QUANTILE: 分位数ベース
            - ADAPTIVE: 適応的閾値
            - DYNAMIC_VOLATILITY: 動的ボラティリティベース
            - KBINS_DISCRETIZER: KBinsDiscretizerベース（推奨）

    Returns:
        pd.Series: "UP" / "RANGE" / "DOWN" の3値ラベル（文字列）

    Raises:
        ValueError: 時間足が未サポート、データフレームが不正、必要なカラムが欠落している場合

    Example:
        >>> import pandas as pd
        >>> from app.utils.label_generation import forward_classification_preset
        >>>
        >>> # OHLCVデータを準備
        >>> df = pd.DataFrame({
        ...     'open': [100, 101, 102, 103, 104],
        ...     'high': [102, 103, 104, 105, 106],
        ...     'low': [99, 100, 101, 102, 103],
        ...     'close': [101, 102, 103, 104, 105],
        ...     'volume': [1000, 1100, 1200, 1300, 1400]
        ... })
        >>>
        >>> # 4h足、4本先、閾値0.2%でラベル生成
        >>> labels = forward_classification_preset(
        ...     df,
        ...     timeframe="4h",
        ...     horizon_n=4,
        ...     threshold=0.002
        ... )
        >>> print(labels.value_counts())
        RANGE    3
        UP       1
        DOWN     1
        dtype: int64

    Note:
        - 生成されるラベルは文字列型 ("UP", "RANGE", "DOWN") です
        - LabelGeneratorが返す数値ラベル (0, 1, 2) を文字列に変換しています
        - horizon_n本先を見るため、最後のN本はラベルが生成されません（NaN）
    """
    # 時間足の検証
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"未サポートの時間足です: {timeframe}. "
            f"サポートされている時間足: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )

    # データフレームの検証
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df は pandas.DataFrame である必要があります")

    if len(df) == 0:
        raise ValueError("空のデータフレームです")

    # 必要なカラムの検証
    if price_column not in df.columns:
        raise ValueError(
            f"指定された価格カラム '{price_column}' がデータフレームに存在しません。"
            f"利用可能なカラム: {', '.join(df.columns)}"
        )

    # horizon_nの検証
    if horizon_n < 1:
        raise ValueError(f"horizon_n は1以上である必要があります: {horizon_n}")

    if horizon_n >= len(df):
        raise ValueError(
            f"horizon_n ({horizon_n}) がデータ長 ({len(df)}) 以上です。"
            "十分なデータがありません。"
        )

    # ログ出力
    logger.info(
        f"ラベル生成開始: timeframe={timeframe}, horizon_n={horizon_n}, "
        f"threshold={threshold}, method={threshold_method.value}"
    )

    try:
        # LabelGeneratorを使用してラベル生成
        label_generator = LabelGenerator()

        # N本先の価格データを取得（forward参照）
        price_series = df[price_column].shift(-horizon_n)

        # ラベル生成（内部でforward変化率を計算）
        numeric_labels, threshold_info = label_generator.generate_labels(
            price_data=price_series,
            method=threshold_method,
            threshold=threshold,
        )

        # 数値ラベル (0, 1, 2) を文字列ラベル ("DOWN", "RANGE", "UP") に変換
        label_map = {0: "DOWN", 1: "RANGE", 2: "UP"}
        string_labels = numeric_labels.map(label_map)

        # ログ出力（分布情報）
        up_pct = threshold_info["up_ratio"] * 100
        range_pct = threshold_info["range_ratio"] * 100
        down_pct = threshold_info["down_ratio"] * 100
        logger.info(
            f"ラベル生成完了: "
            f"UP={threshold_info['up_count']}({up_pct:.1f}%), "
            f"RANGE={threshold_info['range_count']}({range_pct:.1f}%), "
            f"DOWN={threshold_info['down_count']}({down_pct:.1f}%)"
        )

        return string_labels

    except Exception as e:
        logger.error(f"ラベル生成エラー: {e}")
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
        >>> from app.utils.label_generation import get_common_presets
        >>>
        >>> presets = get_common_presets()
        >>> print(list(presets.keys()))
        ['15m_4bars', '30m_4bars', '1h_4bars', '4h_4bars', '1d_4bars', ...]
        >>>
        >>> # プリセットの詳細を確認
        >>> print(presets['4h_4bars'])
        {
            'timeframe': '4h',
            'horizon_n': 4,
            'threshold': 0.002,
            'threshold_method': ThresholdMethod.FIXED,
            'description': '4時間足、4本先（16時間先）、0.2%閾値'
        }
    """
    presets = {
        # 15分足プリセット
        "15m_4bars": {
            "timeframe": "15m",
            "horizon_n": 4,
            "threshold": 0.001,  # 0.1%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "15分足、4本先（1時間先）、0.1%閾値",
        },
        "15m_8bars": {
            "timeframe": "15m",
            "horizon_n": 8,
            "threshold": 0.0015,  # 0.15%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "15分足、8本先（2時間先）、0.15%閾値",
        },
        # 30分足プリセット
        "30m_4bars": {
            "timeframe": "30m",
            "horizon_n": 4,
            "threshold": 0.0015,  # 0.15%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "30分足、4本先（2時間先）、0.15%閾値",
        },
        "30m_8bars": {
            "timeframe": "30m",
            "horizon_n": 8,
            "threshold": 0.002,  # 0.2%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "30分足、8本先（4時間先）、0.2%閾値",
        },
        # 1時間足プリセット
        "1h_4bars": {
            "timeframe": "1h",
            "horizon_n": 4,
            "threshold": 0.002,  # 0.2%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "1時間足、4本先（4時間先）、0.2%閾値",
        },
        "1h_8bars": {
            "timeframe": "1h",
            "horizon_n": 8,
            "threshold": 0.003,  # 0.3%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "1時間足、8本先（8時間先）、0.3%閾値",
        },
        "1h_16bars": {
            "timeframe": "1h",
            "horizon_n": 16,
            "threshold": 0.004,  # 0.4%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "1時間足、16本先（16時間先）、0.4%閾値",
        },
        # 4時間足プリセット（デフォルト推奨）
        "4h_4bars": {
            "timeframe": "4h",
            "horizon_n": 4,
            "threshold": 0.002,  # 0.2%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "4時間足、4本先（16時間先）、0.2%閾値",
        },
        "4h_6bars": {
            "timeframe": "4h",
            "horizon_n": 6,
            "threshold": 0.003,  # 0.3%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "4時間足、6本先（24時間先）、0.3%閾値",
        },
        # 1日足プリセット
        "1d_4bars": {
            "timeframe": "1d",
            "horizon_n": 4,
            "threshold": 0.005,  # 0.5%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "1日足、4本先（4日先）、0.5%閾値",
        },
        "1d_7bars": {
            "timeframe": "1d",
            "horizon_n": 7,
            "threshold": 0.008,  # 0.8%
            "threshold_method": ThresholdMethod.FIXED,
            "description": "1日足、7本先（1週間先）、0.8%閾値",
        },
        # 動的閾値プリセット（推奨：データに応じて自動調整）
        "4h_4bars_dynamic": {
            "timeframe": "4h",
            "horizon_n": 4,
            "threshold": 0.002,  # 目安値
            "threshold_method": ThresholdMethod.KBINS_DISCRETIZER,
            "description": "4時間足、4本先、動的閾値（KBinsDiscretizer）",
        },
        "1h_4bars_dynamic": {
            "timeframe": "1h",
            "horizon_n": 4,
            "threshold": 0.002,  # 目安値
            "threshold_method": ThresholdMethod.KBINS_DISCRETIZER,
            "description": "1時間足、4本先、動的閾値（KBinsDiscretizer）",
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

    get_common_presets()で定義されたプリセットを名前で指定して、
    ラベル生成を実行します。

    Args:
        df: OHLCV データフレーム
        preset_name: プリセット名（例: "4h_4bars", "1h_4bars_dynamic"）
        price_column: 価格カラム名（デフォルト: "close"）

    Returns:
        Tuple[pd.Series, Dict]: ラベルSeriesとプリセット情報の辞書

    Raises:
        ValueError: 指定されたプリセット名が存在しない場合

    Example:
        >>> import pandas as pd
        >>> from app.utils.label_generation import apply_preset_by_name
        >>>
        >>> # OHLCVデータを準備
        >>> df = pd.DataFrame({
        ...     'open': [100, 101, 102, 103, 104, 105],
        ...     'high': [102, 103, 104, 105, 106, 107],
        ...     'low': [99, 100, 101, 102, 103, 104],
        ...     'close': [101, 102, 103, 104, 105, 106],
        ...     'volume': [1000, 1100, 1200, 1300, 1400, 1500]
        ... })
        >>>
        >>> # プリセット名でラベル生成
        >>> labels, preset_info = apply_preset_by_name(df, "4h_4bars")
        >>> print(f"使用したプリセット: {preset_info['description']}")
        使用したプリセット: 4時間足、4本先（16時間先）、0.2%閾値
        >>> print(labels.value_counts())
        RANGE    4
        UP       1
        DOWN     1
        dtype: int64
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

    # ラベル生成
    labels = forward_classification_preset(
        df=df, price_column=price_column, **preset_params
    )

    # プリセット情報を返却
    preset_info = {
        "preset_name": preset_name,
        "description": description,
        **preset_params,
    }

    return labels, preset_info
