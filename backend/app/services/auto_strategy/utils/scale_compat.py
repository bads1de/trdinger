"""
条件オペランド間のスケール互換性ユーティリティ

価格フィールド（close 等）と値域の桁が異なる指標（FUNDING_RATE_LEVEL、
SQUEEZE_PROBABILITY、ADX 等）を直接比較すると恒真/恒偽の退化条件になる。
本モジュールは「close と直接比較できる指標型」の判定と、スケール既定の
閾値取得を一元化する。
"""

from __future__ import annotations

from app.services.indicators.config import (
    IndicatorScaleType,
    indicator_registry,
)

# 価格フィールドのオペランド（volume は価格ではないため含めない）
PRICE_OPERANDS = frozenset({"close", "open", "high", "low"})

# bullish / bearish 判定に使う演算子
_BULLISH_OPERATORS = frozenset({">", ">=", "crosses_above"})

# close と同尺度で推移する価格オーバーレイ指標型のホワイトリスト。
# レジストリの category はカスタム指標で一律 "custom" になり、また
# FISHER・TSI・PPO 等のオシレーター系も PRICE_ABSOLUTE に分類されている
# ため、スケール型だけでは判定できず、意味的に価格帯で推移する型を
# 明示的に列挙する。
_CLOSE_COMPARABLE_TYPES = frozenset(
    {
        # 移動平均系
        "SMA",
        "EMA",
        "WMA",
        "HMA",
        "KAMA",
        "TRIMA",
        "DEMA",
        "TEMA",
        "T3",
        "ALMA",
        "VIDYA",
        "JMA",
        "RMA",
        "ZLEMA",
        "ZLMA",
        "SINWMA",
        "PWMA",
        "SWMA",
        "FWMA",
        "VWMA",
        "SSF",
        "MAMA",
        "FRAMA",
        "ERI",
        "EHLERS_INSTANTANEOUS_TRENDLINE",
        "HWMA",
        "MAVP",
        "MCGD",
        "RAINBOW",
        "PMAX",
        "LINREG",
        "TSF",
        "LINREGINTERCEPT",
        # 価格変換系
        "AVGPRICE",
        "MEDPRICE",
        "TYPPRICE",
        "HL2",
        "HLC3",
        "OHLC4",
        "WCP",
        "MIDPRICE",
        "MEDIAN",
        "MIDPOINT",
        "HA",
        # 価格帯で推移するトレンド系オーバーレイ
        "PSAR",
        "SAR",
        "SUPERTREND",
        "VWAP",
        # バンド系（参照名のデフォルト出力は中央線）
        "BBANDS",
    }
)


def is_close_comparable_type(indicator_type: str) -> bool:
    """指標が close と直接比較できる価格スケールかを判定する。"""
    normalized = indicator_type.upper()
    if normalized in _CLOSE_COMPARABLE_TYPES:
        return True

    # 正しいカテゴリが付与されている指標はスケール定義から判定する
    cfg = indicator_registry.get_indicator_config(normalized)
    if cfg is None:
        return False
    return (
        getattr(cfg, "category", None) in ("trend", "overlap")
        and cfg.scale_type == IndicatorScaleType.PRICE_ABSOLUTE
        and not cfg.thresholds
    )


def is_price_operand(operand: object) -> bool:
    """オペランドが価格フィールド（close/open/high/low）かを判定する。"""
    return isinstance(operand, str) and operand.lower() in PRICE_OPERANDS


def default_threshold_for(indicator_type: str, *, bullish: bool) -> float:
    """
    スケール既定の比較閾値を返す。

    レジストリに閾値プロファイルが定義されている場合はその normal
    プロファイル値（long_gt / short_lt）を優先し、無い場合はスケール型に
    応じた中心値（0-100 オシレーターは 50、比率系は 1.0、その他は 0）を
    返す。bullish には比較演算子が上向き（>）かを渡す。
    """
    cfg = indicator_registry.get_indicator_config(indicator_type.upper())
    if cfg is not None and cfg.thresholds:
        profile = cfg.thresholds.get("normal") or {}
        value = profile.get("long_gt") if bullish else profile.get("short_lt")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    scale_type = cfg.scale_type if cfg is not None else None
    if scale_type == IndicatorScaleType.OSCILLATOR_0_100:
        return 50.0
    if scale_type == IndicatorScaleType.PRICE_RATIO:
        return 1.0
    return 0.0


def is_bullish_operator(operator: str) -> bool:
    """比較演算子が上向き（> 系）かを判定する。"""
    return operator in _BULLISH_OPERATORS
