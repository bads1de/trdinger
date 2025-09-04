"""
インジケーター設定管理クラス

JSON形式でのインジケーター設定を管理し、
パラメータ埋め込み文字列からの移行をサポートします。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class IndicatorScaleType(Enum):
    """指標のスケールタイプ"""

    OSCILLATOR_0_100 = "oscillator_0_100"  # 0-100スケール（RSI, STOCH等）
    OSCILLATOR_0_1 = "oscillator_0_1"  # 0-1スケール（ML予測確率等）
    OSCILLATOR_PLUS_MINUS_100 = "oscillator_plus_minus_100"  # ±100スケール（CCI等）
    MOMENTUM_ZERO_CENTERED = "momentum_zero_centered"  # ゼロ近辺変動（TRIX, PPO等）
    PRICE_RATIO = "price_ratio"  # 価格比率（SMA, EMA等）
    FUNDING_RATE = "funding_rate"  # ファンディングレート
    OPEN_INTEREST = "open_interest"  # オープンインタレスト
    VOLUME = "volume"  # 出来高
    PRICE_ABSOLUTE = "price_absolute"  # 絶対価格
    PATTERN_BINARY = "pattern_binary"  # パターン認識バイナリ（0/1または-1/1）


logger = logging.getLogger(__name__)


class IndicatorResultType(Enum):
    """インジケーター結果タイプ"""

    SINGLE = "single"  # 単一値（例：RSI、SMA）
    COMPLEX = "complex"  # 複数値（例：MACD、Bollinger Bands）


@dataclass
class ParameterConfig:
    """パラメータ設定"""

    name: str  # パラメータ名（例：period, fast_period）
    default_value: Union[int, float, str, bool]  # デフォルト値
    min_value: Optional[Union[int, float, str, bool]] = None  # 最小値
    max_value: Optional[Union[int, float, str, bool]] = None  # 最大値
    description: Optional[str] = None  # パラメータの説明

    def __init__(
        self,
        name: str,
        default_value: Union[int, float, str, bool],
        min_value: Optional[Union[int, float, str, bool]] = None,
        max_value: Optional[Union[int, float, str, bool]] = None,
        description: Optional[str] = None,
    ) -> None:
        self.name = name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.description = description

    def validate_value(self, value: Union[int, float, str, bool]) -> bool:
        """値の妥当性を検証"""
        # 数値型でない場合は検証をスキップ
        if not isinstance(value, (int, float)):
            return True

        if (
            self.min_value is not None
            and isinstance(self.min_value, (int, float))
            and value < self.min_value
        ):
            return False
        if (
            self.max_value is not None
            and isinstance(self.max_value, (int, float))
            and value > self.max_value
        ):
            return False
        return True


@dataclass
class IndicatorConfig:
    """インジケーター設定のメタ情報を保持するシンプルなデータクラス

    最小限の属性とユーティリティを提供し、他モジュールから参照される想定の API を満たします。
    """

    indicator_name: str
    adapter_function: Optional[Any] = None
    required_data: List[str] = field(default_factory=list)
    result_type: IndicatorResultType = IndicatorResultType.SINGLE
    scale_type: IndicatorScaleType = IndicatorScaleType.PRICE_RATIO
    category: Optional[str] = None
    output_names: Optional[List[str]] = None
    default_output: Optional[str] = None
    aliases: Optional[List[str]] = None
    param_map: Dict[str, Optional[str]] = field(default_factory=dict)
    # 後方互換性のための追加
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """後処理でパラメータを構築"""
        from .indicator_definitions import PANDAS_TA_CONFIG

        self.parameters = self._build_parameters_from_config(PANDAS_TA_CONFIG)

    def _build_parameters_from_config(self, pandas_ta_config) -> Dict[str, Any]:
        """pandas-ta設定からパラメータを構築"""
        if self.indicator_name in pandas_ta_config:
            config = pandas_ta_config[self.indicator_name]
            params = {}
            for param_name, aliases in config["params"].items():
                default_value = config["default_values"].get(param_name, 14)
                # ParameterConfig を作成
                param_config = ParameterConfig(
                    name=aliases[0],  # メインエイリアス
                    default_value=default_value,
                    min_value=2 if any(word in aliases[0] for word in ["length", "period"]) else None,
                    max_value=200 if any(word in aliases[0] for word in ["length", "period"]) else None,
                )
                params[aliases[0]] = param_config
            return params
        else:
            # デフォルトパラメータ
            return {
                "period": ParameterConfig(
                    name="period",
                    default_value=14,
                    min_value=2,
                    max_value=200
                )
            }

    def add_parameter(self, param: ParameterConfig) -> None:
        """互換性のためのダミー実装: パラメータを param_map に追加しないが呼び出しを許可する。"""
        # 実際の実装ではパラメータリストを保持するが、ここでは最小限にとどめる
        return None

    def generate_json_name(self) -> str:
        """JSON用の名称を生成（現状は indicator_name を返す）"""
        return self.indicator_name

    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """パラメータ範囲を取得"""
        from .indicator_definitions import PANDAS_TA_CONFIG

        ranges = {}
        if self.indicator_name in PANDAS_TA_CONFIG:
            config = PANDAS_TA_CONFIG[self.indicator_name]
            for param_name, aliases in config["params"].items():
                default_value = config["default_values"].get(param_name, 14)
                ranges[aliases[0]] = {
                    "min": 2,
                    "max": 200 if any(word in aliases[0] for word in ["length", "period"]) else 100,
                    "default": default_value
                }
        else:
            # デフォルト: periodパラメータ
            ranges["period"] = {"min": 2, "max": 200, "default": 14}

        return ranges

    def normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータ正規化"""
        from .indicator_definitions import PANDAS_TA_CONFIG

        normalized = params.copy()

        # デバッグ用ログ
        # logger.debug(f"Normalizing params for {self.indicator_name}: {params}")

        if self.indicator_name in PANDAS_TA_CONFIG:
            config = PANDAS_TA_CONFIG[self.indicator_name]
            # logger.debug(f"Found PANDAS_TA_CONFIG for {self.indicator_name}: {config}")

            # AOの特別処理（パラメータなし）
            if self.indicator_name == "AO":
                return normalized

            # UOの特別処理
            if self.indicator_name == "UO":
                for param_name, aliases in config["params"].items():
                    for alias in aliases:
                        if alias in params:
                            value = params[alias]
                            # UOの場合、fast, medium, slowをマッピング
                            normalized[param_name] = value

                            # 範囲チェック (UOのパラメータは通常2-100)
                            if isinstance(value, (int, float)) and value < 2:
                                normalized[param_name] = 2
                            elif isinstance(value, (int, float)) and value > 100:
                                normalized[param_name] = 100

            for param_name, aliases in config["params"].items():
                # logger.debug(f"Processing param {param_name} with aliases {aliases}")
                for alias in aliases:
                    if alias in params:
                        value = params[alias]
                        # logger.debug(f"Found alias {alias} with value {value}")

                        # エイリアスマッピング
                        if param_name == "length" and alias in ["period"]:
                            # period -> length変換
                            normalized[param_name] = value
                            if alias in normalized:
                                del normalized[alias]
                            # logger.debug(f"Converted {alias} to {param_name}: {value}")
                        elif alias == "period":
                            # period -> length変換 (特殊ケース)
                            normalized["length"] = value
                            if "period" in normalized:
                                del normalized["period"]
                            # logger.debug(f"Converted {alias} to length: {value}")
                        else:
                            # 標準エイリアスマッピング
                            normalized[param_name] = value
                            if alias != param_name and alias in normalized:
                                del normalized[alias]
                            # logger.debug(f"Standard mapping: {alias} -> {param_name}")

                        # 範囲チェック
                        if any(word in alias for word in ["length", "period"]):
                            if isinstance(value, (int, float)) and value < 2:
                                normalized[param_name if alias != "period" else "length"] = 2
                            elif isinstance(value, (int, float)) and value > 200:
                                normalized[param_name if alias != "period" else "length"] = 200
                        else:
                            if isinstance(value, (int, float)) and value < 1:
                                normalized[param_name] = config["default_values"].get(param_name, 14)
                            elif isinstance(value, (int, float)) and value > 100:
                                normalized[param_name] = 100
        else:
            # logger.debug(f"No PANDAS_TA_CONFIG found for {self.indicator_name}")
            pass

        # logger.debug(f"Normalized params result: {normalized}")
        return normalized


class IndicatorConfigRegistry:
    """インジケーター設定レジストリ"""

    def __init__(self):
        self._configs: Dict[str, IndicatorConfig] = {}
        # 実験的インジケータ集合（ジェネレーターから参照）
        self.experimental_indicators = {
            "RMI",
            "DPO",
            "VORTEX",
            "EOM",
            "KVO",
            "PVT",
            "CMF",
        }

        # フォールバックマッピングをレジストリ内に定義
        self._fallback_indicators = self._setup_fallback_indicators()

    def _setup_fallback_indicators(self) -> Dict[str, str]:
        """未対応指標の代替指標マッピングを設定（オートストラテジー用）"""
        return {
            "WMA": "SMA",
            "HMA": "EMA",
            "KAMA": "EMA",
            "TEMA": "EMA",
            "DEMA": "EMA",
            "T3": "EMA",
            "ZLEMA": "EMA",
            "MIDPOINT": "SMA",
            "MIDPRICE": "SMA",
            "TRIMA": "SMA",
            "VWMA": "SMA",
            "STOCHRSI": "RSI",
            "STOCHF": "STOCH",
            "WILLR": "CCI",
            "MOMENTUM": "RSI",
            "MOM": "RSI",
            "ROC": "RSI",
            "ROCP": "RSI",
            "ROCR": "RSI",
            "AROON": "ADX",
            "AROONOSC": "ADX",
            "MFI": "RSI",
            "CMO": "RSI",
            "TRIX": "RSI",
            "ULTOSC": "RSI",
            "BOP": "RSI",
            "APO": "MACD",
            "PPO": "MACD",
            "DX": "ADX",
            "ADXR": "ADX",
            "PLUS_DI": "ADX",
            "MINUS_DI": "ADX",
            "NATR": "ATR",
            "TRANGE": "ATR",
            "KELTNER": "BB",
            "STDDEV": "ATR",
            "DONCHIAN": "BB",
            "AD": "OBV",
            "ADOSC": "OBV",
            "VWAP": "OBV",
            "PVT": "OBV",
            "EMV": "OBV",
            "PSAR": "SMA",
        }

    def register(self, config: IndicatorConfig) -> None:
        """設定を登録"""
        self._configs[config.indicator_name] = config
        # エイリアスも登録
        if config.aliases:
            for alias in config.aliases:
                self._configs[alias] = config

    def get_indicator_config(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """設定を取得 (get をリネーム)"""
        return self._configs.get(indicator_name)

    def list_indicators(self) -> List[str]:
        """登録されているインジケーター名のリストを取得"""
        return list(self._configs.keys())

    def get_supported_indicator_names(self) -> List[str]:
        """サポートされている指標の名前のリストを取得 (新規追加)"""
        return list(self._configs.keys())

    def generate_parameters_for_indicator(self, indicator_type: str) -> Dict[str, Any]:
        """
        指標タイプに応じたパラメータを生成

        IndicatorParameterManagerシステムを使用した統一されたパラメータ生成。
        """
        from app.services.indicators.parameter_manager import IndicatorParameterManager

        try:
            config = self.get_indicator_config(indicator_type)
            if config:
                manager = IndicatorParameterManager()
                return manager.generate_parameters(indicator_type, config)
            else:
                logger.warning(f"指標 {indicator_type} の設定が見つかりません")
                return {}
        except Exception as e:
            logger.error(f"指標 {indicator_type} のパラメータ生成に失敗: {e}")
            return {}

    def generate_json_name(self, indicator_name: str) -> str:
        """JSON形式の名前を生成"""
        config = self.get_indicator_config(indicator_name)
        if config:
            return config.generate_json_name()

        # 設定が見つからない場合のフォールバック
        return indicator_name


# グローバルレジストリインスタンス
indicator_registry = IndicatorConfigRegistry()
