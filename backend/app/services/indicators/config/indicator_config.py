"""
インジケーター設定管理クラス

JSON形式でのインジケーター設定を管理し、
パラメータ埋め込み文字列からの移行をサポートします。
"""

import json
import logging
from dataclasses import asdict, dataclass, field
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
        description: Optional[str] = None
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

        if self.min_value is not None and isinstance(self.min_value, (int, float)) and value < self.min_value:
            return False
        if self.max_value is not None and isinstance(self.max_value, (int, float)) and value > self.max_value:
            return False
        return True


@dataclass
class IndicatorConfig:
    """インジケーター設定クラス"""

    # 基本情報
    indicator_name: str  # インジケーター名（例：RSI、MACD）
    adapter_function: Optional[Any] = None  # アダプター関数への参照
    required_data: List[str] = field(default_factory=list)  # 必要なデータ列
    result_type: IndicatorResultType = IndicatorResultType.SINGLE

    # パラメータ設定
    parameters: Dict[str, ParameterConfig] = field(default_factory=dict)

    # 結果処理設定
    result_handler: Optional[str] = None  # 複数値結果の処理ハンドラー

    # データキー→関数引数名マッピング（オーバーライド用。未設定なら従来マッピングを使用）
    param_map: Dict[str, Optional[str]] = field(default_factory=dict)

    # メタデータ（遺伝子生成用）
    scale_type: Optional[IndicatorScaleType] = None  # スケールタイプ
    category: Optional[str] = None  # 指標カテゴリ（例: trend, momentum）
    needs_normalization: bool = False  # 入力データの正規化が必要か

    # 名前解決用メタデータ（リファクタリング追加）
    output_names: List[str] = field(
        default_factory=list
    )  # 出力名リスト（例: ["MACD_0", "MACD_1", "MACD_2"]）
    default_output: Optional[str] = None  # デフォルト出力名（例: "MACD_0"）
    aliases: List[str] = field(
        default_factory=list
    )  # エイリアス（例: ["MACD", "BB_Middle"]）

    def add_parameter(self, param_config: ParameterConfig) -> None:
        """パラメータを追加"""
        self.parameters[param_config.name] = param_config

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """パラメータの妥当性を検証"""
        for param_name, value in params.items():
            if param_name in self.parameters:
                if not self.parameters[param_name].validate_value(value):
                    logger.warning(f"Invalid parameter value: {param_name}={value}")
                    return False
        return True

    def get_parameter_ranges(self) -> Dict[str, Dict[str, Union[int, float, str, bool]]]:
        """
        パラメータの範囲情報を取得

        Returns:
            パラメータ範囲情報の辞書
        """
        ranges = {}
        for param_name, param_config in self.parameters.items():
            ranges[param_name] = {
                "min": param_config.min_value,
                "max": param_config.max_value,
                "default": param_config.default_value,
            }
        return ranges

    def generate_json_name(self) -> str:
        """JSON形式のインジケーター名（=指標名）を生成"""
        # JSON形式では指標名にパラメータを含めない
        return self.indicator_name

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        # Enumを文字列に変換
        result["result_type"] = self.result_type.value
        # 関数参照は除外
        result.pop("adapter_function", None)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndicatorConfig":
        """辞書から復元"""
        # Enumを復元
        if "result_type" in data:
            data["result_type"] = IndicatorResultType(data["result_type"])

        # ParameterConfigを復元
        if "parameters" in data:
            params = {}
            for name, param_data in data["parameters"].items():
                params[name] = ParameterConfig(**param_data)
            data["parameters"] = params

        return cls(**data)

    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "IndicatorConfig":
        """JSON文字列から復元"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        指標パラメータ正規化メソッド

        - period -> length 変換
        - 必須 length のデフォルト補完
        - 各指標固有のパラメータマッピング
        """
        import inspect
        from typing import Dict, Any

        converted_params: Dict[str, Any] = {}

        # 特殊処理が必要な指標の変換ルールを定義
        special_conversions = {
            # SAR: acceleration -> af, maximum -> max_af, 予期しないパラメータを除外
            "SAR": {
                "param_map": {"acceleration": "af", "maximum": "max_af"},
                "exclude_params": {"period", "length"},
                "include_all_mapped": False,
            },
            # VWMA: period -> length, close -> data (param_mapがあれば優先)
            "VWMA": {
                "param_map": {"period": "length", "close": "data"},
                "fallback_map": {"period": "length"},
                "use_config_param_map": True,
            },
            # RMA: period -> length, close -> data
            "RMA": {
                "param_map": {"period": "length", "close": "data"},
            },
            # STC: close -> data
            "STC": {
                "param_map": {"close": "data"},
            },
            # RSI_EMA_CROSS: close -> data
            "RSI_EMA_CROSS": {
                "param_map": {"close": "data"},
            },
            # VP: period -> width, lengthパラメータを除外
            "VP": {
                "param_map": {"period": "width"},
                "exclude_params": {"length"},
            },
        }

        # パラメータを一切受け取らない指標（ボリューム系指標）
        volume_indicators = {"NVI", "PVI", "PVT", "AD", "PVR"}

        # NO_LENGTH_INDICATORS (lengthパラメータを使用しない指標)
        NO_LENGTH_INDICATORS = {
            "SAR",
            "OBV",
            "VWAP",
            "AD",
            "ADOSC",
            "AO",
            "ICHIMOKU",
            "PVT",
            "PVOL",
            "PVR",
            "PPO",
            "APO",
            "ULTOSC",
            "BOP",
            "STC",
            "KDJ",
            "CDL_PIERCING",
            "CDL_HAMMER",
            "CDL_HANGING_MAN",
            "CDL_HARAMI",
            "CDL_DARK_CLOUD_COVER",
            "CDL_THREE_BLACK_CROWS",
            "CDL_THREE_WHITE_SOLDIERS",
            "CDL_MARUBOZU",
            "CDL_SPINNING_TOP",
            "CDL_SHOOTING_STAR",
            "CDL_ENGULFING",
            "CDL_MORNING_STAR",
            "CDL_EVENING_STAR",
            "CDL_DOJI",
            "HAMMER",
            "ENGULFING_PATTERN",
            "MORNING_STAR",
            "EVENING_STAR",
            "RSI_EMA_CROSS",
            "NVI",
            "PVI",
            "PVT",
            "CMF",
            "EOM",
            "KVO",
            "STOCH",
            "STOCHF",
            "KST",
            "SMI",
            "UO",
            "PVO",
            "TRANGE",
            "BB",
            "ACOS",
            "ASIN",
            "ATAN",
            "COS",
            "COSH",
            "SIN",
            "SINH",
            "TAN",
            "TANH",
            "SQRT",
            "EXP",
            "LN",
            "LOG10",
            "CEIL",
            "FLOOR",
            "ADD",
            "SUB",
            "MULT",
            "DIV",
            "WCP",
            "HLC3",
            "HL2",
            "OHLC4",
            "VP",
            "AOBV",
            "HWC",
        }

        # 特殊処理が必要な指標の場合
        if self.indicator_name in special_conversions:
            conversion_rule = special_conversions[self.indicator_name]

            for key, value in params.items():
                # 除外パラメータはスキップ
                if (
                    "exclude_params" in conversion_rule
                    and key in conversion_rule["exclude_params"]
                ):
                    continue

                # パラメータマッピングの適用
                if "param_map" in conversion_rule and key in conversion_rule["param_map"]:
                    converted_params[conversion_rule["param_map"][key]] = value
                elif (
                    "use_config_param_map" in conversion_rule
                    and self.param_map
                ):
                    # config.param_map の値にキーが含まれているかチェック
                    if key in [v for v in self.param_map.values() if v is not None]:
                        converted_params[key] = value
                    elif key in conversion_rule.get("fallback_map", {}):
                        converted_params[conversion_rule["fallback_map"][key]] = value
                    elif conversion_rule.get("include_all_mapped", True):
                        converted_params[key] = value
                else:
                    # マッピングルールがない場合はそのまま
                    converted_params[key] = value

            return converted_params

        # パラメータを一切受け取らない指標（ボリューム系指標）
        if self.indicator_name in volume_indicators:
            # period や length などのパラメータを除外
            for key, value in params.items():
                if key not in {"period", "length"}:
                    converted_params[key] = value
            return converted_params

        # NO_LENGTH_INDICATORS の包括的特殊処理
        if self.indicator_name in NO_LENGTH_INDICATORS:
            # これらの指標は period や length パラメータを受け取らない
            for key, value in params.items():
                if key not in {"period", "length"}:
                    converted_params[key] = value
            return converted_params

        # すべての指標がlengthパラメータを使用するため、変換は不要
        # ただし、互換性のためperiodパラメータが提供された場合はlengthに変換
        for key, value in params.items():
            if key == "period":
                converted_params["length"] = value
            else:
                converted_params[key] = value

        # length 必須のアダプタにデフォルト補完
        # SAR には length パラメータを追加しない（af, max_af のみを使用）
        if self.indicator_name == "SAR":
            pass  # SAR には length を追加しない
        elif self.indicator_name in NO_LENGTH_INDICATORS:
            pass  # これらの指標には length を追加しない
        elif self.indicator_name.startswith("CDL_") and "length" not in converted_params:
            pass  # すべてのパターン認識指標には length を追加しない
        elif self.adapter_function is not None and "length" in inspect.signature(self.adapter_function).parameters and "length" not in converted_params:
            # period_to_length_indicators に含まれる指標のみ length を自動追加
            default_len = params.get("period")
            if default_len is None and self.parameters:
                if "period" in self.parameters:
                    default_len = self.parameters["period"].default_value
                elif "length" in self.parameters:
                    default_len = self.parameters["length"].default_value
            converted_params["length"] = (
                default_len if default_len is not None else 14  # DEFAULT_LENGTH
            )

        return converted_params


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
