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


logger = logging.getLogger(__name__)


class IndicatorResultType(Enum):
    """インジケーター結果タイプ"""

    SINGLE = "single"  # 単一値（例：RSI、SMA）
    COMPLEX = "complex"  # 複数値（例：MACD、Bollinger Bands）


@dataclass
class ParameterConfig:
    """パラメータ設定"""

    name: str  # パラメータ名（例：period, fast_period）
    default_value: Union[int, float]  # デフォルト値
    min_value: Optional[Union[int, float]] = None  # 最小値
    max_value: Optional[Union[int, float]] = None  # 最大値
    description: Optional[str] = None  # パラメータの説明

    def validate_value(self, value: Union[int, float]) -> bool:
        """値の妥当性を検証"""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
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
    param_map: Dict[str, str] = field(default_factory=dict)

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

    def get_parameter_default(self, param_name: str) -> Union[int, float, None]:
        """パラメータのデフォルト値を取得"""
        if param_name in self.parameters:
            return self.parameters[param_name].default_value
        return None

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """パラメータの妥当性を検証"""
        for param_name, value in params.items():
            if param_name in self.parameters:
                if not self.parameters[param_name].validate_value(value):
                    logger.warning(f"Invalid parameter value: {param_name}={value}")
                    return False
        return True

    def generate_random_parameters(self) -> Dict[str, Any]:
        """
        ランダムなパラメータを生成

        Returns:
            生成されたパラメータ辞書
        """
        from app.services.indicators.parameter_manager import (
            IndicatorParameterManager,
        )

        manager = IndicatorParameterManager()
        return manager.generate_parameters(self.indicator_name, self)

    def get_parameter_ranges(self) -> Dict[str, Dict[str, Union[int, float]]]:
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

    def has_parameters(self) -> bool:
        """パラメータが定義されているかチェック"""
        return len(self.parameters) > 0

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


class IndicatorConfigRegistry:
    """インジケーター設定レジストリ"""

    def __init__(self):
        self._configs: Dict[str, IndicatorConfig] = {}
        # 実験的インジケータ集合（ジェネレーターから参照）
        self.experimental_indicators = {
            "RMI",
            "DPO",
            "CHOP",
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
            "MAMA": "EMA",
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
            "AVGPRICE": "SMA",
            "MEDPRICE": "SMA",
            "TYPPRICE": "SMA",
            "WCLPRICE": "SMA",
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

    # 互換メソッド（既存テスト向け）
    def get_all_indicator_names(self) -> List[str]:
        """登録済みの全指標名を返す（互換用エイリアス）"""
        return self.get_supported_indicator_names()

    def is_indicator_supported(self, indicator_name: str) -> bool:
        """指標が直接サポートされているかチェック (新規追加)"""
        return indicator_name in self._configs

    def resolve_indicator_type(self, indicator_type: str) -> Optional[str]:
        """指標タイプを解決し、未対応の場合は代替指標を返す (新規追加)"""
        if indicator_type in self._configs:
            return indicator_type
        elif indicator_type in self._fallback_indicators:
            fallback_type = self._fallback_indicators[indicator_type]
            logger.info(f"未対応指標 {indicator_type} を {fallback_type} で代替")
            return fallback_type
        return None

    def generate_parameters_for_indicator(
        self, indicator_type: str
    ) -> Dict[str, Any]:
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

    def resolve_indicator_name(self, name: str) -> Optional[str]:
        """
        指標名を動的に解決

        Args:
            name: 解決対象の名前（例: "MACD_0", "BB_1", "RSI"）

        Returns:
            解決された指標名、または None
        """
        # 直接的な指標名チェック
        if name in self._configs:
            return name

        # エイリアスチェック
        for indicator_name, config in self._configs.items():
            if name in config.aliases:
                return indicator_name

        # 出力名チェック（例: "MACD_0" -> "MACD"）
        for indicator_name, config in self._configs.items():
            if name in config.output_names:
                return indicator_name

        # パターンマッチング（例: "MACD_0" -> "MACD"）
        if "_" in name:
            base_name = name.split("_")[0]
            if base_name in self._configs:
                return base_name

        # フォールバック指標チェック
        if name in self._fallback_indicators:
            fallback_name = self._fallback_indicators[name]
            logger.info(f"未対応指標 {name} を {fallback_name} で代替")
            return fallback_name

        return None

    def get_output_index(self, name: str) -> Optional[int]:
        """
        出力インデックスを取得

        Args:
            name: 出力名（例: "MACD_0", "BB_1"）

        Returns:
            出力インデックス、または None
        """
        # パターンマッチング（例: "MACD_0" -> 0）
        if "_" in name:
            parts = name.split("_")
            if len(parts) >= 2:
                try:
                    return int(parts[-1])
                except ValueError:
                    pass

        # 設定から検索
        for config in self._configs.values():
            if name in config.output_names:
                return config.output_names.index(name)

        return None

    def get_default_output_name(self, indicator_name: str) -> Optional[str]:
        """
        デフォルト出力名を取得

        Args:
            indicator_name: 指標名

        Returns:
            デフォルト出力名、または None
        """
        config = self.get_indicator_config(indicator_name)
        if config:
            return config.default_output or config.indicator_name
        return None


# グローバルレジストリインスタンス
indicator_registry = IndicatorConfigRegistry()
