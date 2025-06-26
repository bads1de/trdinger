"""
インジケーター設定管理クラス

JSON形式でのインジケーター設定を管理し、
パラメータ埋め込み文字列からの移行をサポートします。
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import logging

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
    
    # 命名設定（後方互換性のため）
    legacy_name_format: Optional[str] = None  # 旧形式の名前フォーマット
    
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
    
    def generate_json_name(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """JSON形式のインジケーター名を生成"""
        # パラメータにデフォルト値を適用
        resolved_params = {}
        for param_name, param_config in self.parameters.items():
            resolved_params[param_name] = parameters.get(
                param_name, param_config.default_value
            )
        
        return {
            "indicator": self.indicator_name,
            "parameters": resolved_params
        }
    
    def generate_legacy_name(self, parameters: Dict[str, Any]) -> str:
        """レガシー形式の名前を生成（後方互換性）"""
        if not self.legacy_name_format:
            return self.indicator_name
        
        # パラメータにデフォルト値を適用
        format_params = {"indicator": self.indicator_name}
        for param_name, param_config in self.parameters.items():
            value = parameters.get(param_name, param_config.default_value)
            format_params[param_name] = value
        
        try:
            return self.legacy_name_format.format(**format_params)
        except KeyError as e:
            logger.warning(f"Legacy name format error: {e}")
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
    
    def register(self, config: IndicatorConfig) -> None:
        """設定を登録"""
        self._configs[config.indicator_name] = config
    
    def get(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """設定を取得"""
        return self._configs.get(indicator_name)
    
    def list_indicators(self) -> List[str]:
        """登録されているインジケーター名のリストを取得"""
        return list(self._configs.keys())
    
    def generate_json_name(self, indicator_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """JSON形式の名前を生成"""
        config = self.get(indicator_name)
        if config:
            return config.generate_json_name(parameters)
        
        # 設定が見つからない場合のフォールバック
        return {
            "indicator": indicator_name,
            "parameters": parameters
        }
    
    def generate_legacy_name(self, indicator_name: str, parameters: Dict[str, Any]) -> str:
        """レガシー形式の名前を生成"""
        config = self.get(indicator_name)
        if config:
            return config.generate_legacy_name(parameters)
        
        # 設定が見つからない場合のフォールバック
        return indicator_name


# グローバルレジストリインスタンス
indicator_registry = IndicatorConfigRegistry()
