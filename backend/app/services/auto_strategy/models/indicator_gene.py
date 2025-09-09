"""
指標遺伝子モデル
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class IndicatorGene:
    """
    指標遺伝子

    単一のテクニカル指標の設定を表現します。
    """

    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    json_config: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """指標遺伝子の妥当性を検証"""
        from .validator import GeneValidator

        validator = GeneValidator()
        return validator.validate_indicator_gene(self)

    def get_json_config(self) -> Dict[str, Any]:
        """JSON形式の設定を取得"""
        try:
            from app.services.indicators.config import indicator_registry

            config = indicator_registry.get_indicator_config(self.type)
            if config:
                resolved_params = {}
                for param_name, param_config in config.parameters.items():
                    resolved_params[param_name] = self.parameters.get(
                        param_name, param_config.default_value
                    )
                return {"indicator": self.type, "parameters": resolved_params}
            return {"indicator": self.type, "parameters": self.parameters}
        except ImportError:
            return {"indicator": self.type, "parameters": self.parameters}

@dataclass
class IndicatorParams:
    """
    指標パラメータ

    指標計算に必要なパラメータを定義します。
    """
    indicator_type: str
    period: int
    source: str