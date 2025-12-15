"""
指標遺伝子モデル
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class IndicatorGene:
    """
    指標遺伝子

    単一のテクニカル指標の設定を表現します。

    Attributes:
        type: 指標タイプ（例: "SMA", "RSI"）
        parameters: 指標パラメータ（例: {"period": 20}）
        enabled: 指標が有効かどうか
        timeframe: この指標が計算されるタイムフレーム。
            None の場合は戦略のデフォルトタイムフレームを使用。
            例: "1h", "4h", "1d" など
        id: 指標の一意識別子（オプション）。複数の同じ種類の指標を区別するために使用。
        json_config: JSON形式の設定キャッシュ
    """

    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    timeframe: Optional[str] = None
    id: Optional[str] = None
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


