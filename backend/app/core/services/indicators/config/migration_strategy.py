"""
インジケーター命名形式の移行戦略

パラメータ埋め込み文字列からJSON形式への段階的移行をサポートします。
"""

import re
import logging
from typing import Dict, Any, Optional, Union, Tuple
from .indicator_config import indicator_registry

logger = logging.getLogger(__name__)


class IndicatorNameMigrator:
    """インジケーター名の移行管理クラス"""

    def __init__(self):
        # レガシー形式のパターン定義
        self.legacy_patterns = {
            # 単一パラメータパターン
            r"^([A-Z]+)_(\d+)$": ["indicator", "period"],
            # 複数パラメータパターン
            r"^([A-Z]+)_(\d+)_(\d+)$": ["indicator", "fast_period", "slow_period"],
            r"^([A-Z]+)_(\d+)_(\d+)_(\d+)$": [
                "indicator",
                "fast_period",
                "slow_period",
                "signal_period",
            ],
            # 特殊パターン
            r"^BB_(UPPER|MIDDLE|LOWER)_(\d+)$": ["bb_type", "period"],
            r"^MACD_(LINE|SIGNAL|HISTOGRAM)_(\d+)$": ["macd_type", "fast_period"],
        }

    def parse_legacy_name(self, legacy_name: str) -> Optional[Dict[str, Any]]:
        """レガシー形式の名前を解析してJSON形式に変換"""

        # パラメータなしの指標（例：OBV、AD）
        if legacy_name in ["OBV", "AD", "PVT"]:
            return {"indicator": legacy_name, "parameters": {}}

        # パターンマッチング
        for pattern, param_names in self.legacy_patterns.items():
            match = re.match(pattern, legacy_name)
            if match:
                groups = match.groups()

                # 特殊パターンの処理
                if pattern.startswith("^BB_"):
                    return self._parse_bb_legacy(groups)
                elif pattern.startswith("^MACD_"):
                    return self._parse_macd_legacy(groups)

                # 通常パターンの処理
                result = {"indicator": groups[0], "parameters": {}}
                for i, param_name in enumerate(param_names[1:], 1):
                    if i < len(groups):
                        result["parameters"][param_name] = int(groups[i])

                return result

        logger.warning(f"Unknown legacy name format: {legacy_name}")
        return None

    def _parse_bb_legacy(self, groups: Tuple[str, ...]) -> Dict[str, Any]:
        """Bollinger Bandsのレガシー形式を解析"""
        bb_type, period = groups
        return {
            "indicator": "BB",
            "parameters": {"period": int(period)},
            "component": bb_type.lower(),  # upper, middle, lower
        }

    def _parse_macd_legacy(self, groups: Tuple[str, ...]) -> Dict[str, Any]:
        """MACDのレガシー形式を解析"""
        macd_type, fast_period = groups
        return {
            "indicator": "MACD",
            "parameters": {
                "fast_period": int(fast_period),
                "slow_period": 26,  # デフォルト値
                "signal_period": 9,  # デフォルト値
            },
            "component": macd_type.lower(),  # line, signal, histogram
        }

    def generate_json_name(
        self, indicator: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """JSON形式の名前を生成"""
        return indicator_registry.generate_json_name(indicator, parameters)

    def generate_legacy_name(self, json_config: Dict[str, Any]) -> str:
        """JSON形式からレガシー形式の名前を生成"""
        indicator = json_config["indicator"]
        parameters = json_config.get("parameters", {})

        return indicator_registry.generate_legacy_name(indicator, parameters)

    def is_legacy_format(self, name: str) -> bool:
        """レガシー形式かどうかを判定"""
        if isinstance(name, dict):
            return False  # JSON形式

        # 文字列の場合、パターンマッチングで判定
        for pattern in self.legacy_patterns.keys():
            if re.match(pattern, name):
                return True

        # パラメータなしの指標
        if name in ["OBV", "AD", "PVT"]:
            return True

        return False

    def normalize_name(self, name: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """名前を正規化（JSON形式に統一）"""
        if isinstance(name, dict):
            return name  # 既にJSON形式

        # レガシー形式を解析
        parsed = self.parse_legacy_name(name)
        if parsed:
            return parsed

        # 解析できない場合のフォールバック
        return {"indicator": name, "parameters": {}}


class BackwardCompatibilityManager:
    """後方互換性管理クラス"""

    def __init__(self):
        self.migrator = IndicatorNameMigrator()
        self.compatibility_mode = False  # JSON形式を優先（レガシー形式は非推奨）

    def enable_compatibility_mode(self):
        """互換性モードを有効化"""
        self.compatibility_mode = True
        logger.info("Backward compatibility mode enabled")

    def disable_compatibility_mode(self):
        """互換性モードを無効化"""
        self.compatibility_mode = False
        logger.info("Backward compatibility mode disabled")

    def resolve_indicator_name(
        self, name: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """インジケーター名を解決（互換性を考慮）"""
        if not self.compatibility_mode:
            # 互換性モード無効時はJSON形式のみ受け入れ
            if isinstance(name, str):
                raise ValueError(f"Legacy format not supported: {name}")
            return name

        # 互換性モード有効時は両形式をサポート
        return self.migrator.normalize_name(name)

    def generate_name(
        self, indicator: str, parameters: Dict[str, Any], format_type: str = "auto"
    ) -> Union[str, Dict[str, Any]]:
        """指定された形式で名前を生成

        Args:
            indicator: インジケーター名
            parameters: パラメータ
            format_type: "legacy", "json", "auto"
        """
        if format_type == "legacy":
            json_config = {"indicator": indicator, "parameters": parameters}
            return self.migrator.generate_legacy_name(json_config)
        elif format_type == "json":
            return self.migrator.generate_json_name(indicator, parameters)
        else:  # auto
            # JSON形式を優先（レガシー形式は非推奨）
            return self.migrator.generate_json_name(indicator, parameters)


# グローバルインスタンス
migrator = IndicatorNameMigrator()
compatibility_manager = BackwardCompatibilityManager()
