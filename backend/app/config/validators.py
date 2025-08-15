"""
設定バリデーター

設定値の妥当性を検証するためのバリデーションロジックを提供します。
単一責任原則に従い、バリデーション機能のみを担当します。
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class MarketDataValidator:
    """市場データ設定のバリデーター"""

    @staticmethod
    def validate_symbol(symbol: str, supported_symbols: List[str]) -> bool:
        """
        シンボルが有効かどうかを検証

        Args:
            symbol: 検証するシンボル
            supported_symbols: サポートされているシンボルのリスト

        Returns:
            有効な場合True、無効な場合False
        """
        return symbol in supported_symbols

    @staticmethod
    def validate_timeframe(timeframe: str, supported_timeframes: List[str]) -> bool:
        """
        時間軸が有効かどうかを検証

        Args:
            timeframe: 検証する時間軸
            supported_timeframes: サポートされている時間軸のリスト

        Returns:
            有効な場合True、無効な場合False
        """
        return timeframe in supported_timeframes

    @staticmethod
    def normalize_symbol(
        symbol: str, symbol_mapping: Dict[str, str], supported_symbols: List[str]
    ) -> str:
        """
        シンボルを正規化

        Args:
            symbol: 正規化するシンボル
            symbol_mapping: シンボルマッピング辞書
            supported_symbols: サポートされているシンボルのリスト

        Returns:
            正規化されたシンボル

        Raises:
            ValueError: サポートされていないシンボルの場合
        """
        # 大文字に変換し、空白を除去
        symbol = symbol.strip().upper()

        # マッピングテーブルから検索
        if symbol in symbol_mapping:
            normalized = symbol_mapping[symbol]
        else:
            normalized = symbol

        # サポートされているシンボルかチェック
        if normalized not in supported_symbols:
            raise ValueError(
                f"サポートされていないシンボルです: '{symbol}'。"
                f"サポートされているシンボルは {', '.join(supported_symbols)} です。"
            )

        return normalized


class MLConfigValidator:
    """ML設定のバリデーター"""

    @staticmethod
    def validate_predictions(predictions: Dict[str, float]) -> bool:
        """
        予測値の妥当性を検証

        Args:
            predictions: 予測値の辞書

        Returns:
            有効な場合True、無効な場合False
        """
        try:
            # 必要なキーの存在確認
            required_keys = ["up", "down", "range"]
            if not all(key in predictions for key in required_keys):
                return False

            # 値の範囲確認（0-1の範囲）
            for key, value in predictions.items():
                if not isinstance(value, (int, float)):
                    return False
                if not (0.0 <= value <= 1.0):
                    return False

            # 合計値の確認（0.8-1.2の範囲）
            total = sum(predictions.values())
            if not (0.8 <= total <= 1.2):
                return False

            return True

        except Exception:
            return False
