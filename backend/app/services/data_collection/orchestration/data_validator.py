"""
データバリデーションモジュール

シンボルと時間軸のバリデーションを担当します。
"""

import logging


from app.config.unified_config import unified_config
from app.utils.error_handler import safe_operation

logger = logging.getLogger(__name__)


class DataValidator:
    """
    データバリデーションクラス

    シンボルと時間軸のバリデーションを行います。
    """

    @safe_operation(context="シンボル・時間軸バリデーション", is_api_call=False)
    def validate_symbol_and_timeframe(self, symbol: str, timeframe: str) -> str:
        """
        シンボルと時間軸のバリデーション

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            正規化されたシンボル

        Raises:
            ValueError: バリデーションエラー
        """
        # シンボル正規化
        normalized_symbol = unified_config.market.symbol_mapping.get(symbol, symbol)
        if normalized_symbol not in unified_config.market.supported_symbols:
            raise ValueError(f"サポートされていないシンボル: {symbol}")

        # 時間軸検証
        if timeframe not in unified_config.market.supported_timeframes:
            raise ValueError(f"無効な時間軸: {timeframe}")

        return normalized_symbol
