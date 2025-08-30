"""
シンボル正規化サービス

シンボルの正規化、重複除去、統一インターフェースを提供します。
複数の取引所やサービスで使用されるシンボルのドキュメントを統一化します。
"""

import logging
from typing import Any, Dict, List, Optional

from app.config.unified_config import unified_config

logger = logging.getLogger(__name__)


class SymbolNormalizationService:
    """
    統一化されたシンボル正規化サービス

    このクラスは以下の機能を統合して提供します：
    - Bybit 形式の変換 (:USDT, :USD)
    - 汎用マッピング処理
    - 妥当性検証
    - デフォルト値設定

    注意: 現在のシステムではBTC/USDT:USDTのみをサポートしています。
    """

    # デフォルト値（現在のシステム仕様に合わせる）
    DEFAULT_SYMBOL = "BTC/USDT:USDT"

    # プロバイダーマッピング
    PROVIDER_MAPPING = {
        "bybit": {
            "spot_pattern": "{symbol}:USDT",
            "perpetual_delimiter": ":",
            "supported_suffixes": ["/USDT", "/USD"],
        },
        "binance": {
            "spot_pattern": "{symbol}:USDT",
            "perpetual_delimiter": ":",
            "supported_suffixes": ["/USDT", "/USDC"],
        },
        "generic": {
            "spot_pattern": "{symbol}:USDT",
            "perpetual_delimiter": ":",
            "supported_suffixes": ["/USDT"],
        },
    }

    @staticmethod
    def normalize_symbol(
        symbol: Optional[str],
        provider: str = "bybit",
        symbol_mapping: Optional[Dict[str, str]] = None,
        supported_symbols: Optional[List[str]] = None
    ) -> str:
        """
        シンボルを正規化（統合版）

        Args:
            symbol: 正規化するシンボル
            provider: プロバイダー名 ("bybit", "binance", "generic")
            symbol_mapping: 追加マッピング辞書
            supported_symbols: サポートされているシンボルのリスト

        Returns:
            正規化されたシンボル

        Raises:
            ValueError: シンボルが無効な場合
        """
        # Noneまたは空文字の場合デフォルト値を返す
        if not symbol:
            return SymbolNormalizationService.DEFAULT_SYMBOL

        # プロバイダー設定を取得
        provider_config = SymbolNormalizationService.PROVIDER_MAPPING.get(
            provider, SymbolNormalizationService.PROVIDER_MAPPING["generic"])

        symbol = str(symbol).strip()

        # 大文字変換（シンボルは大文字が標準）
        symbol = symbol.upper()

        # マッピング適用（オプション）
        if symbol_mapping and symbol in symbol_mapping:
            symbol = symbol_mapping[symbol]

        # プロバイダー固有の処理
        normalized_symbol = SymbolNormalizationService._apply_provider_normalization(
            symbol, provider_config)

        # サポートシンボルチェック（オプション）
        if supported_symbols:
            if normalized_symbol not in supported_symbols:
                raise ValueError(
                    f"サポートされていないシンボルです: '{symbol}' → '{normalized_symbol}'。"
                    f"サポートされているシンボル: {supported_symbols}")

        return normalized_symbol

    @staticmethod
    def _apply_provider_normalization(symbol: str, provider_config: Dict[str, Any]) -> str:
        """
        プロバイダー固有の正規化処理

        Args:
            symbol: 正規化対象のシンボル
            provider_config: プロバイダー設定

        Returns:
            正規化されたシンボル
        """
        delimiters = provider_config["perpetual_delimiter"]
        supported_suffixes = provider_config["supported_suffixes"]
        spot_pattern = provider_config["spot_pattern"]

        # 既に無期限取引ペア形式の場合そのまま返す
        if delimiters in symbol:
            return symbol

        # 各サフィックスのチェックと変換
        for suffix in supported_suffixes:
            if symbol.endswith(suffix):
                # /USDT → :USDT
                base_symbol = symbol[:-len(suffix)]  # prefix除去
                perpetual_suffix = suffix.replace("/", delimiters)
                return f"{base_symbol}{perpetual_suffix}"

        # デフォルト処理: 区切り文字がなければ追加
        return spot_pattern.format(symbol=symbol)

    @staticmethod
    def is_valid_symbol_format(symbol: str) -> bool:
        """
        シンボルのフォーマットが有効かをチェック

        Args:
            symbol: 検証対象のシンボル

        Returns:
            有効なフォーマットの場合True
        """
        if not symbol or not isinstance(symbol, str):
            return False

        # 基本形式チェック（アルファベット、数字、-、/、: のみ許可）
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/:')
        return all(c in allowed_chars for c in symbol.upper())

    @staticmethod
    def get_supported_symbols(provider: str = "bybit") -> List[str]:
        """
        プロバイダーでサポートされているシンボルを取得

        Args:
            provider: プロバイダー名

        Returns:
            指定プロバイダーでサポートされているシンボルのリスト
        """
        try:
            # unified_configからの取得を優先
            supported_symbols = unified_config.market.supported_symbols[:]
            return supported_symbols
        except Exception:
            # Fallback: 現在のシステム仕様に基づく（BTC/USDT:USDTのみ）
            return ["BTC/USDT:USDT"]

    @staticmethod
    def validate_symbol_symbol(
        symbol: str,
        supported_symbols: Optional[List[str]] = None,
        provider: str = "bybit"
    ) -> bool:
        """
        シンボルが指定のリストに含まれているかを検証（名前の衝突回避のためvalidate_symbol_symbol）

        Args:
            symbol: 検証対象のシンボル
            supported_symbols: サポートシンボルリスト（Noneの場合は自動取得）
            provider: プロバイダー名

        Returns:
            有効な場合True
        """
        if not supported_symbols:
            supported_symbols = SymbolNormalizationService.get_supported_symbols(provider)

        return symbol in supported_symbols

    @staticmethod
    def get_symbol_mapping(provider: str = "bybit") -> Dict[str, str]:
        """
        プロバイダー固有のシンボルマッピングを取得

        Args:
            provider: プロバイダー名

        Returns:
            マッピング辞書
        """
        try:
            # unified_configからの取得を優先
            return unified_config.market.symbol_mapping.copy()
        except Exception:
            # Fallback: 現在のシステム仕様に基づくマッピング
            return {
                "BTCUSDT": "BTC/USDT:USDT",
                "BTC-USDT": "BTC/USDT:USDT",
                "BTC/USDT": "BTC/USDT:USDT",
                "BTC/USDT:USDT": "BTC/USDT:USDT",
                "BTCUSDT_PERP": "BTC/USDT:USDT",
            }


# 便利関数（後方互換性維持）
def normalize_symbol(
    symbol: Optional[str],
    provider: str = "bybit",
    symbol_mapping: Optional[Dict[str, str]] = None,
    supported_symbols: Optional[List[str]] = None
) -> str:
    """
    シンボルを正規化（便利関数）

    Args:
        symbol: 正規化するシンボル
        provider: プロバイダー名
        symbol_mapping: シンボルマッピング辞書
        supported_symbols: サポートシンボルリスト

    Returns:
        正規化されたシンボル
    """
    return SymbolNormalizationService.normalize_symbol(
        symbol, provider, symbol_mapping, supported_symbols)


# validate_symbol関数（名前の衝突回避）
def validate_symbol_symbol(
    symbol: str,
    supported_symbols: Optional[List[str]] = None,
    provider: str = "bybit"
) -> bool:
    """
    シンボルが有効かどうかを検証（名前の衝突回避のためvalidate_symbol_symbol）

    Args:
        symbol: 検証対象のシンボル
        supported_symbols: サポートシンボルリスト
        provider: プロバイダー名

    Returns:
        有効な場合True
    """
    return SymbolNormalizationService.validate_symbol_symbol(
        symbol, supported_symbols, provider)