"""
テクニカル指標サービス（互換性レイヤー）

既存APIとの互換性を維持しながら、新しい分割されたテクニカル指標システムを使用します。
このファイルは既存のインポートとの互換性を保つためのラッパーです。
"""

# 新しい分割されたサービスをインポート
from app.core.services.indicators import TechnicalIndicatorService

# 既存のインポートとの互換性を維持
__all__ = ["TechnicalIndicatorService"]
