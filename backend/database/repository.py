"""
データアクセス層（リポジトリパターン）
レガシー互換性のため、新しい分割されたリポジトリクラスを再エクスポート
"""

# 新しい分割されたリポジトリクラスをインポート
from .repositories.ohlcv_repository import OHLCVRepository
from .repositories.funding_rate_repository import FundingRateRepository
from .repositories.open_interest_repository import OpenInterestRepository
from .repositories.data_collection_log_repository import DataCollectionLogRepository

# レガシー互換性のため、既存のインポートを維持
from .models import OHLCVData, DataCollectionLog, FundingRateData, OpenInterestData
from .connection import get_db, ensure_db_initialized

# 既存のコードとの互換性を保つため、新しいリポジトリクラスを再エクスポート
# 実際の実装は repositories/ ディレクトリ内の分割されたファイルにあります

__all__ = [
    "OHLCVRepository",
    "FundingRateRepository",
    "OpenInterestRepository",
    "DataCollectionLogRepository",
    "OHLCVData",
    "DataCollectionLog",
    "FundingRateData",
    "OpenInterestData",
    "get_db",
    "ensure_db_initialized"
]
