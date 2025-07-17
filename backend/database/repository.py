"""
データアクセス層（リポジトリパターン）
レガシー互換性のため、新しい分割されたリポジトリクラスを再エクスポート
"""

from .repositories.ohlcv_repository import OHLCVRepository
from .repositories.funding_rate_repository import FundingRateRepository
from .repositories.open_interest_repository import OpenInterestRepository
from .repositories.fear_greed_repository import FearGreedIndexRepository
from .repositories.ga_experiment_repository import GAExperimentRepository
from .repositories.generated_strategy_repository import GeneratedStrategyRepository
from .models import OHLCVData, FundingRateData, OpenInterestData, FearGreedIndexData
from .connection import get_db, ensure_db_initialized

# 既存のコードとの互換性を保つため、新しいリポジトリクラスを再エクスポート
# 実際の実装は repositories/ ディレクトリ内の分割されたファイルにあります

__all__ = [
    "OHLCVRepository",
    "FundingRateRepository",
    "OpenInterestRepository",
    "FearGreedIndexRepository",
    "GAExperimentRepository",
    "GeneratedStrategyRepository",
    "OHLCVData",
    "FundingRateData",
    "OpenInterestData",
    "FearGreedIndexData",
    "get_db",
    "ensure_db_initialized",
]
