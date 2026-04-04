"""
リポジトリパッケージ

データベースアクセスを抽象化するリポジトリパターンを実装します。
各リポジトリは特定のエンティティに対するCRUD操作を提供し、
サービス層とデータベース間の結合を解消します。

主なリポジトリ:
- base_repository.py: 共通CRUD操作の基底クラス
- ohlcv_repository.py: OHLCVデータアクセス
- funding_rate_repository.py: ファンディングレートデータアクセス
- open_interest_repository.py: オープンインタレストデータアクセス
- backtest_result_repository.py: バックテスト結果データアクセス
- ga_experiment_repository.py: GA実験データアクセス
- generated_strategy_repository.py: 生成戦略データアクセス
- long_short_ratio_repository.py: ロング/ショート比率データアクセス
"""

from .backtest_result_repository import BacktestResultRepository
from .base_repository import BaseRepository
from .funding_rate_repository import FundingRateRepository
from .ga_experiment_repository import GAExperimentRepository
from .generated_strategy_repository import GeneratedStrategyRepository
from .long_short_ratio_repository import LongShortRatioRepository
from .ohlcv_repository import OHLCVRepository
from .open_interest_repository import OpenInterestRepository

__all__ = [
    "BaseRepository",
    "OHLCVRepository",
    "FundingRateRepository",
    "OpenInterestRepository",
    "BacktestResultRepository",
    "GAExperimentRepository",
    "GeneratedStrategyRepository",
    "LongShortRatioRepository",
]



