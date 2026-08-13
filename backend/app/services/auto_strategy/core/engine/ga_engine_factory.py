"""
GAエンジンファクトリ

GAエンジンの構築とコンポーネントの初期化を担当します。
"""

import logging
from typing import TYPE_CHECKING

from app.services.auto_strategy.config.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.backtest.services.backtest_service import BacktestService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .ga_engine import GeneticAlgorithmEngine


class GeneticAlgorithmEngineFactory:
    """
    GAエンジンの構築を行うファクトリクラス
    """

    @staticmethod
    def create_engine(
        backtest_service: BacktestService,
        ga_config: GAConfig,
        seed_strategy_provider: object | None = None,
    ) -> "GeneticAlgorithmEngine":
        """
        指定された設定とサービス依存関係を用いて GA エンジンを構築します。

        このメソッドは以下の手順でエンジンを構成します：
        1. ログレベルの動的設定。
        2. 遺伝子生成器（`RandomGeneGenerator`）の初期化。
        3. 全ての依存関係を注入した `GeneticAlgorithmEngine` インスタンスを生成。

        Args:
            backtest_service (BacktestService): 個体評価に使用するバックテスト実行サービス。
            ga_config (GAConfig): アルゴリズムのパラメータ、目的関数を含む統合構成。
            seed_strategy_provider (Optional[object]): 反復改善ループ用の
                シード戦略プロバイダ（get_seed_strategies(config) を実装）。

        Returns:
            GeneticAlgorithmEngine: 実行準備が整ったGAエンジンインスタンス。
        """
        # ログレベルの設定
        auto_strategy_logger = logging.getLogger("app.services.auto_strategy")
        auto_strategy_logger.setLevel(
            getattr(logging, ga_config.log_level.upper(), logging.INFO)
        )

        # 遺伝子生成器の初期化
        gene_generator = RandomGeneGenerator(ga_config)

        # エンジンの生成
        from .ga_engine import GeneticAlgorithmEngine

        engine = GeneticAlgorithmEngine(
            backtest_service=backtest_service,
            gene_generator=gene_generator,
            seed_strategy_provider=seed_strategy_provider,
        )

        logger.debug("GAエンジンを初期化しました (Mode: Standard)")
        return engine
