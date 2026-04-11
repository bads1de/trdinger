"""
GAエンジンファクトリ

GAエンジンの構築とコンポーネントの初期化を担当します。
"""

import logging
from typing import TYPE_CHECKING

from app.services.auto_strategy.config.ga import GAConfig
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
        backtest_service: BacktestService, ga_config: GAConfig
    ) -> "GeneticAlgorithmEngine":
        """
        指定された設定とサービス依存関係を用いて GA エンジンを構築します。

        このメソッドは以下の手順でエンジンを構成します：
        1. ログレベルの動的設定。
        2. 遺伝子生成器（`RandomGeneGenerator`）の初期化。
        3. ハイブリッド設定（MLフィルタの併用）が有効な場合、必要なMLコンポーネントを準備。
        4. 全ての依存関係を注入した `GeneticAlgorithmEngine` インスタンスを生成。

        Args:
            backtest_service (BacktestService): 個体評価に使用するバックテスト実行サービス。
            ga_config (GAConfig): アルゴリズムのパラメータ、目的関数、ハイブリッド設定を含む統合構成。

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

        # ハイブリッドコンポーネントの準備
        hybrid_predictor = None
        hybrid_feature_adapter = None

        hybrid_config = ga_config.hybrid_config
        if hybrid_config.mode:
            hybrid_predictor, hybrid_feature_adapter = (
                GeneticAlgorithmEngineFactory._setup_hybrid_components(ga_config)
            )

        # エンジンの生成
        from .ga_engine import GeneticAlgorithmEngine

        engine = GeneticAlgorithmEngine(
            backtest_service=backtest_service,
            gene_generator=gene_generator,
            hybrid_mode=hybrid_config.mode,
            hybrid_predictor=hybrid_predictor,
            hybrid_feature_adapter=hybrid_feature_adapter,
        )

        logger.info(
            f"GAエンジンを初期化しました (Mode: {'Hybrid' if hybrid_config.mode else 'Standard'})"
        )
        return engine

    @staticmethod
    def _setup_hybrid_components(ga_config: GAConfig) -> tuple:
        """
        GAとMLを組み合わせた「ハイブリッドモード」に必要な予測・変換コンポーネントを準備します。

        このメソッドは、最新の学習済みモデルを自動的に探索・ロードし、
        GAの評価プロセス（バックテスト内）で使用される予測器を構成します。

        Args:
            ga_config (GAConfig): ハイブリッド設定（使用モデル、アンサンブル構成等）を含む設定。

        Returns:
            tuple[HybridPredictor, HybridFeatureAdapter]: 
                (構成済みの予測器, 特徴量エンジニアリング用アダプター) のタプル。
        """
        from ..hybrid.hybrid_feature_adapter import HybridFeatureAdapter
        from ..hybrid.hybrid_predictor import HybridPredictor

        logger.info("🔬 ハイブリッドGA+MLモードのコンポーネントを準備中")

        # 予測器の初期化
        hybrid_config = ga_config.hybrid_config
        model_types = hybrid_config.model_types
        if model_types and len(model_types) > 1:
            logger.info(f"複数モデルアンサンブルを使用: {model_types}")
            predictor = HybridPredictor(trainer_type="single", model_types=model_types)
        else:
            model_type = hybrid_config.model_type or "lightgbm"
            logger.info(f"単一モデルを使用: {model_type}")
            predictor = HybridPredictor(trainer_type="single", model_type=model_type)

        try:
            if predictor.load_latest_models():
                logger.info("ハイブリッド予測器に最新モデルをロードしました")
            else:
                logger.info(
                    "ロード可能な最新モデルが見つからないため中立予測で継続します"
                )
        except Exception as exc:
            logger.warning(
                f"最新モデルのロードに失敗したため中立予測で継続します: {exc}"
            )

        # 特徴量アダプタの初期化
        adapter = HybridFeatureAdapter()

        return predictor, adapter
