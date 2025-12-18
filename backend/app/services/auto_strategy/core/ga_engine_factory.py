"""
GAエンジンファクトリ

GAエンジンの構築とコンポーネントの初期化を担当します。
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.backtest.backtest_service import BacktestService
from .ga_engine import GeneticAlgorithmEngine
from ..config.ga import GAConfig
from ..generators.random_gene_generator import RandomGeneGenerator

logger = logging.getLogger(__name__)


class GeneticAlgorithmEngineFactory:
    """
    GAエンジンの構築を行うファクトリクラス
    """

    @staticmethod
    def create_engine(
        backtest_service: BacktestService, ga_config: GAConfig
    ) -> GeneticAlgorithmEngine:
        """
        設定に基づいてGAエンジンを作成

        Args:
            backtest_service: バックテストサービス
            ga_config: GA設定

        Returns:
            構築済みのGeneticAlgorithmEngine
        """
        # ログレベルの設定
        auto_strategy_logger = logging.getLogger("app.services.auto_strategy")
        auto_strategy_logger.setLevel(getattr(logging, ga_config.log_level.upper(), logging.INFO))

        # 遺伝子生成器の初期化
        gene_generator = RandomGeneGenerator(ga_config)

        # ハイブリッドコンポーネントの準備
        hybrid_predictor = None
        hybrid_feature_adapter = None

        if ga_config.hybrid_mode:
            hybrid_predictor, hybrid_feature_adapter = GeneticAlgorithmEngineFactory._setup_hybrid_components(ga_config)

        # エンジンの生成
        engine = GeneticAlgorithmEngine(
            backtest_service=backtest_service,
            gene_generator=gene_generator,
            hybrid_mode=ga_config.hybrid_mode,
            hybrid_predictor=hybrid_predictor,
            hybrid_feature_adapter=hybrid_feature_adapter,
        )

        logger.info(f"GAエンジンを初期化しました (Mode: {'Hybrid' if ga_config.hybrid_mode else 'Standard'})")
        return engine

    @staticmethod
    def _setup_hybrid_components(ga_config: GAConfig) -> tuple:
        """ハイブリッドモード用コンポーネントのセットアップ"""
        from .hybrid_predictor import HybridPredictor
        from .hybrid_feature_adapter import HybridFeatureAdapter

        logger.info("🔬 ハイブリッドGA+MLモードのコンポーネントを準備中")
        
        # 予測器の初期化
        model_types = ga_config.hybrid_model_types
        if model_types and len(model_types) > 1:
            logger.info(f"複数モデルアンサンブルを使用: {model_types}")
            predictor = HybridPredictor(trainer_type="single", model_types=model_types)
        else:
            model_type = ga_config.hybrid_model_type or "lightgbm"
            logger.info(f"単一モデルを使用: {model_type}")
            predictor = HybridPredictor(trainer_type="single", model_type=model_type)

        # 特徴量アダプタの初期化
        adapter = HybridFeatureAdapter()

        return predictor, adapter
