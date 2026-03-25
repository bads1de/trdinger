"""
GAエンジンファクトリ

GAエンジンの構築とコンポーネントの初期化を担当します。
"""

import logging


from app.services.backtest.backtest_service import BacktestService
from .ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator

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
        GA 設定および各種サービスに基づき、GA エンジンを構築

        ロガーの初期化、遺伝子生成器のセットアップ、さらにハイブリッドモード
        （ML との併用）が有効な場合は予測器やアダプターの準備を行い、
        依存関係が注入された `GeneticAlgorithmEngine` を返します。

        Args:
            backtest_service: バックテストの実行を担うサービス
            ga_config: GA の世代交代数、報酬設計、モデル構成等の設定

        Returns:
            初期化済みの GeneticAlgorithmEngine インスタンス
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

        if ga_config.hybrid_mode:
            hybrid_predictor, hybrid_feature_adapter = (
                GeneticAlgorithmEngineFactory._setup_hybrid_components(ga_config)
            )

        # エンジンの生成
        engine = GeneticAlgorithmEngine(
            backtest_service=backtest_service,
            gene_generator=gene_generator,
            hybrid_mode=ga_config.hybrid_mode,
            hybrid_predictor=hybrid_predictor,
            hybrid_feature_adapter=hybrid_feature_adapter,
        )

        logger.info(
            f"GAエンジンを初期化しました (Mode: {'Hybrid' if ga_config.hybrid_mode else 'Standard'})"
        )
        return engine

    @staticmethod
    def _setup_hybrid_components(ga_config: GAConfig) -> tuple:
        """
        ハイブリッドモード（GA+ML）用コンポーネントのセットアップ

        GA設定に基づき、予測モデル（HybridPredictor）と
        特徴量アダプター（HybridFeatureAdapter）を初期化します。
        複数モデルのアンサンブルや、単一モデルの選択に対応しています。

        Args:
            ga_config: GA設定情報

        Returns:
            (predictor, adapter) のタプル
        """
        from ..hybrid.hybrid_predictor import HybridPredictor
        from ..hybrid.hybrid_feature_adapter import HybridFeatureAdapter

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
