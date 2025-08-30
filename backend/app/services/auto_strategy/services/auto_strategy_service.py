"""
自動戦略生成サービス

GA実行、進捗管理、結果保存、戦略テストを統合的に管理します。
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks

from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.backtest_service import BacktestService
from database.connection import SessionLocal
from app.services.symbol.normalization_service import (
    normalize_symbol as normalize_symbol_unified,
)

from .experiment_manager import ExperimentManager
from ..config import GAConfig, get_default_config
from .experiment_persistence_service import ExperimentPersistenceService

logger = logging.getLogger(__name__)


class AutoStrategyService:
    """
    自動戦略生成サービス

    GA実行、進捗管理、結果保存、戦略テストを統合的に管理します。
    """

    def __init__(self, enable_smart_generation: bool = True):
        """
        初期化

        Args:
            enable_smart_generation: SmartConditionGeneratorを使用するか
        """
        # データベースセッションファクトリ
        self.db_session_factory = SessionLocal
        self.enable_smart_generation = enable_smart_generation

        # サービスの初期化
        self.backtest_service: BacktestService
        self.persistence_service: ExperimentPersistenceService

        # 管理マネージャー
        self.experiment_manager: Optional[ExperimentManager] = None

        self._init_services()

    def _init_services(self):
        """
        サービスの初期化

        必要最小限のサービス初期化を行います。
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="サービス初期化", is_api_call=False)
        def _init_services_impl():
            # データベースリポジトリの初期化
            with self.db_session_factory() as db:
                from database.repositories.funding_rate_repository import (
                    FundingRateRepository,
                )
                from database.repositories.ohlcv_repository import OHLCVRepository
                from database.repositories.open_interest_repository import (
                    OpenInterestRepository,
                )

                ohlcv_repo = OHLCVRepository(db)
                oi_repo = OpenInterestRepository(db)
                fr_repo = FundingRateRepository(db)

                # バックテストサービスの初期化
                data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
                self.backtest_service = BacktestService(data_service)

            # 永続化サービスの初期化
            self.persistence_service = ExperimentPersistenceService(
                self.db_session_factory, self.backtest_service
            )

            # 実験管理マネージャーの初期化
            self.experiment_manager = ExperimentManager(
                self.backtest_service, self.persistence_service
            )

        _init_services_impl()

    def start_strategy_generation(
        self,
        experiment_id: str,
        experiment_name: str,
        ga_config_dict: Dict[str, Any],
        backtest_config_dict: Dict[str, Any],
        background_tasks: BackgroundTasks,
    ) -> str:
        """
        戦略生成を開始

        Args:
            experiment_id: 実験ID（フロントエンドで生成されたUUID）
            experiment_name: 実験名
            ga_config_dict: GA設定の辞書
            backtest_config_dict: バックテスト設定の辞書
            background_tasks: FastAPIのバックグラウンドタスク

        Returns:
            実験ID（入力されたものと同じ）
        """
        logger.info(f"戦略生成開始: {experiment_name}")

        # 1. GA設定の構築と検証
        from app.utils.error_handler import safe_operation

        @safe_operation(context="GA設定構築と検証", is_api_call=True)
        def _validate_ga_config():
            ga_config = GAConfig.from_dict(ga_config_dict)
            is_valid, errors = ga_config.validate()
            if not is_valid:
                raise ValueError(f"無効なGA設定です: {', '.join(errors)}")
            return ga_config

        ga_config = _validate_ga_config()

        # 2. バックテスト設定のシンボル正規化
        backtest_config = backtest_config_dict.copy()
        backtest_config["symbol"] = normalize_symbol_unified(
            backtest_config.get("symbol"), "generic"
        )

        # 3. 実験を作成（統合版）
        # フロントエンドから送信されたexperiment_idを使用
        self.persistence_service.create_experiment(
            experiment_id, experiment_name, ga_config, backtest_config
        )

        # 4. GAエンジンを初期化
        if not self.experiment_manager:
            raise RuntimeError("実験管理マネージャーが初期化されていません。")
        self.experiment_manager.initialize_ga_engine(ga_config)

        # 5. 実験をバックグラウンドで開始
        background_tasks.add_task(
            self.experiment_manager.run_experiment,
            experiment_id,
            ga_config,
            backtest_config,
        )

        logger.info(
            f"戦略生成実験のバックグラウンドタスクを追加しました: {experiment_id}"
        )

        return experiment_id

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        実験一覧を取得

        Returns:
            実験一覧のリスト
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験一覧取得", is_api_call=False, default_return=[])
        def _list_experiments():
            return self.persistence_service.list_experiments()

        return _list_experiments()

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        実験ステータスを取得

        Args:
            experiment_id: 実験ID

        Returns:
            実験ステータス情報
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="実験ステータス取得",
            is_api_call=False,
            default_return={"status": "error", "message": "不明なエラー"},
        )
        def _get_experiment_status():
            # get_experiment_infoメソッドを使用
            experiment_info = self.persistence_service.get_experiment_info(
                experiment_id
            )
            if experiment_info:
                return {
                    "status": experiment_info.get("status", "unknown"),
                    "progress": experiment_info.get("progress", 0),
                    "message": experiment_info.get("message", "実験情報を取得しました"),
                    "experiment_info": experiment_info,
                }
            else:
                return {"status": "not_found", "message": "実験が見つかりません"}

        return _get_experiment_status()

    def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        実験を停止

        Args:
            experiment_id: 実験ID

        Returns:
            停止結果
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="実験停止",
            is_api_call=False,
            default_return={
                "success": False,
                "message": "実験停止でエラーが発生しました",
            },
        )
        def _stop_experiment():
            if self.experiment_manager:
                # ExperimentManager.stop_experiment()はboolを返すので、Dict形式に変換
                stop_result = self.experiment_manager.stop_experiment(experiment_id)
                if stop_result:
                    return {
                        "success": True,
                        "message": "実験が正常に停止されました",
                    }
                else:
                    return {
                        "success": False,
                        "message": "実験の停止に失敗しました",
                    }
            else:
                return {
                    "success": False,
                    "message": "実験管理マネージャーが初期化されていません",
                }

        return _stop_experiment()

    def get_default_config(self) -> Dict[str, Any]:
        """
        デフォルト設定を取得（AutoStrategyConfig統合版）

        Returns:
            デフォルト設定
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="デフォルト設定取得", is_api_call=False, default_return={}
        )
        def _get_default_config():
            # AutoStrategyConfigからデフォルトGA設定を作成
            auto_config = get_default_config()
            ga_config = GAConfig.from_auto_strategy_config(auto_config)
            return ga_config.to_dict()

        return _get_default_config()

    def get_presets(self) -> Dict[str, Any]:
        """
        プリセット設定を取得（AutoStrategyConfig統合版）

        Returns:
            プリセット設定
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="プリセット取得", is_api_call=False, default_return={})
        def _get_presets():
            # AutoStrategyConfigから各プリセットを作成
            auto_config = get_default_config()

            presets = {}

            # defaultプリセット
            ga_base = GAConfig.from_auto_strategy_config(auto_config)
            presets["default"] = ga_base.to_dict()

            # fastプリセット
            ga_fast = GAConfig.from_auto_strategy_config(auto_config)
            ga_fast.population_size = 10
            ga_fast.generations = 5
            ga_fast.elite_size = 2
            ga_fast.max_indicators = 3
            presets["fast"] = ga_fast.to_dict()

            # thoroughプリセット
            ga_thorough = GAConfig.from_auto_strategy_config(auto_config)
            ga_thorough.population_size = 200
            ga_thorough.generations = 100
            ga_thorough.crossover_rate = 0.85
            ga_thorough.mutation_rate = 0.05
            ga_thorough.elite_size = 20
            ga_thorough.max_indicators = 5
            presets["thorough"] = ga_thorough.to_dict()

            # multi_objectiveプリセット
            ga_multi = GAConfig.from_auto_strategy_config(auto_config)
            ga_multi.population_size = 50
            ga_multi.generations = 30
            ga_multi.enable_multi_objective = True
            ga_multi.objectives = ["total_return", "max_drawdown"]
            ga_multi.objective_weights = [1.0, -1.0]
            ga_multi.max_indicators = 3
            presets["multi_objective"] = ga_multi.to_dict()

            return presets

        return _get_presets()

    async def test_strategy(self, request) -> Dict[str, Any]:
        """
        単一戦略のテスト実行

        指定された戦略遺伝子から戦略を生成し、バックテストを実行します。

        Args:
            request: StrategyTestRequest

        Returns:
            テスト結果
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="戦略テスト",
            is_api_call=False,
            default_return={
                "success": False,
                "errors": ["戦略テストでエラーが発生しました"],
                "message": "戦略テストに失敗しました",
            },
        )
        def _test_strategy():
            logger.info("戦略テスト開始")

            # 戦略遺伝子の復元
            from ..serializers.gene_serialization import GeneSerializer
            from ..models.strategy_models import StrategyGene

            gene_serializer = GeneSerializer()
            strategy_gene = gene_serializer.dict_to_strategy_gene(
                request.strategy_gene, StrategyGene
            )

            # バックテスト設定の正規化
            backtest_config = request.backtest_config.copy()
            backtest_config["symbol"] = normalize_symbol_unified(
                backtest_config.get("symbol"), "generic"
            )

            # 戦略遺伝子からバックテスト設定を作成
            from ..generators.strategy_factory import StrategyFactory

            strategy_factory = StrategyFactory()
            strategy_class = strategy_factory.create_strategy_class(strategy_gene)

            backtest_full_config = {
                **backtest_config,
                "strategy_name": f"test_strategy_{strategy_gene.id[:8]}",
                "strategy_class": strategy_class,
                "strategy_config": strategy_gene.to_dict(),
            }

            # バックテスト実行
            result = self.backtest_service.run_backtest(backtest_full_config)

            logger.info("戦略テスト完了")
            return {
                "success": True,
                "result": result,
                "message": "戦略テストが正常に完了しました",
            }

        return _test_strategy()
