"""
自動戦略生成サービス
GA実行、進捗管理、結果保存を統合的に管理します。
複雑な分離構造を削除し、直接的で理解しやすい実装に変更しました。
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks

from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.backtest_service import BacktestService
from database.connection import SessionLocal

from ..managers.experiment_manager import ExperimentManager
from ..models.ga_config import GAConfig
from ..models.gene_strategy import StrategyGene
from ..persistence.experiment_persistence_service import ExperimentPersistenceService

logger = logging.getLogger(__name__)


class AutoStrategyService:
    """
    自動戦略生成サービス

    GA実行、進捗管理、結果保存を統合的に管理します。
    """

    # NOTE: Orchestration層との責務境界
    # - 本サービスは依存初期化とGA実験の開始/取得/停止の薄いファサードに限定
    # - 戦略テストや結果整形、停止時バリデーションは orchestration に集約
    # TODO(phase2): orchestration に寄せられる一部関数の更なる縮退検討

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
        複雑な依存関係管理を削除し、シンプルな実装に変更しました。
        """
        try:
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

        except Exception:
            raise

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
        try:
            ga_config = GAConfig.from_dict(ga_config_dict)
            is_valid, errors = ga_config.validate()
            if not is_valid:
                raise ValueError(f"無効なGA設定です: {', '.join(errors)}")
        except Exception as e:
            raise ValueError(f"GA設定の構築または検証に失敗しました: {e}")

        # 2. バックテスト設定のシンボル正規化
        backtest_config = backtest_config_dict.copy()
        original_symbol = backtest_config.get("symbol")
        if original_symbol and ":" not in original_symbol:
            normalized_symbol = f"{original_symbol}:USDT"
            backtest_config["symbol"] = normalized_symbol

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

    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """実験結果を取得（統合版）"""
        return self.persistence_service.get_experiment_result(experiment_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """実験一覧を取得（統合版）"""
        return self.persistence_service.list_experiments()

    def stop_experiment(self, experiment_id: str) -> bool:
        """実験を停止"""
        if not self.experiment_manager:
            return False
        return self.experiment_manager.stop_experiment(experiment_id)

    def validate_strategy_gene(self, gene: StrategyGene) -> tuple[bool, List[str]]:
        """
        戦略遺伝子の妥当性を検証

        Args:
            gene: 検証する戦略遺伝子

        Returns:
            (is_valid, error_messages)
        """
        if not self.experiment_manager:
            return False, ["実験管理マネージャーが初期化されていません。"]
        return self.experiment_manager.validate_strategy_gene(gene)

    def test_strategy_generation(
        self, gene: StrategyGene, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        単一戦略のテスト生成・実行

        Args:
            gene: テストする戦略遺伝子
            backtest_config: バックテスト設定

        Returns:
            テスト結果
        """
        if not self.experiment_manager:
            return {
                "success": False,
                "error": "実験管理マネージャーが初期化されていません。",
            }
        return self.experiment_manager.test_strategy_generation(gene, backtest_config)
