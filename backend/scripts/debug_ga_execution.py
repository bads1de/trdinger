"""
GA実行の問題を調査するスクリプト

なぜGA実験が途中で停止しているのかを詳しく調査します。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
from app.core.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.core.services.auto_strategy.models.ga_config import GAConfig
import json
from datetime import datetime, timedelta
import logging

# ログレベルを設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_experiment_status():
    """実験ステータスの詳細確認"""
    print("=== GA実験ステータス詳細確認 ===")

    db = SessionLocal()
    try:
        exp_repo = GAExperimentRepository(db)

        # 最近の実験を取得
        recent_experiments = exp_repo.get_recent_experiments(limit=5)

        for exp in recent_experiments:
            print(f"\n実験: {exp.name}")
            print(f"  ID: {exp.id}")
            print(f"  ステータス: {exp.status}")
            print(f"  進捗: {exp.progress:.2%}")
            print(f"  現在世代: {exp.current_generation}/{exp.total_generations}")
            print(f"  作成日時: {exp.created_at}")
            print(f"  完了日時: {exp.completed_at}")
            if hasattr(exp, "error_message"):
                print(f"  エラーメッセージ: {exp.error_message}")
            else:
                print(f"  エラーメッセージ: なし")

            # 設定の詳細
            if exp.config:
                config = exp.config
                print(f"  設定:")
                print(f"    個体数: {config.get('population_size', 'N/A')}")
                print(f"    世代数: {config.get('generations', 'N/A')}")
                print(f"    交叉率: {config.get('crossover_rate', 'N/A')}")
                print(f"    突然変異率: {config.get('mutation_rate', 'N/A')}")

            # バックテスト設定
            if hasattr(exp, "backtest_config") and exp.backtest_config:
                bt_config = exp.backtest_config
                print(f"  バックテスト設定:")
                print(f"    シンボル: {bt_config.get('symbol', 'N/A')}")
                print(
                    f"    期間: {bt_config.get('start_date', 'N/A')} - {bt_config.get('end_date', 'N/A')}"
                )
                print(f"    初期資金: {bt_config.get('initial_capital', 'N/A')}")
            else:
                print(f"  バックテスト設定: 利用不可")

    finally:
        db.close()


def test_simple_ga_execution():
    """シンプルなGA実行テスト"""
    print("\n=== シンプルなGA実行テスト ===")

    try:
        # AutoStrategyServiceを初期化
        print("AutoStrategyServiceを初期化中...")
        service = AutoStrategyService()

        # テスト用のGA設定（正しい初期化方法）
        from app.core.services.auto_strategy.models.ga_config import (
            EvolutionConfig,
            IndicatorConfig,
            GeneGenerationConfig,
        )

        ga_config = GAConfig(
            evolution=EvolutionConfig(
                population_size=5,  # 小さな個体数でテスト
                generations=2,  # 少ない世代数でテスト
                crossover_rate=0.8,
                mutation_rate=0.2,
            ),
            indicators=IndicatorConfig(
                allowed_indicators=["RSI", "SMA", "CCI"],  # 制限された指標
                max_indicators=3,
            ),
            gene_generation=GeneGenerationConfig(
                numeric_threshold_probability=0.8,  # 80%の確率で数値を使用
                min_compatibility_score=0.8,  # 最小互換性スコア
                strict_compatibility_score=0.9,  # 厳密な互換性スコア
            ),
        )

        # テスト用のバックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-12-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
        }

        print("GA実行を開始...")

        # BackgroundTasksのモック作成
        class MockBackgroundTasks:
            def add_task(self, func, *args, **kwargs):
                # 実際にはバックグラウンドで実行せず、直接実行
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"バックグラウンドタスクエラー: {e}")

        mock_tasks = MockBackgroundTasks()

        experiment_id = service.start_strategy_generation(
            experiment_name=f"DEBUG_TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ga_config_dict=ga_config.to_dict(),  # 辞書形式に変換
            backtest_config_dict=backtest_config,
            background_tasks=mock_tasks,
        )

        print(f"実験ID: {experiment_id}")

        # 進捗を監視
        print("進捗監視中...")
        import time

        max_wait = 120  # 2分間待機
        start_time = time.time()

        while time.time() - start_time < max_wait:
            progress = service.get_experiment_progress(experiment_id)
            if progress:
                print(
                    f"  進捗: {progress.progress:.1%}, 世代: {progress.current_generation}/{progress.total_generations}"
                )

                if progress.status == "completed":
                    print("✅ GA実行完了")

                    # 結果を取得
                    result = service.get_experiment_result(experiment_id)
                    if result:
                        print(f"最高フィットネス: {result['best_fitness']}")
                        print(f"実行時間: {result['execution_time']:.2f}秒")
                    break
                elif progress.status == "failed":
                    print(f"❌ GA実行失敗: {progress.error_message}")
                    break

            time.sleep(5)  # 5秒間隔で確認
        else:
            print("⏰ タイムアウト: GA実行が完了しませんでした")

            # 最終状態を確認
            final_progress = service.get_experiment_progress(experiment_id)
            if final_progress:
                print(f"最終状態: {final_progress.status}")
                print(f"最終進捗: {final_progress.progress:.1%}")
                if final_progress.error_message:
                    print(f"エラー: {final_progress.error_message}")

    except Exception as e:
        print(f"❌ GA実行テストエラー: {e}")
        logger.exception("GA実行テスト中にエラーが発生")


def check_dependencies():
    """依存関係の確認"""
    print("\n=== 依存関係確認 ===")

    try:
        # データベース接続確認
        print("データベース接続確認...")
        db = SessionLocal()
        try:
            # 簡単なクエリを実行
            from sqlalchemy import text

            result = db.execute(text("SELECT 1")).fetchone()
            print("✅ データベース接続OK")
        finally:
            db.close()

        # 必要なサービスの初期化確認
        print("AutoStrategyService初期化確認...")
        service = AutoStrategyService()
        print("✅ AutoStrategyService初期化OK")

        # バックテストサービス確認
        if hasattr(service, "backtest_service") and service.backtest_service:
            print("✅ BacktestService OK")
        else:
            print("❌ BacktestService 初期化失敗")

        # GAエンジン確認
        if hasattr(service, "ga_engine") and service.ga_engine:
            print("✅ GeneticAlgorithmEngine OK")
        else:
            print("❌ GeneticAlgorithmEngine 初期化失敗")

    except Exception as e:
        print(f"❌ 依存関係確認エラー: {e}")
        logger.exception("依存関係確認中にエラーが発生")


def check_data_availability():
    """データ可用性の確認"""
    print("\n=== データ可用性確認 ===")

    try:
        from database.repositories.ohlcv_repository import OHLCVRepository
        from database.repositories.open_interest_repository import (
            OpenInterestRepository,
        )
        from database.repositories.funding_rate_repository import FundingRateRepository

        db = SessionLocal()
        try:
            ohlcv_repo = OHLCVRepository(db)
            oi_repo = OpenInterestRepository(db)
            fr_repo = FundingRateRepository(db)

            # OHLCV データ確認
            ohlcv_count = ohlcv_repo.count_records("BTC/USDT:USDT", "1h")
            print(f"OHLCV データ数: {ohlcv_count}")

            if ohlcv_count > 0:
                latest_ohlcv = ohlcv_repo.get_latest_ohlcv_data("BTC/USDT:USDT", "1h")
                print(
                    f"最新OHLCV: {latest_ohlcv.timestamp if latest_ohlcv else 'None'}"
                )

            # OI データ確認
            oi_count = oi_repo.count_records("BTC/USDT:USDT")
            print(f"OI データ数: {oi_count}")

            # FR データ確認
            fr_count = fr_repo.count_records("BTC/USDT:USDT")
            print(f"FR データ数: {fr_count}")

            if ohlcv_count == 0:
                print("❌ OHLCVデータが不足しています")
            else:
                print("✅ 基本データは利用可能です")

        finally:
            db.close()

    except Exception as e:
        print(f"❌ データ可用性確認エラー: {e}")
        logger.exception("データ可用性確認中にエラーが発生")


def main():
    """メイン実行関数"""
    print("🔍 GA実行問題調査開始")
    print(f"実行時刻: {datetime.now()}")

    # 1. 実験ステータス確認
    check_experiment_status()

    # 2. 依存関係確認
    check_dependencies()

    # 3. データ可用性確認
    check_data_availability()

    # 4. シンプルなGA実行テスト
    test_simple_ga_execution()

    print(f"\n🔍 調査完了")


if __name__ == "__main__":
    main()
