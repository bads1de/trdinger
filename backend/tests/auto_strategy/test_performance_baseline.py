"""
自動戦略生成機能のパフォーマンス・ベースライン測定

既存のBacktestServiceを使用して、代表的な戦略の実行時間を測定し、
GA全体の実行時間を現実的に見積もります。
"""

import time
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.connection import SessionLocal


class TestPerformanceBaseline:
    """パフォーマンス・ベースライン測定テスト"""

    @pytest.fixture
    def backtest_service(self):
        """BacktestServiceのセットアップ"""
        db = SessionLocal()
        try:
            ohlcv_repo = OHLCVRepository(db)
            data_service = BacktestDataService(ohlcv_repo)
            return BacktestService(data_service)
        finally:
            db.close()

    def test_sma_cross_performance_baseline(self, backtest_service):
        """SMAクロス戦略のパフォーマンス・ベースライン測定"""
        
        # テスト期間の設定（複数期間で測定）
        test_periods = [
            # 短期間（1ヶ月）
            {
                "name": "1_month",
                "start_date": "2024-11-01",
                "end_date": "2024-12-01",
            },
            # 中期間（3ヶ月）
            {
                "name": "3_months", 
                "start_date": "2024-09-01",
                "end_date": "2024-12-01",
            },
            # 長期間（6ヶ月）
            {
                "name": "6_months",
                "start_date": "2024-06-01", 
                "end_date": "2024-12-01",
            },
        ]

        # 基本設定
        base_config = {
            "strategy_name": "PERFORMANCE_BASELINE_SMA_CROSS",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 100000,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50}
            }
        }

        performance_results = {}

        for period in test_periods:
            print(f"\n=== {period['name']} 期間のパフォーマンス測定 ===")
            
            # 設定を更新
            config = base_config.copy()
            config.update({
                "start_date": period["start_date"],
                "end_date": period["end_date"]
            })

            # 実行時間を測定
            start_time = time.time()
            
            try:
                result = backtest_service.run_backtest(config)
                execution_time = time.time() - start_time
                
                # 結果を記録
                performance_results[period["name"]] = {
                    "execution_time": execution_time,
                    "success": True,
                    "total_return": result.get("performance_metrics", {}).get("total_return", 0),
                    "total_trades": result.get("performance_metrics", {}).get("total_trades", 0),
                    "data_points": len(result.get("equity_curve", [])),
                }
                
                print(f"実行時間: {execution_time:.2f}秒")
                print(f"データポイント数: {performance_results[period['name']]['data_points']}")
                print(f"総取引数: {performance_results[period['name']]['total_trades']}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                performance_results[period["name"]] = {
                    "execution_time": execution_time,
                    "success": False,
                    "error": str(e)
                }
                print(f"エラー発生: {e}")

        # GA実行時間の見積もり
        self._estimate_ga_execution_time(performance_results)

        # 結果をアサート（基本的な妥当性チェック）
        for period_name, result in performance_results.items():
            if result["success"]:
                # 実行時間が合理的な範囲内であることを確認
                assert result["execution_time"] < 60, f"{period_name}: 実行時間が60秒を超えています"
                assert result["data_points"] > 0, f"{period_name}: データポイントが0です"

    def _estimate_ga_execution_time(self, baseline_results: Dict[str, Any]):
        """GA実行時間の見積もり"""
        print("\n=== GA実行時間見積もり ===")
        
        # 基準となる実行時間（3ヶ月期間を使用）
        if "3_months" in baseline_results and baseline_results["3_months"]["success"]:
            single_backtest_time = baseline_results["3_months"]["execution_time"]
            
            # GA設定例
            population_size = 100
            generations = 50
            total_evaluations = population_size * generations
            
            # 並列処理を考慮（CPUコア数を仮定）
            import multiprocessing
            cpu_cores = multiprocessing.cpu_count()
            parallel_efficiency = 0.7  # 並列処理効率
            
            # 見積もり計算
            sequential_time = total_evaluations * single_backtest_time
            parallel_time = sequential_time / (cpu_cores * parallel_efficiency)
            
            print(f"単一バックテスト時間: {single_backtest_time:.2f}秒")
            print(f"総評価回数: {total_evaluations}")
            print(f"CPUコア数: {cpu_cores}")
            print(f"逐次実行時間見積もり: {sequential_time/3600:.1f}時間")
            print(f"並列実行時間見積もり: {parallel_time/60:.1f}分")
            
            # 現実的な推奨値
            recommended_population = 50
            recommended_generations = 30
            recommended_evaluations = recommended_population * recommended_generations
            recommended_time = (recommended_evaluations * single_backtest_time) / (cpu_cores * parallel_efficiency)
            
            print(f"\n推奨設定:")
            print(f"個体数: {recommended_population}, 世代数: {recommended_generations}")
            print(f"推定実行時間: {recommended_time/60:.1f}分")
            
        else:
            print("ベースライン測定に失敗したため、見積もりを計算できません")


if __name__ == "__main__":
    # 直接実行用
    test = TestPerformanceBaseline()
    
    # BacktestServiceのセットアップ
    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        backtest_service = BacktestService(data_service)
        
        # テスト実行
        test.test_sma_cross_performance_baseline(backtest_service)
        
    finally:
        db.close()
