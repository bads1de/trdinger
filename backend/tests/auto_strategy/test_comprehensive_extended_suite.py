"""
オートストラテジー包括的拡張テストスイート

全39個のテストを統合実行し、システムの本番運用準備状況を評価します。
"""

import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

import logging
import time
import psutil
from datetime import datetime
import traceback
from typing import Dict, List, Any

# テストモジュールをインポート
from tests.auto_strategy.test_auto_strategy_comprehensive import TestAutoStrategyComprehensive
from tests.auto_strategy.test_auto_strategy_advanced import TestAutoStrategyAdvanced
from tests.auto_strategy.test_auto_strategy_integration import TestAutoStrategyIntegration
from tests.auto_strategy.test_edge_cases import TestEdgeCases
from tests.auto_strategy.test_integration_scenarios import TestIntegrationScenarios
from tests.auto_strategy.test_precision_quality import TestPrecisionQuality
from tests.auto_strategy.test_performance import TestPerformance

logger = logging.getLogger(__name__)


class ComprehensiveExtendedTestSuite:
    """包括的拡張テストスイート"""
    
    def __init__(self):
        """初期化"""
        self.start_time = None
        self.end_time = None
        self.total_tests = 39
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
        self.system_metrics = {
            "start_memory": 0,
            "peak_memory": 0,
            "end_memory": 0,
            "start_cpu": 0,
            "peak_cpu": 0,
            "end_cpu": 0
        }
    
    def run_all_tests(self):
        """全てのテストを実行"""
        logger.info("🚀 オートストラテジー包括的拡張テストスイート開始")
        logger.info("=" * 100)
        logger.info("MLとオートストラテジーの完全連携・計算精度・本番運用準備状況の包括的検証")
        logger.info("=" * 100)
        
        self.start_time = time.time()
        self._record_system_metrics("start")
        
        # 各カテゴリのテストを実行
        self._run_basic_comprehensive_tests()
        self._run_advanced_tests()
        self._run_integration_tests()
        self._run_edge_case_tests()
        self._run_integration_scenario_tests()
        self._run_precision_quality_tests()
        self._run_performance_tests()
        
        self.end_time = time.time()
        self._record_system_metrics("end")
        self._display_comprehensive_summary()
        
        return self.passed_tests == self.total_tests
    
    def _record_system_metrics(self, phase: str):
        """システムメトリクスを記録"""
        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if phase == "start":
                self.system_metrics["start_memory"] = memory_mb
                self.system_metrics["start_cpu"] = cpu_percent
                self.system_metrics["peak_memory"] = memory_mb
                self.system_metrics["peak_cpu"] = cpu_percent
            elif phase == "end":
                self.system_metrics["end_memory"] = memory_mb
                self.system_metrics["end_cpu"] = cpu_percent
            
            # ピーク値を更新
            if memory_mb > self.system_metrics["peak_memory"]:
                self.system_metrics["peak_memory"] = memory_mb
            if cpu_percent > self.system_metrics["peak_cpu"]:
                self.system_metrics["peak_cpu"] = cpu_percent
                
        except Exception as e:
            logger.warning(f"システムメトリクス記録エラー: {e}")
    
    def _run_basic_comprehensive_tests(self):
        """基本包括的テスト実行（テスト1-10）"""
        logger.info("\n📋 カテゴリ1: 基本包括的テスト（基本機能・計算精度）")
        logger.info("-" * 80)
        
        test_instance = TestAutoStrategyComprehensive()
        
        tests = [
            ("テスト1: MLとオートストラテジーの統合", test_instance.test_ml_auto_strategy_integration),
            ("テスト2: TP/SL計算精度", test_instance.test_tpsl_calculation_accuracy),
            ("テスト3: TP/SL自動決定サービス", test_instance.test_tpsl_auto_decision_service),
            ("テスト4: バックテスト統合", test_instance.test_backtest_integration),
            ("テスト5: 戦略遺伝子検証", test_instance.test_strategy_gene_validation),
            ("テスト6: オートストラテジー統合管理", test_instance.test_auto_strategy_orchestration),
            ("テスト7: ML予測精度", test_instance.test_ml_prediction_accuracy),
            ("テスト8: リスク管理計算", test_instance.test_risk_management_calculations),
            ("テスト9: 戦略パフォーマンスメトリクス", test_instance.test_strategy_performance_metrics),
            ("テスト10: データ検証パイプライン", test_instance.test_data_validation_pipeline),
        ]
        
        self._execute_test_category(test_instance, tests)
    
    def _run_advanced_tests(self):
        """高度テスト実行（テスト11-15）"""
        logger.info("\n🔬 カテゴリ2: 高度テスト（GA最適化・パフォーマンス）")
        logger.info("-" * 80)
        
        test_instance = TestAutoStrategyAdvanced()
        
        tests = [
            ("テスト11: GA最適化統合", test_instance.test_ga_optimization_integration),
            ("テスト12: 並行戦略実行", test_instance.test_concurrent_strategy_execution),
            ("テスト13: 極端な市場条件", test_instance.test_extreme_market_conditions),
            ("テスト14: メモリ・パフォーマンス最適化", test_instance.test_memory_performance_optimization),
            ("テスト15: エラー回復・復元力", test_instance.test_error_recovery_resilience),
        ]
        
        self._execute_test_category(test_instance, tests)
    
    def _run_integration_tests(self):
        """統合テスト実行（テスト16-20）"""
        logger.info("\n🔗 カテゴリ3: 統合テスト（エンドツーエンド・API連携）")
        logger.info("-" * 80)
        
        test_instance = TestAutoStrategyIntegration()
        
        tests = [
            ("テスト16: エンドツーエンド戦略生成", test_instance.test_end_to_end_strategy_generation),
            ("テスト17: ML-オートストラテジー完全パイプライン", test_instance.test_ml_auto_strategy_full_pipeline),
            ("テスト18: API統合シミュレーション", test_instance.test_api_integration_simulation),
            ("テスト19: データフロー一貫性", test_instance.test_data_flow_consistency),
            ("テスト20: 設定検証", test_instance.test_configuration_validation),
        ]
        
        self._execute_test_category(test_instance, tests)
    
    def _run_edge_case_tests(self):
        """エッジケーステスト実行（テスト21-25）"""
        logger.info("\n⚠️ カテゴリ4: エッジケーステスト（極端条件・境界値）")
        logger.info("-" * 80)
        
        test_instance = TestEdgeCases()
        
        tests = [
            ("テスト21: 極小データセット処理", test_instance.test_minimal_dataset_processing),
            ("テスト22: フラット価格データ処理", test_instance.test_flat_price_processing),
            ("テスト23: 高欠損率データ処理", test_instance.test_high_missing_data_processing),
            ("テスト24: 極端ボラティリティ処理", test_instance.test_extreme_volatility_processing),
            ("テスト25: 極端TP/SL設定処理", test_instance.test_extreme_tpsl_settings),
        ]
        
        self._execute_test_category(test_instance, tests)
    
    def _run_integration_scenario_tests(self):
        """統合シナリオテスト実行（テスト26-30）"""
        logger.info("\n🔄 カテゴリ5: 統合シナリオテスト（複雑統合・長時間実行）")
        logger.info("-" * 80)
        
        test_instance = TestIntegrationScenarios()
        
        tests = [
            ("テスト26: 並行戦略実行競合", test_instance.test_concurrent_strategy_execution),
            ("テスト27: メモリリーク検出", test_instance.test_memory_leak_detection),
            ("テスト28: 高負荷並行リクエスト", test_instance.test_high_load_concurrent_requests),
            ("テスト29: データベース接続エラー回復", test_instance.test_database_connection_error_recovery),
            ("テスト30: 長時間実行安定性", test_instance.test_long_running_stability),
        ]
        
        self._execute_test_category(test_instance, tests)
    
    def _run_precision_quality_tests(self):
        """精度・品質テスト実行（テスト31-35）"""
        logger.info("\n🎯 カテゴリ6: 精度・品質テスト（統計的有意性・数学的正確性）")
        logger.info("-" * 80)
        
        test_instance = TestPrecisionQuality()
        
        tests = [
            ("テスト31: ML予測統計的有意性", test_instance.test_ml_prediction_statistical_significance),
            ("テスト32: バックテスト再現性", test_instance.test_backtest_reproducibility),
            ("テスト33: TP/SL数学的正確性", test_instance.test_tpsl_mathematical_accuracy),
            ("テスト34: リスク管理境界値", test_instance.test_risk_management_boundary_values),
            ("テスト35: 市場条件別予測精度", test_instance.test_market_condition_prediction_accuracy),
        ]
        
        self._execute_test_category(test_instance, tests)
    
    def _run_performance_tests(self):
        """パフォーマンステスト実行（テスト36-39）"""
        logger.info("\n🚀 カテゴリ7: パフォーマンステスト（大規模処理・最適化）")
        logger.info("-" * 80)
        
        test_instance = TestPerformance()
        
        tests = [
            ("テスト36: 大規模データ処理速度", test_instance.test_large_dataset_processing_speed),
            ("テスト37: 同時接続数上限", test_instance.test_concurrent_connection_limit),
            ("テスト38: CPU/メモリ最適化", test_instance.test_cpu_memory_optimization),
            ("テスト39: レスポンス時間一貫性", test_instance.test_response_time_consistency),
        ]
        
        self._execute_test_category(test_instance, tests)
    
    def _execute_test_category(self, test_instance, tests):
        """テストカテゴリを実行"""
        try:
            test_instance.setup_method()
        except:
            pass
        
        for test_name, test_method in tests:
            self._run_single_test(test_name, test_method, test_instance)
        
        try:
            test_instance.teardown_method()
        except:
            pass
    
    def _run_single_test(self, test_name, test_method, test_instance):
        """単一テストを実行"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            logger.info(f"🔍 実行中: {test_name}")
            
            # テスト実行
            test_method()
            
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
            self.passed_tests += 1
            
            result = {
                "name": test_name,
                "status": "PASSED",
                "execution_time": execution_time,
                "memory_delta": memory_delta,
                "error": None
            }
            
            logger.info(f"✅ 成功: {test_name} ({execution_time:.3f}秒, {memory_delta:+.1f}MB)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory

            self.failed_tests += 1

            result = {
                "name": test_name,
                "status": "FAILED",
                "execution_time": execution_time,
                "memory_delta": memory_delta,
                "error": str(e)
            }

            logger.error(f"❌ 失敗: {test_name} ({execution_time:.3f}秒)")
            logger.error(f"   エラー: {e}")

            # デバッグ用の詳細エラー情報
            if logger.level <= logging.DEBUG:
                logger.debug(f"   スタックトレース:\n{traceback.format_exc()}")

        finally:
            if 'result' in locals():
                self.test_results.append(result)
            else:
                # エラー時のフォールバック
                self.test_results.append({
                    "name": test_name,
                    "status": "ERROR",
                    "execution_time": time.time() - start_time,
                    "memory_delta": 0,
                    "error": "Unknown error"
                })
            
            # システムメトリクス更新
            self._record_system_metrics("update")
            
            # テスト間のクリーンアップ
            try:
                test_instance.teardown_method()
                test_instance.setup_method()
            except:
                pass
    
    def _display_comprehensive_summary(self):
        """包括的サマリーを表示"""
        total_time = self.end_time - self.start_time
        success_rate = (self.passed_tests / self.total_tests) * 100
        
        logger.info("\n" + "=" * 100)
        logger.info("🎯 オートストラテジー包括的拡張テスト結果サマリー")
        logger.info("=" * 100)
        
        # 総合結果
        logger.info(f"📊 総合結果:")
        logger.info(f"   • 総テスト数: {self.total_tests}")
        logger.info(f"   • 成功: {self.passed_tests} ✅")
        logger.info(f"   • 失敗: {self.failed_tests} ❌")
        logger.info(f"   • 成功率: {success_rate:.1f}%")
        logger.info(f"   • 総実行時間: {total_time:.2f}秒 ({total_time/60:.1f}分)")
        
        # カテゴリ別結果
        logger.info(f"\n📋 カテゴリ別結果:")
        
        categories = [
            ("基本包括的テスト", 0, 10),
            ("高度テスト", 10, 15),
            ("統合テスト", 15, 20),
            ("エッジケーステスト", 20, 25),
            ("統合シナリオテスト", 25, 30),
            ("精度・品質テスト", 30, 35),
            ("パフォーマンステスト", 35, 39)
        ]
        
        for category_name, start_idx, end_idx in categories:
            category_results = self.test_results[start_idx:end_idx]
            category_passed = sum(1 for r in category_results if r["status"] == "PASSED")
            category_total = len(category_results)
            category_rate = (category_passed / category_total) * 100 if category_total > 0 else 0
            
            logger.info(f"   • {category_name}: {category_passed}/{category_total} ({category_rate:.1f}%)")
        
        # システムリソース使用量
        logger.info(f"\n💻 システムリソース使用量:")
        memory_delta = self.system_metrics["end_memory"] - self.system_metrics["start_memory"]
        memory_peak_delta = self.system_metrics["peak_memory"] - self.system_metrics["start_memory"]
        
        logger.info(f"   • メモリ: 開始={self.system_metrics['start_memory']:.1f}MB, "
                   f"ピーク={self.system_metrics['peak_memory']:.1f}MB, "
                   f"終了={self.system_metrics['end_memory']:.1f}MB")
        logger.info(f"   • メモリ変化: 最終={memory_delta:+.1f}MB, ピーク増加={memory_peak_delta:+.1f}MB")
        logger.info(f"   • CPU: 開始={self.system_metrics['start_cpu']:.1f}%, "
                   f"ピーク={self.system_metrics['peak_cpu']:.1f}%, "
                   f"終了={self.system_metrics['end_cpu']:.1f}%")
        
        # パフォーマンス統計
        execution_times = [r["execution_time"] for r in self.test_results]
        memory_deltas = [r["memory_delta"] for r in self.test_results]
        
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        
        avg_memory = sum(memory_deltas) / len(memory_deltas)
        max_memory = max(memory_deltas)
        min_memory = min(memory_deltas)
        
        logger.info(f"\n⏱️ パフォーマンス統計:")
        logger.info(f"   • 実行時間: 平均={avg_time:.3f}秒, 最長={max_time:.3f}秒, 最短={min_time:.3f}秒")
        logger.info(f"   • メモリ変化: 平均={avg_memory:+.1f}MB, 最大={max_memory:+.1f}MB, 最小={min_memory:+.1f}MB")
        
        # 失敗したテストの詳細
        if self.failed_tests > 0:
            logger.info(f"\n❌ 失敗したテスト:")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    logger.info(f"   • {result['name']}: {result['error']}")
        
        # 本番運用準備状況評価
        logger.info(f"\n🎯 本番運用準備状況評価:")
        
        if success_rate >= 95:
            readiness_level = "完全準備完了"
            readiness_emoji = "🌟"
            readiness_desc = "全ての機能が完璧に動作し、本番運用に完全対応可能です。"
        elif success_rate >= 90:
            readiness_level = "準備完了"
            readiness_emoji = "✅"
            readiness_desc = "ほぼ全ての機能が正常に動作し、本番運用可能です。"
        elif success_rate >= 80:
            readiness_level = "準備中"
            readiness_emoji = "⚠️"
            readiness_desc = "一部の機能で問題があります。修正後に本番運用を推奨します。"
        elif success_rate >= 70:
            readiness_level = "要改善"
            readiness_emoji = "🔧"
            readiness_desc = "重要な問題があります。本番運用前に修正が必要です。"
        else:
            readiness_level = "未準備"
            readiness_emoji = "🚨"
            readiness_desc = "重大な問題があります。本番運用は推奨されません。"
        
        logger.info(f"   {readiness_emoji} 準備状況: {readiness_level} ({success_rate:.1f}%)")
        logger.info(f"   📝 評価: {readiness_desc}")
        
        # 推奨事項
        logger.info(f"\n💡 推奨事項:")
        if success_rate >= 95:
            logger.info("   • システムは本番運用に完全対応しています")
            logger.info("   • 定期的な監視とメンテナンスを継続してください")
        elif success_rate >= 90:
            logger.info("   • 失敗したテストの原因を調査し、必要に応じて修正してください")
            logger.info("   • 本番環境での段階的展開を推奨します")
        else:
            logger.info("   • 失敗したテストを優先的に修正してください")
            logger.info("   • 修正後に再度包括的テストを実行してください")
            logger.info("   • 本番運用前に追加の検証を実施してください")
        
        logger.info("=" * 100)


def main():
    """メイン実行関数"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # テストスイート実行
    test_suite = ComprehensiveExtendedTestSuite()
    success = test_suite.run_all_tests()
    
    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
