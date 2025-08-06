"""
オートストラテジー包括的テストスイート実行器

全20個のオートストラテジーテストを統合実行し、
MLとオートストラテジーの完全連携と計算精度を検証します。
"""

import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

import logging
import time
from datetime import datetime
import traceback

# テストモジュールをインポート
from tests.auto_strategy.test_auto_strategy_comprehensive import TestAutoStrategyComprehensive
from tests.auto_strategy.test_auto_strategy_advanced import TestAutoStrategyAdvanced
from tests.auto_strategy.test_auto_strategy_integration import TestAutoStrategyIntegration

logger = logging.getLogger(__name__)


class AutoStrategyTestSuite:
    """オートストラテジー包括的テストスイート"""
    
    def __init__(self):
        """初期化"""
        self.start_time = None
        self.end_time = None
        self.total_tests = 20
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def run_all_tests(self):
        """全てのオートストラテジーテストを実行"""
        logger.info("🚀 オートストラテジー包括的テストスイート開始")
        logger.info("=" * 80)
        logger.info("MLとオートストラテジーの完全連携・計算精度検証")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # テストカテゴリ別実行
        self._run_comprehensive_tests()
        self._run_advanced_tests()
        self._run_integration_tests()
        
        self.end_time = time.time()
        self._display_final_summary()
        
        return self.passed_tests == self.total_tests
    
    def _run_comprehensive_tests(self):
        """包括的テスト実行（テスト1-10）"""
        logger.info("\n📋 カテゴリ1: 包括的テスト（基本機能・計算精度）")
        logger.info("-" * 60)
        
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
        logger.info("-" * 60)
        
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
        logger.info("-" * 60)
        
        test_instance = TestAutoStrategyIntegration()
        
        tests = [
            ("テスト16: エンドツーエンド戦略生成", test_instance.test_end_to_end_strategy_generation),
            ("テスト17: ML-オートストラテジー完全パイプライン", test_instance.test_ml_auto_strategy_full_pipeline),
            ("テスト18: API統合シミュレーション", test_instance.test_api_integration_simulation),
            ("テスト19: データフロー一貫性", test_instance.test_data_flow_consistency),
            ("テスト20: 設定検証", test_instance.test_configuration_validation),
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
        
        try:
            logger.info(f"🔍 実行中: {test_name}")
            
            # テスト実行
            test_method()
            
            execution_time = time.time() - start_time
            self.passed_tests += 1
            
            result = {
                "name": test_name,
                "status": "PASSED",
                "execution_time": execution_time,
                "error": None
            }
            
            logger.info(f"✅ 成功: {test_name} ({execution_time:.3f}秒)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.failed_tests += 1

            result = {
                "name": test_name,
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }

            logger.error(f"❌ 失敗: {test_name} ({execution_time:.3f}秒)")
            logger.error(f"   エラー: {e}")

            # デバッグ用の詳細エラー情報
            if logger.level <= logging.DEBUG:
                logger.debug(f"   スタックトレース:\n{traceback.format_exc()}")

        finally:
            # resultが定義されていない場合のフォールバック
            if 'result' not in locals():
                result = {
                    "name": test_name,
                    "status": "ERROR",
                    "execution_time": time.time() - start_time,
                    "error": "テスト実行中に予期しないエラーが発生しました"
                }
            self.test_results.append(result)
            
            # テスト間のクリーンアップ
            try:
                test_instance.teardown_method()
                test_instance.setup_method()
            except:
                pass
    
    def _display_final_summary(self):
        """最終サマリーを表示"""
        total_time = self.end_time - self.start_time
        success_rate = (self.passed_tests / self.total_tests) * 100
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 オートストラテジー包括的テスト結果サマリー")
        logger.info("=" * 80)
        
        logger.info(f"📊 総合結果:")
        logger.info(f"   • 総テスト数: {self.total_tests}")
        logger.info(f"   • 成功: {self.passed_tests} ✅")
        logger.info(f"   • 失敗: {self.failed_tests} ❌")
        logger.info(f"   • 成功率: {success_rate:.1f}%")
        logger.info(f"   • 総実行時間: {total_time:.2f}秒")
        
        # カテゴリ別結果
        logger.info(f"\n📋 カテゴリ別結果:")
        
        categories = [
            ("包括的テスト", 0, 10),
            ("高度テスト", 10, 15),
            ("統合テスト", 15, 20)
        ]
        
        for category_name, start_idx, end_idx in categories:
            category_results = self.test_results[start_idx:end_idx]
            category_passed = sum(1 for r in category_results if r["status"] == "PASSED")
            category_total = len(category_results)
            category_rate = (category_passed / category_total) * 100 if category_total > 0 else 0
            
            logger.info(f"   • {category_name}: {category_passed}/{category_total} ({category_rate:.1f}%)")
        
        # 失敗したテストの詳細
        if self.failed_tests > 0:
            logger.info(f"\n❌ 失敗したテスト:")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    logger.info(f"   • {result['name']}: {result['error']}")
        
        # パフォーマンス統計
        execution_times = [r["execution_time"] for r in self.test_results]
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        
        logger.info(f"\n⏱️ パフォーマンス統計:")
        logger.info(f"   • 平均実行時間: {avg_time:.3f}秒")
        logger.info(f"   • 最長実行時間: {max_time:.3f}秒")
        logger.info(f"   • 最短実行時間: {min_time:.3f}秒")
        
        # 最終評価
        logger.info(f"\n🎯 最終評価:")
        if success_rate >= 90:
            logger.info("🌟 優秀: MLとオートストラテジーの連携が完璧に動作しています！")
        elif success_rate >= 80:
            logger.info("✅ 良好: MLとオートストラテジーの連携が良好に動作しています。")
        elif success_rate >= 70:
            logger.info("⚠️ 注意: 一部の機能で問題があります。修正を推奨します。")
        else:
            logger.info("🚨 警告: 重大な問題があります。緊急修正が必要です。")
        
        logger.info("=" * 80)
    
    def run_specific_category(self, category: str):
        """特定のカテゴリのみを実行"""
        logger.info(f"🎯 特定カテゴリ実行: {category}")
        
        self.start_time = time.time()
        
        if category == "comprehensive":
            self._run_comprehensive_tests()
        elif category == "advanced":
            self._run_advanced_tests()
        elif category == "integration":
            self._run_integration_tests()
        else:
            logger.error(f"未知のカテゴリ: {category}")
            logger.info("利用可能なカテゴリ: comprehensive, advanced, integration")
            return False
        
        self.end_time = time.time()
        self._display_final_summary()
        
        return self.passed_tests > 0


def main():
    """メイン実行関数"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # テストスイート実行
    test_suite = AutoStrategyTestSuite()
    
    # コマンドライン引数の確認
    if len(sys.argv) > 1:
        category = sys.argv[1]
        success = test_suite.run_specific_category(category)
    else:
        success = test_suite.run_all_tests()
    
    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
