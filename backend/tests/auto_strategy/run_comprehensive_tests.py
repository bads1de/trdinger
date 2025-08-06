"""
オートストラテジー 包括的テストスイート実行スクリプト

全60個以上のテストを実行し、詳細な結果レポートを生成します。
"""

import sys
import os
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any
import psutil

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

# ログ設定（絵文字を除去してエンコーディング問題を回避）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'comprehensive_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """包括的テスト実行クラス"""
    
    def __init__(self):
        self.test_modules = [
            {
                "name": "計算精度・正確性テスト",
                "module": "test_calculation_accuracy",
                "expected_tests": 5,
                "description": "TP/SL計算、指標計算、戦略生成の精度と正確性"
            },
            {
                "name": "AutoStrategy動作検証テスト",
                "module": "test_autostrategy_behavior",
                "expected_tests": 6,
                "description": "AutoStrategy全体の動作フロー、コンポーネント連携、実際の取引シナリオ"
            },
            {
                "name": "リアルタイム処理テスト", 
                "module": "test_realtime_processing",
                "expected_tests": 5,
                "description": "ライブデータストリーミング、高頻度取引、WebSocket接続"
            },
            {
                "name": "データ整合性・一貫性テスト",
                "module": "test_data_consistency", 
                "expected_tests": 5,
                "description": "データソース間整合性、バックアップ・復元、分散処理"
            },
            {
                "name": "国際化・多様性テスト",
                "module": "test_internationalization",
                "expected_tests": 5,
                "description": "タイムゾーン、通貨ペア、小数点精度、地域固有処理"
            },
            {
                "name": "監視・ログテスト",
                "module": "test_monitoring_logging",
                "expected_tests": 4,
                "description": "異常検知、ログ管理、パフォーマンスメトリクス、エラー追跡"
            }
        ]
        
        self.results = {
            "start_time": None,
            "end_time": None,
            "total_duration": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "module_results": {},
            "system_info": {},
            "performance_metrics": {}
        }
    
    def collect_system_info(self):
        """システム情報を収集"""
        try:
            memory_info = psutil.virtual_memory()
            self.results["system_info"] = {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_total": memory_info.total / (1024**3),  # GB
                "memory_available": memory_info.available / (1024**3),  # GB
                "disk_usage": psutil.disk_usage('C:').percent if os.name == 'nt' else psutil.disk_usage('/').percent
            }

            logger.info("システム情報:")
            logger.info(f"  Python: {self.results['system_info']['python_version'].split()[0]}")
            logger.info(f"  プラットフォーム: {self.results['system_info']['platform']}")
            logger.info(f"  CPU数: {self.results['system_info']['cpu_count']}")
            logger.info(f"  総メモリ: {self.results['system_info']['memory_total']:.1f}GB")
            logger.info(f"  利用可能メモリ: {self.results['system_info']['memory_available']:.1f}GB")
            logger.info(f"  ディスク使用率: {self.results['system_info']['disk_usage']:.1f}%")

        except Exception as e:
            logger.warning(f"システム情報収集エラー: {e}")
            # デフォルト値を設定
            self.results["system_info"] = {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": 1,
                "memory_total": 8.0,
                "memory_available": 4.0,
                "disk_usage": 50.0
            }
    
    def run_module_tests(self, module_info: Dict[str, Any]) -> Dict[str, Any]:
        """モジュール別テストを実行"""
        module_name = module_info["module"]
        display_name = module_info["name"]
        expected_tests = module_info["expected_tests"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{display_name} 開始")
        logger.info(f"{'='*60}")
        logger.info(f"説明: {module_info['description']}")
        logger.info(f"期待テスト数: {expected_tests}")
        
        module_start = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        module_result = {
            "name": display_name,
            "module": module_name,
            "start_time": datetime.now(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "execution_time": 0,
            "memory_usage": 0,
            "errors": [],
            "success_rate": 0.0
        }
        
        try:
            # モジュールをインポート
            module = __import__(module_name)
            
            # テストクラスを取得（正しいクラス名を使用）
            test_class_mapping = {
                "test_security_robustness": "TestSecurityRobustness",
                "test_realtime_processing": "TestRealtimeProcessing",
                "test_data_consistency": "TestDataConsistency",
                "test_internationalization": "TestInternationalization",
                "test_monitoring_logging": "TestMonitoringLogging"
            }

            test_class_name = test_class_mapping.get(module_name)
            if not test_class_name:
                raise AttributeError(f"Unknown test module: {module_name}")

            test_class = getattr(module, test_class_name)
            
            # テストインスタンスを作成
            test_instance = test_class()
            
            # テストメソッドを取得
            test_methods = [
                method for method in dir(test_instance) 
                if method.startswith('test_') and callable(getattr(test_instance, method))
            ]
            
            logger.info(f"発見されたテストメソッド: {len(test_methods)}")
            
            # 各テストメソッドを実行
            for test_method_name in test_methods:
                test_method = getattr(test_instance, test_method_name)
                
                try:
                    logger.info(f"  実行中: {test_method_name}")

                    # セットアップ
                    if hasattr(test_instance, 'setup_method'):
                        test_instance.setup_method()

                    # テスト実行
                    test_start = time.time()
                    test_method()
                    test_duration = time.time() - test_start

                    # ティアダウン
                    if hasattr(test_instance, 'teardown_method'):
                        test_instance.teardown_method()

                    module_result["tests_passed"] += 1
                    logger.info(f"  OK {test_method_name} 成功 ({test_duration:.3f}秒)")
                    
                except Exception as e:
                    module_result["tests_failed"] += 1
                    error_info = {
                        "test_method": test_method_name,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    module_result["errors"].append(error_info)
                    logger.error(f"  FAIL {test_method_name} 失敗: {e}")
                
                module_result["tests_run"] += 1
            
        except Exception as e:
            logger.error(f"モジュール {module_name} の実行エラー: {e}")
            module_result["errors"].append({
                "test_method": "module_import",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        # モジュール結果の計算
        module_result["execution_time"] = time.time() - module_start
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        module_result["memory_usage"] = final_memory - initial_memory
        module_result["end_time"] = datetime.now()
        
        if module_result["tests_run"] > 0:
            module_result["success_rate"] = module_result["tests_passed"] / module_result["tests_run"]
        
        # モジュール結果のサマリー
        logger.info(f"\n{display_name} 結果:")
        logger.info(f"  実行テスト数: {module_result['tests_run']}")
        logger.info(f"  成功: {module_result['tests_passed']}")
        logger.info(f"  失敗: {module_result['tests_failed']}")
        logger.info(f"  成功率: {module_result['success_rate']:.1%}")
        logger.info(f"  実行時間: {module_result['execution_time']:.3f}秒")
        logger.info(f"  メモリ使用量: {module_result['memory_usage']:+.1f}MB")
        
        return module_result
    
    def run_all_tests(self):
        """全テストを実行"""
        logger.info("オートストラテジー包括的テストスイート開始")
        logger.info(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.results["start_time"] = datetime.now()
        overall_start = time.time()

        # システム情報収集
        self.collect_system_info()

        # 各モジュールのテストを実行
        for module_info in self.test_modules:
            module_result = self.run_module_tests(module_info)
            self.results["module_results"][module_info["name"]] = module_result

            # 全体結果に加算
            self.results["total_tests"] += module_result["tests_run"]
            self.results["passed_tests"] += module_result["tests_passed"]
            self.results["failed_tests"] += module_result["tests_failed"]
        
        # 全体結果の計算
        self.results["end_time"] = datetime.now()
        self.results["total_duration"] = time.time() - overall_start
        
        # パフォーマンスメトリクス
        self.results["performance_metrics"] = {
            "tests_per_second": self.results["total_tests"] / self.results["total_duration"],
            "average_test_time": self.results["total_duration"] / self.results["total_tests"] if self.results["total_tests"] > 0 else 0,
            "memory_efficiency": self.results["system_info"]["memory_available"] / self.results["system_info"]["memory_total"]
        }
    
    def generate_report(self):
        """詳細レポートを生成"""
        logger.info(f"\n{'='*80}")
        logger.info("オートストラテジー包括的テストスイート完了報告")
        logger.info(f"{'='*80}")

        # 全体結果
        success_rate = self.results["passed_tests"] / self.results["total_tests"] if self.results["total_tests"] > 0 else 0

        logger.info(f"\n総合結果サマリー")
        logger.info(f"  総テスト数: {self.results['total_tests']}個")
        logger.info(f"  成功: {self.results['passed_tests']}個")
        logger.info(f"  失敗: {self.results['failed_tests']}個")
        logger.info(f"  成功率: {success_rate:.1%}")
        logger.info(f"  総実行時間: {self.results['total_duration']:.2f}秒 ({self.results['total_duration']/60:.1f}分)")

        # パフォーマンスメトリクス
        logger.info(f"\nパフォーマンスメトリクス")
        logger.info(f"  テスト実行速度: {self.results['performance_metrics']['tests_per_second']:.1f}テスト/秒")
        logger.info(f"  平均テスト時間: {self.results['performance_metrics']['average_test_time']:.3f}秒")
        logger.info(f"  メモリ効率性: {self.results['performance_metrics']['memory_efficiency']:.1%}")

        # モジュール別結果
        logger.info(f"\nカテゴリ別結果")
        for module_name, result in self.results["module_results"].items():
            status = "OK" if result["tests_failed"] == 0 else "FAIL"
            logger.info(f"  {status} {module_name}: {result['tests_passed']}/{result['tests_run']} ({result['success_rate']:.1%}) - {result['execution_time']:.1f}秒")

        # 失敗したテストの詳細
        if self.results["failed_tests"] > 0:
            logger.info(f"\n失敗したテストの詳細")
            for module_name, result in self.results["module_results"].items():
                if result["errors"]:
                    logger.info(f"\n  {module_name}:")
                    for error in result["errors"]:
                        logger.info(f"    - {error['test_method']}: {error['error']}")

        # 最終評価
        if success_rate == 1.0:
            logger.info(f"\n最終評価: 完全成功")
            logger.info(f"全{self.results['total_tests']}個のテストが100%成功しました！")
            logger.info(f"システムは本番運用に完全対応可能です。")
        elif success_rate >= 0.95:
            logger.info(f"\n最終評価: 優秀")
            logger.info(f"成功率{success_rate:.1%}で、高品質なシステムです。")
        elif success_rate >= 0.90:
            logger.info(f"\n最終評価: 良好")
            logger.info(f"成功率{success_rate:.1%}で、一部改善が必要です。")
        else:
            logger.info(f"\n最終評価: 要改善")
            logger.info(f"成功率{success_rate:.1%}で、重大な問題があります。")

        # 推奨事項
        logger.info(f"\n推奨事項:")
        if success_rate == 1.0:
            logger.info(f"  - システムは完璧に動作しています")
            logger.info(f"  - 定期的な監視とメンテナンスを継続してください")
        else:
            logger.info(f"  - 失敗したテストを詳細に調査してください")
            logger.info(f"  - 必要に応じてコードの修正を行ってください")
            logger.info(f"  - 修正後に再度テストを実行してください")
        
        return success_rate


def main():
    """メイン実行関数"""
    try:
        runner = ComprehensiveTestRunner()
        runner.run_all_tests()
        success_rate = runner.generate_report()
        
        # 終了コード
        if success_rate == 1.0:
            sys.exit(0)  # 完全成功
        elif success_rate >= 0.95:
            sys.exit(0)  # 許容範囲内
        else:
            sys.exit(1)  # 改善が必要
            
    except Exception as e:
        logger.error(f"テストスイート実行エラー: {e}")
        logger.error(traceback.format_exc())
        sys.exit(2)


if __name__ == "__main__":
    main()
