"""
オートストラテジー包括的テスト実行スクリプト

作成された12個の包括的テストファイルを順次実行し、
結果をレポートとして出力します。
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'comprehensive_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """包括的テスト実行クラス"""
    
    def __init__(self):
        self.test_files = [
            "test_auto_strategy_service_comprehensive.py",
            "test_ml_orchestrator_comprehensive.py", 
            "test_smart_condition_generator_comprehensive.py",
            "test_tpsl_auto_decision_comprehensive.py",
            "test_experiment_manager_comprehensive.py",
            "test_backtest_integration_comprehensive.py",
            "test_persistence_service_comprehensive.py",
            "test_api_endpoints_comprehensive.py",
            "test_error_handling_edge_cases_comprehensive.py",
            "test_performance_data_validation_comprehensive.py",
            "test_frontend_integration_comprehensive.py"
        ]
        
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def run_single_test(self, test_file: str) -> dict:
        """単一テストファイルを実行"""
        logger.info(f"🧪 テスト実行開始: {test_file}")
        
        start_time = time.time()
        
        try:
            # pytestコマンドを実行
            cmd = [
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                "--disable-warnings",
                "--maxfail=5"  # 5個のテストが失敗したら停止
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5分でタイムアウト
                cwd=Path(__file__).parent
            )
            
            execution_time = time.time() - start_time
            
            # 結果解析
            output_lines = result.stdout.split('\n')
            error_lines = result.stderr.split('\n')
            
            # テスト結果の統計を抽出
            stats = self._parse_test_stats(output_lines)
            
            test_result = {
                "file": test_file,
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stats": stats,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "summary": self._generate_test_summary(stats, execution_time)
            }
            
            if test_result["success"]:
                logger.info(f"✅ テスト成功: {test_file} ({execution_time:.2f}秒)")
                logger.info(f"   統計: {stats}")
            else:
                logger.error(f"❌ テスト失敗: {test_file} ({execution_time:.2f}秒)")
                logger.error(f"   エラー: {result.stderr[:200]}...")
                
            return test_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"⏰ テストタイムアウト: {test_file} ({execution_time:.2f}秒)")
            
            return {
                "file": test_file,
                "success": False,
                "execution_time": execution_time,
                "return_code": -1,
                "stats": {"timeout": True},
                "stdout": "",
                "stderr": "Test timed out after 300 seconds",
                "summary": f"タイムアウト ({execution_time:.2f}秒)"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"💥 テスト実行エラー: {test_file} - {e}")
            
            return {
                "file": test_file,
                "success": False,
                "execution_time": execution_time,
                "return_code": -2,
                "stats": {"error": str(e)},
                "stdout": "",
                "stderr": str(e),
                "summary": f"実行エラー: {e}"
            }
    
    def _parse_test_stats(self, output_lines: list) -> dict:
        """テスト統計を解析"""
        stats = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "warnings": 0,
            "total": 0
        }
        
        for line in output_lines:
            line = line.strip()
            
            # pytest の結果行を解析
            if " passed" in line and " failed" in line:
                # 例: "5 passed, 2 failed, 1 skipped in 10.5s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        stats["passed"] = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        stats["failed"] = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        stats["skipped"] = int(parts[i-1])
                    elif part == "error" and i > 0:
                        stats["errors"] = int(parts[i-1])
            elif " passed in " in line:
                # 例: "10 passed in 5.2s"
                parts = line.split()
                if len(parts) > 0 and parts[0].isdigit():
                    stats["passed"] = int(parts[0])
            elif "FAILED" in line:
                stats["failed"] += 1
            elif "PASSED" in line:
                stats["passed"] += 1
            elif "SKIPPED" in line:
                stats["skipped"] += 1
        
        stats["total"] = stats["passed"] + stats["failed"] + stats["skipped"] + stats["errors"]
        return stats
    
    def _generate_test_summary(self, stats: dict, execution_time: float) -> str:
        """テスト結果サマリーを生成"""
        if "timeout" in stats:
            return "タイムアウト"
        elif "error" in stats:
            return f"エラー: {stats['error']}"
        
        total = stats["total"]
        passed = stats["passed"]
        failed = stats["failed"]
        skipped = stats["skipped"]
        
        if total == 0:
            return "テストが見つかりませんでした"
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        return f"{passed}/{total} 成功 ({success_rate:.1f}%), {failed} 失敗, {skipped} スキップ, {execution_time:.2f}秒"
    
    def run_all_tests(self):
        """すべてのテストを実行"""
        logger.info("🚀 オートストラテジー包括的テスト開始")
        logger.info(f"📋 実行予定テストファイル数: {len(self.test_files)}")
        
        self.start_time = time.time()
        
        for test_file in self.test_files:
            if not os.path.exists(test_file):
                logger.warning(f"⚠️  テストファイルが見つかりません: {test_file}")
                self.results[test_file] = {
                    "file": test_file,
                    "success": False,
                    "execution_time": 0,
                    "return_code": -3,
                    "stats": {"file_not_found": True},
                    "stdout": "",
                    "stderr": "Test file not found",
                    "summary": "ファイルが見つかりません"
                }
                continue
            
            result = self.run_single_test(test_file)
            self.results[test_file] = result
            
            # 短い休憩（システムリソースの回復）
            time.sleep(1)
        
        self.end_time = time.time()
        
        # 最終レポート生成
        self._generate_final_report()
    
    def _generate_final_report(self):
        """最終レポートを生成"""
        total_time = self.end_time - self.start_time
        
        logger.info("=" * 80)
        logger.info("📊 オートストラテジー包括的テスト結果レポート")
        logger.info("=" * 80)
        
        # 全体統計
        total_tests = len(self.test_files)
        successful_tests = sum(1 for r in self.results.values() if r["success"])
        failed_tests = total_tests - successful_tests
        
        logger.info(f"🎯 全体結果:")
        logger.info(f"   総テストファイル数: {total_tests}")
        logger.info(f"   成功: {successful_tests}")
        logger.info(f"   失敗: {failed_tests}")
        logger.info(f"   成功率: {(successful_tests/total_tests)*100:.1f}%")
        logger.info(f"   総実行時間: {total_time:.2f}秒")
        
        # 個別結果
        logger.info(f"\n📋 個別テスト結果:")
        for test_file, result in self.results.items():
            status = "✅" if result["success"] else "❌"
            logger.info(f"   {status} {test_file}: {result['summary']}")
        
        # 詳細統計
        total_passed = sum(r["stats"].get("passed", 0) for r in self.results.values())
        total_failed = sum(r["stats"].get("failed", 0) for r in self.results.values())
        total_skipped = sum(r["stats"].get("skipped", 0) for r in self.results.values())
        total_individual_tests = total_passed + total_failed + total_skipped
        
        if total_individual_tests > 0:
            logger.info(f"\n🔍 詳細統計:")
            logger.info(f"   総個別テスト数: {total_individual_tests}")
            logger.info(f"   成功: {total_passed}")
            logger.info(f"   失敗: {total_failed}")
            logger.info(f"   スキップ: {total_skipped}")
            logger.info(f"   個別テスト成功率: {(total_passed/total_individual_tests)*100:.1f}%")
        
        # パフォーマンス分析
        execution_times = [r["execution_time"] for r in self.results.values() if r["execution_time"] > 0]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            logger.info(f"\n⏱️  パフォーマンス分析:")
            logger.info(f"   平均実行時間: {avg_time:.2f}秒")
            logger.info(f"   最大実行時間: {max_time:.2f}秒")
            logger.info(f"   最小実行時間: {min_time:.2f}秒")
        
        # 失敗したテストの詳細
        failed_results = [r for r in self.results.values() if not r["success"]]
        if failed_results:
            logger.info(f"\n❌ 失敗したテストの詳細:")
            for result in failed_results:
                logger.info(f"   📁 {result['file']}:")
                logger.info(f"      理由: {result['summary']}")
                if result["stderr"]:
                    logger.info(f"      エラー: {result['stderr'][:200]}...")
        
        # 推奨事項
        logger.info(f"\n💡 推奨事項:")
        if failed_tests > 0:
            logger.info(f"   • {failed_tests}個のテストファイルが失敗しました。個別に確認してください。")
        if total_failed > 0:
            logger.info(f"   • {total_failed}個の個別テストが失敗しました。詳細を確認してください。")
        if total_skipped > 0:
            logger.info(f"   • {total_skipped}個のテストがスキップされました。依存関係を確認してください。")
        
        if successful_tests == total_tests and total_failed == 0:
            logger.info("   🎉 すべてのテストが成功しました！オートストラテジー機能は正常に動作しています。")
        
        logger.info("=" * 80)


def main():
    """メイン実行関数"""
    try:
        # 現在のディレクトリを確認
        current_dir = Path(__file__).parent
        logger.info(f"📂 テスト実行ディレクトリ: {current_dir}")
        
        # テストランナーを作成して実行
        runner = ComprehensiveTestRunner()
        runner.run_all_tests()
        
        # 結果に基づいて終了コードを設定
        failed_count = sum(1 for r in runner.results.values() if not r["success"])
        if failed_count > 0:
            logger.warning(f"⚠️  {failed_count}個のテストファイルが失敗しました")
            sys.exit(1)
        else:
            logger.info("🎉 すべてのテストが成功しました！")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("⏹️  ユーザーによってテスト実行が中断されました")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 テスト実行中に予期しないエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
