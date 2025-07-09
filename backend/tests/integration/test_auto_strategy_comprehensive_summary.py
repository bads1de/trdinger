"""
オートストラテジー包括的テスト実行サマリー

全てのコンポーネントテストを実行し、結果をまとめて報告します。
"""

import pytest
import subprocess
import sys
import time
from typing import Dict, List, Tuple


class AutoStrategyTestSummary:
    """オートストラテジーテストサマリークラス"""

    def __init__(self):
        self.test_files = [
            "tests/integration/test_technical_indicators_comprehensive.py",
            "tests/integration/test_tpsl_functionality_comprehensive.py", 
            "tests/integration/test_position_sizing_comprehensive.py",
            "tests/integration/test_auto_strategy_integration_comprehensive.py",
        ]
        self.results = {}

    def run_all_tests(self) -> Dict[str, Dict]:
        """全てのテストを実行"""
        print("=" * 80)
        print("オートストラテジー包括的テスト実行開始")
        print("=" * 80)
        
        overall_start_time = time.time()
        
        for test_file in self.test_files:
            print(f"\n🧪 実行中: {test_file}")
            print("-" * 60)
            
            start_time = time.time()
            result = self._run_single_test(test_file)
            end_time = time.time()
            
            result['execution_time'] = end_time - start_time
            self.results[test_file] = result
            
            # 結果の即座表示
            self._print_test_result(test_file, result)
        
        overall_end_time = time.time()
        total_time = overall_end_time - overall_start_time
        
        # 最終サマリー表示
        self._print_final_summary(total_time)
        
        return self.results

    def _run_single_test(self, test_file: str) -> Dict:
        """単一テストファイルを実行"""
        try:
            # pytestを実行
            cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=".",
                timeout=300  # 5分のタイムアウト
            )
            
            # 結果の解析
            parsed_result = self._parse_pytest_output(result)

            # デバッグ用出力
            if parsed_result['total'] == 0 and 'passed' in result.stdout:
                print(f"DEBUG: 解析結果 for {test_file}: {parsed_result}")
                lines = result.stdout.split('\n')
                for line in lines:
                    if '=====' in line and 'passed' in line:
                        print(f"DEBUG: 見つかった行: '{line}'")
                        clean_line = line.replace('=', '').strip()
                        print(f"DEBUG: クリーン後: '{clean_line}'")
                        parts = clean_line.split()
                        print(f"DEBUG: 分割後: {parts}")
                        break

            return parsed_result
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'passed': 0,
                'failed': 0,
                'total': 0,
                'error_message': 'テスト実行がタイムアウトしました',
                'stdout': '',
                'stderr': ''
            }
        except Exception as e:
            return {
                'status': 'error',
                'passed': 0,
                'failed': 0,
                'total': 0,
                'error_message': str(e),
                'stdout': '',
                'stderr': ''
            }

    def _parse_pytest_output(self, result: subprocess.CompletedProcess) -> Dict:
        """pytest出力を解析"""
        stdout = result.stdout
        stderr = result.stderr
        
        # 基本情報
        test_result = {
            'stdout': stdout,
            'stderr': stderr,
            'return_code': result.returncode,
            'passed': 0,
            'failed': 0,
            'total': 0,
            'status': 'unknown'
        }
        
        # 成功/失敗の解析
        import re

        # ANSIエスケープシーケンスを除去
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_stdout = ansi_escape.sub('', stdout)

        lines = clean_stdout.split('\n')
        for line in lines:
            # 最終結果行を探す（例: "====== 7 passed in 0.68s ======"）
            if '=====' in line and ('passed' in line or 'failed' in line):
                # "=====" を除去してから解析
                clean_line = line.replace('=', '').strip()
                parts = clean_line.split()
                try:
                    # "X passed" または "X failed, Y passed" の形式を解析
                    if 'failed' in clean_line and 'passed' in clean_line:
                        # "X failed, Y passed in Z.ZZs" 形式
                        for i, part in enumerate(parts):
                            if part == 'failed' and i > 0:
                                test_result['failed'] = int(parts[i-1])
                            elif part == 'passed' and i > 0:
                                test_result['passed'] = int(parts[i-1])
                    elif 'passed' in clean_line:
                        # "X passed in Z.ZZs" 形式
                        for i, part in enumerate(parts):
                            if part == 'passed' and i > 0:
                                test_result['passed'] = int(parts[i-1])
                                break
                    elif 'failed' in clean_line:
                        # "X failed in Z.ZZs" 形式
                        for i, part in enumerate(parts):
                            if part == 'failed' and i > 0:
                                test_result['failed'] = int(parts[i-1])
                                break
                except (ValueError, IndexError):
                    continue
                break
        
        test_result['total'] = test_result['passed'] + test_result['failed']
        
        # ステータス決定
        if result.returncode == 0:
            test_result['status'] = 'success'
        elif test_result['failed'] > 0:
            test_result['status'] = 'failed'
        else:
            test_result['status'] = 'error'
            test_result['error_message'] = stderr or "不明なエラー"
        
        return test_result

    def _print_test_result(self, test_file: str, result: Dict):
        """テスト結果を表示"""
        status = result['status']
        passed = result['passed']
        failed = result['failed']
        total = result['total']
        exec_time = result.get('execution_time', 0)
        
        # ファイル名を短縮
        short_name = test_file.split('/')[-1].replace('test_', '').replace('_comprehensive.py', '')
        
        if status == 'success':
            print(f"✅ {short_name}: {passed}/{total} 成功 ({exec_time:.2f}秒)")
        elif status == 'failed':
            print(f"❌ {short_name}: {passed}/{total} 成功, {failed} 失敗 ({exec_time:.2f}秒)")
        elif status == 'timeout':
            print(f"⏰ {short_name}: タイムアウト ({exec_time:.2f}秒)")
        else:
            print(f"💥 {short_name}: エラー - {result.get('error_message', '不明')} ({exec_time:.2f}秒)")

    def _print_final_summary(self, total_time: float):
        """最終サマリーを表示"""
        print("\n" + "=" * 80)
        print("📊 オートストラテジー包括的テスト結果サマリー")
        print("=" * 80)
        
        total_passed = 0
        total_failed = 0
        total_tests = 0
        success_count = 0
        
        for test_file, result in self.results.items():
            short_name = test_file.split('/')[-1].replace('test_', '').replace('_comprehensive.py', '')
            status = result['status']
            passed = result['passed']
            failed = result['failed']
            exec_time = result.get('execution_time', 0)
            
            total_passed += passed
            total_failed += failed
            total_tests += passed + failed
            
            if status == 'success':
                success_count += 1
                status_icon = "✅"
            elif status == 'failed':
                status_icon = "❌"
            elif status == 'timeout':
                status_icon = "⏰"
            else:
                status_icon = "💥"
            
            print(f"{status_icon} {short_name:<30} {passed:>3}/{passed+failed:<3} ({exec_time:>5.2f}秒)")
        
        print("-" * 80)
        print(f"📈 総合結果:")
        print(f"   テストファイル: {len(self.test_files)}個")
        print(f"   成功ファイル: {success_count}個")
        print(f"   総テスト数: {total_tests}個")
        print(f"   成功テスト: {total_passed}個")
        print(f"   失敗テスト: {total_failed}個")
        print(f"   成功率: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%")
        print(f"   総実行時間: {total_time:.2f}秒")
        
        # 全体評価
        if success_count == len(self.test_files) and total_failed == 0:
            print(f"\n🎉 全てのテストが成功しました！")
            print(f"   オートストラテジーシステムは正常に動作しています。")
        elif total_failed == 0:
            print(f"\n✅ 全てのテストが成功しました！")
            print(f"   一部のファイルでエラーがありましたが、実行されたテストは全て成功です。")
        else:
            print(f"\n⚠️  一部のテストが失敗しました。")
            print(f"   失敗したテストを確認し、修正が必要です。")
        
        print("=" * 80)

    def generate_detailed_report(self) -> str:
        """詳細レポートを生成"""
        report = []
        report.append("# オートストラテジー包括的テスト詳細レポート\n")
        
        for test_file, result in self.results.items():
            short_name = test_file.split('/')[-1]
            report.append(f"## {short_name}\n")
            report.append(f"- ステータス: {result['status']}")
            report.append(f"- 成功: {result['passed']}")
            report.append(f"- 失敗: {result['failed']}")
            report.append(f"- 実行時間: {result.get('execution_time', 0):.2f}秒")
            
            if result['status'] != 'success':
                report.append(f"- エラー詳細:")
                if result.get('error_message'):
                    report.append(f"  {result['error_message']}")
                if result.get('stderr'):
                    report.append(f"  標準エラー: {result['stderr'][:200]}...")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """メイン実行関数"""
    summary = AutoStrategyTestSummary()
    results = summary.run_all_tests()
    
    # 詳細レポートをファイルに保存
    try:
        report = summary.generate_detailed_report()
        with open("auto_strategy_test_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📄 詳細レポートを auto_strategy_test_report.md に保存しました。")
    except Exception as e:
        print(f"\n⚠️  レポート保存エラー: {e}")
    
    # 終了コード決定
    total_failed = sum(r['failed'] for r in results.values())
    error_count = sum(1 for r in results.values() if r['status'] in ['error', 'timeout'])
    
    if total_failed > 0 or error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
