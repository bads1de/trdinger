"""
MLトレーニング系の包括的テストスイート

計算正確性、前処理正確性、特徴量計算、データ変換、ラベル生成の
すべてのテストを統合実行し、MLシステム全体の信頼性を検証します。
"""

import logging
import sys
import os
import time
from typing import Dict, List, Tuple, Any
import traceback

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 各テストモジュールをインポート
from tests.calculations.test_ml_calculations import run_all_calculation_tests
from tests.preprocessing.test_preprocessing_accuracy import run_all_preprocessing_tests
from tests.feature_engineering.test_feature_calculations import run_all_feature_calculation_tests
from tests.data_transformations.test_data_transformations import run_all_data_transformation_tests
from tests.label_generation.test_label_generation import run_all_label_generation_tests
from tests.enhanced.test_error_handling import run_all_error_handling_tests
from tests.enhanced.test_performance import run_all_performance_tests

logger = logging.getLogger(__name__)


class MLAccuracyTestSuite:
    """MLトレーニング系の包括的テストスイート"""

    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = None
        self.end_time = None

    def run_test_module(self, test_name: str, test_function) -> bool:
        """個別テストモジュールを実行"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 {test_name} を開始")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            success = test_function()
            execution_time = time.time() - start_time
            
            if success:
                logger.info(f"✅ {test_name} 成功 (実行時間: {execution_time:.2f}秒)")
                self.passed_tests += 1
            else:
                logger.error(f"❌ {test_name} 失敗 (実行時間: {execution_time:.2f}秒)")
                self.failed_tests += 1
            
            self.test_results[test_name] = {
                'success': success,
                'execution_time': execution_time,
                'error': None
            }
            
            return success
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{test_name} でエラーが発生: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.test_results[test_name] = {
                'success': False,
                'execution_time': execution_time,
                'error': error_msg
            }
            
            self.failed_tests += 1
            return False

    def run_all_tests(self) -> bool:
        """すべてのテストを実行"""
        logger.info("🚀 MLトレーニング系包括的テストスイートを開始")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # テストモジュールの定義
        test_modules = [
            ("計算正確性テスト", run_all_calculation_tests),
            ("前処理正確性テスト", run_all_preprocessing_tests),
            ("特徴量計算テスト", run_all_feature_calculation_tests),
            ("データ変換テスト", run_all_data_transformation_tests),
            ("ラベル生成テスト", run_all_label_generation_tests),
            ("エラーハンドリングテスト", run_all_error_handling_tests),
            ("パフォーマンステスト", run_all_performance_tests),
        ]
        
        self.total_tests = len(test_modules)
        
        # 各テストモジュールを実行
        all_passed = True
        for test_name, test_function in test_modules:
            success = self.run_test_module(test_name, test_function)
            if not success:
                all_passed = False
        
        self.end_time = time.time()
        
        # 結果サマリーを表示
        self._display_summary()
        
        return all_passed

    def _display_summary(self):
        """テスト結果のサマリーを表示"""
        total_time = self.end_time - self.start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 テスト結果サマリー")
        logger.info("=" * 80)
        
        logger.info(f"総実行時間: {total_time:.2f}秒")
        logger.info(f"総テスト数: {self.total_tests}")
        logger.info(f"成功: {self.passed_tests}")
        logger.info(f"失敗: {self.failed_tests}")
        logger.info(f"成功率: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        logger.info("\n📋 詳細結果:")
        for test_name, result in self.test_results.items():
            status = "✅ 成功" if result['success'] else "❌ 失敗"
            time_str = f"{result['execution_time']:.2f}秒"
            logger.info(f"  {test_name}: {status} ({time_str})")
            
            if result['error']:
                logger.info(f"    エラー: {result['error']}")
        
        if self.failed_tests == 0:
            logger.info("\n🎉 すべてのテストが正常に完了しました！")
            logger.info("MLトレーニングシステムの計算と前処理の正確性が確認されました。")
        else:
            logger.warning(f"\n⚠️ {self.failed_tests}個のテストが失敗しました。")
            logger.warning("失敗したテストを確認し、問題を修正してください。")

    def run_specific_test(self, test_name: str) -> bool:
        """特定のテストのみを実行"""
        test_mapping = {
            "calculations": ("計算正確性テスト", run_all_calculation_tests),
            "preprocessing": ("前処理正確性テスト", run_all_preprocessing_tests),
            "features": ("特徴量計算テスト", run_all_feature_calculation_tests),
            "transformations": ("データ変換テスト", run_all_data_transformation_tests),
            "labels": ("ラベル生成テスト", run_all_label_generation_tests),
            "errors": ("エラーハンドリングテスト", run_all_error_handling_tests),
            "performance": ("パフォーマンステスト", run_all_performance_tests),
        }
        
        if test_name not in test_mapping:
            logger.error(f"不明なテスト名: {test_name}")
            logger.info(f"利用可能なテスト: {list(test_mapping.keys())}")
            return False
        
        self.start_time = time.time()
        self.total_tests = 1
        
        test_display_name, test_function = test_mapping[test_name]
        success = self.run_test_module(test_display_name, test_function)
        
        self.end_time = time.time()
        self._display_summary()
        
        return success

    def validate_test_environment(self) -> bool:
        """テスト環境の検証"""
        logger.info("🔍 テスト環境を検証中...")
        
        try:
            # 必要なライブラリの確認
            import numpy as np
            import pandas as pd
            import sklearn
            import scipy
            import talib
            
            logger.info("✅ 必要なライブラリが利用可能です")
            
            # プロジェクトモジュールの確認
            from app.utils.data_processing import DataProcessor
            from app.utils.label_generation import LabelGenerator
            from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            logger.info("✅ プロジェクトモジュールが利用可能です")
            
            # 基本的な動作確認
            processor = DataProcessor()
            label_generator = LabelGenerator()
            fe_service = FeatureEngineeringService()
            
            logger.info("✅ 基本的なクラスのインスタンス化が成功しました")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ 必要なライブラリが見つかりません: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ テスト環境の検証でエラーが発生: {e}")
            return False

    def generate_test_report(self, output_file: str = None):
        """テスト結果のレポートを生成"""
        if not self.test_results:
            logger.warning("テスト結果がありません。先にテストを実行してください。")
            return
        
        report_lines = [
            "# MLトレーニング系テスト結果レポート",
            f"実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"総実行時間: {(self.end_time - self.start_time):.2f}秒",
            "",
            "## サマリー",
            f"- 総テスト数: {self.total_tests}",
            f"- 成功: {self.passed_tests}",
            f"- 失敗: {self.failed_tests}",
            f"- 成功率: {(self.passed_tests/self.total_tests)*100:.1f}%",
            "",
            "## 詳細結果"
        ]
        
        for test_name, result in self.test_results.items():
            status = "✅ 成功" if result['success'] else "❌ 失敗"
            report_lines.append(f"### {test_name}")
            report_lines.append(f"- ステータス: {status}")
            report_lines.append(f"- 実行時間: {result['execution_time']:.2f}秒")
            
            if result['error']:
                report_lines.append(f"- エラー: {result['error']}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"📄 テストレポートを保存しました: {output_file}")
            except Exception as e:
                logger.error(f"レポート保存エラー: {e}")
        else:
            logger.info("\n" + report_content)


def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_suite = MLAccuracyTestSuite()
    
    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "validate":
            success = test_suite.validate_test_environment()
            sys.exit(0 if success else 1)
        else:
            success = test_suite.run_specific_test(test_name)
    else:
        # 環境検証
        if not test_suite.validate_test_environment():
            logger.error("テスト環境の検証に失敗しました。")
            sys.exit(1)
        
        # 全テスト実行
        success = test_suite.run_all_tests()
    
    # レポート生成
    test_suite.generate_test_report()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
