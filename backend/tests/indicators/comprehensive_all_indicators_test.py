#!/usr/bin/env python3
"""
すべての登録インジケーター（141個以上）の包括的テストスクリプト

TDD方式により、初期化プロセスと計算機能を徹底的にテストし、
成功/失敗を詳細に分類してレポートを出力します。
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestResultCategory(Enum):
    """テスト結果のカテゴリ"""
    SUCCESS = "正常動作"
    NAN_VALUES = "NaN発生"
    ALL_NAN = "全てNaN"
    CALCULATION_ERROR = "計算エラー"
    CONFIG_MISSING = "レジストリ未登録"
    PARAMETER_ERROR = "パラメータエラー"
    INITIALIZATION_ERROR = "初期化エラー"


@dataclass
class IndicatorTestResult:
    """インジケーター単体テスト結果"""
    indicator_name: str
    category: TestResultCategory
    execution_time: float = 0.0
    valid_values_count: Optional[int] = None
    total_values_count: Optional[int] = None
    param_count: Optional[int] = None
    error_message: Optional[str] = None
    config_exists: bool = False
    calculation_attempted: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComprehensiveTestReport:
    """包括的テストレポート"""
    total_indicators: int = 0
    results_by_category: Dict[TestResultCategory, List[IndicatorTestResult]] = field(default_factory=dict)
    execution_summary: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: IndicatorTestResult):
        """結果を追加"""
        if result.category not in self.results_by_category:
            self.results_by_category[result.category] = []
        self.results_by_category[result.category].append(result)

    def get_summary_stats(self) -> Dict[str, Any]:
        """サマリー統計を生成"""
        stats = {}
        for category in TestResultCategory:
            results = self.results_by_category.get(category, [])
            stats[category.value] = len(results)

        total_time = sum(r.execution_time for category_results in self.results_by_category.values()
                        for r in category_results)

        success_count = len(self.results_by_category.get(TestResultCategory.SUCCESS, []))
        stats['全体成功率'] = f"{(success_count / self.total_indicators * 100):.1f}%" if self.total_indicators > 0 else "0%"
        stats['総実行時間（秒）'] = round(total_time, 2)
        stats['平均実行時間（秒）'] = f"{(total_time / self.total_indicators):.4f}" if self.total_indicators > 0 else "0"

        return stats


def create_test_data():
    """テスト用価格データ作成"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1H')

    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 500)

    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(1, new_price))

    high_prices = [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices]
    low_prices = [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices]
    open_prices = [base_price * (1 + np.random.normal(0, 0.005))] + [
        close_prices[i-1] * (1 + np.random.normal(0, 0.005)) for i in range(1, len(close_prices))
    ]
    volumes = np.random.uniform(1000000, 10000000, 500)

    # MAEインジケーター用の予測値生成
    predicted_prices = [price * (1 + np.random.normal(0, 0.01)) for price in close_prices]

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'predicted': predicted_prices,  # MAE用予測値
        'volume': volumes
    })

    return df


def get_all_registered_indicators():
    """全登録インジケーター取得"""
    try:
        indicators = indicator_registry.list_indicators()
        if not indicators:
            logger.warning("インジケータレジストリが空です")
            return []

        # STCおよびTHERMOインジケーターを除外（タスク要件により削除済み）
        indicators = [ind for ind in indicators if ind not in ['STC', 'THERMO']]

        logger.info(f"全{len(indicators)}個のインジケーターを取得しました（STC, THERMO除外）")
        return indicators
    except Exception as e:
        logger.error(f"インジケーター一覧取得エラー: {e}")
        return []


def test_indicator_initialization(indicator_name: str) -> Tuple[bool, Optional[str]]:
    """インジケーター初期化テスト"""
    try:
        config = indicator_registry.get_indicator_config(indicator_name)
        if config is None:
            return False, "レジストリに設定が見つかりません"

        # 基本的な設定属性の検証
        required_attrs = ["indicator_name", "result_type", "scale_type"]
        for attr in required_attrs:
            if not hasattr(config, attr):
                return False, f"必須属性 '{attr}' が設定に存在しません"

        # アダプター関数の存在確認（ある場合のみ）
        if hasattr(config, 'adapter_function') and config.adapter_function is None:
            logger.warning(f"{indicator_name}: adapter_function が None です")

        return True, None

    except Exception as e:
        return False, str(e)


def test_indicator_calculation(df: DataFrame, indicator_name: str) -> Tuple[TestResultCategory, Optional[np.ndarray], Optional[str]]:
    """インジケーター計算テスト"""
    service = TechnicalIndicatorService()

    try:
        config = indicator_registry.get_indicator_config(indicator_name)
        if config is None:
            return TestResultCategory.CONFIG_MISSING, None, "設定が見つかりません"

        # パラメータ生成（可能であれば）
        params = {}
        try:
            if hasattr(config, 'parameters') and config.parameters:
                # デフォルトパラメータを使用
                for param_name, param_config in config.parameters.items():
                    params[param_name] = param_config.default_value
            elif hasattr(config, 'get_parameter_ranges'):
                ranges = config.get_parameter_ranges()
                for param_name, param_range in ranges.items():
                    params[param_name] = param_range.get('default', 14)
        except Exception as e:
            logger.warning(f"{indicator_name} パラメータ生成警告: {e}")

        # 計算実行
        result = service.calculate_indicator(df, indicator_name, params)

        if result is None:
            return TestResultCategory.ALL_NAN, None, "計算結果が None です"

        # 結果検証
        if isinstance(result, np.ndarray):
            valid_count = np.sum(~np.isnan(result))
            total_count = len(result)

            if valid_count == 0:
                return TestResultCategory.ALL_NAN, result, None
            elif valid_count < total_count:
                return TestResultCategory.NAN_VALUES, result, f"{valid_count}/{total_count} 個の値が有効"

            return TestResultCategory.SUCCESS, result, None

        elif isinstance(result, tuple):
            valid_counts = [np.sum(~np.isnan(arr)) for arr in result if isinstance(arr, np.ndarray)]
            if not valid_counts:
                return TestResultCategory.ALL_NAN, None, "すべての配列が無効"

            total_valid = sum(valid_counts)
            total_arrays = len(result)

            if total_valid == 0:
                return TestResultCategory.ALL_NAN, result, None

            return TestResultCategory.SUCCESS, result, f"{len(valid_counts)}/{total_arrays} 配列に有効値"

        elif isinstance(result, pd.Series) or isinstance(result, pd.DataFrame):
            valid_count = result.notna().sum()
            total_count = result.shape[0]

            if valid_count == 0:
                return TestResultCategory.ALL_NAN, result, None

            return TestResultCategory.SUCCESS, result, f"{valid_count}/{total_count} 個の値が有効"

        else:
            return TestResultCategory.ALL_NAN, result, f"予期せぬ結果型: {type(result)}"

    except Exception as e:
        return TestResultCategory.CALCULATION_ERROR, None, str(e)


def execute_single_indicator_test(df: DataFrame, indicator_name: str) -> IndicatorTestResult:
    """単一インジケーターの全面テスト実行"""
    start_time = time.time()

    result = IndicatorTestResult(indicator_name=indicator_name, category=TestResultCategory.SUCCESS)

    # 1. 初期化テスト
    init_success, init_error = test_indicator_initialization(indicator_name)
    result.config_exists = init_success

    if not init_success:
        result.category = TestResultCategory.INITIALIZATION_ERROR
        result.error_message = init_error
        result.execution_time = time.time() - start_time

        # 設定が存在しない場合は詳細テストスキップ
        if init_error == "レジストリに設定が見つかりません":
            result.category = TestResultCategory.CONFIG_MISSING

        return result

    # 2. 計算テスト
    calc_category, calc_result, calc_details = test_indicator_calculation(df, indicator_name)
    result.category = calc_category
    result.calculation_attempted = True

    if calc_result is not None:
        if isinstance(calc_result, np.ndarray):
            result.valid_values_count = np.sum(~np.isnan(calc_result))
            result.total_values_count = len(calc_result)
        elif isinstance(calc_result, tuple):
            valid_counts = [np.sum(~np.isnan(arr)) for arr in calc_result if isinstance(arr, np.ndarray)]
            result.valid_values_count = sum(valid_counts) if valid_counts else 0
            result.total_values_count = sum(len(arr) for arr in calc_result if isinstance(arr, np.ndarray))
        elif isinstance(calc_result, (pd.Series, pd.DataFrame)):
            result.valid_values_count = calc_result.notna().sum()
            result.total_values_count = calc_result.shape[0]

    if calc_details:
        result.details['calculation_details'] = calc_details

    if result.category == TestResultCategory.CALCULATION_ERROR:
        result.error_message = calc_details

    result.execution_time = time.time() - start_time
    return result


def execute_batch_tests(indicators: List[str], max_workers: int = 4) -> ComprehensiveTestReport:
    """バッチテスト実行（並列処理）"""
    df = create_test_data()
    report = ComprehensiveTestReport()
    report.total_indicators = len(indicators)

    logger.info(f"{len(indicators)}個のインジケーターを{max_workers}ワーカーでテスト実行開始")

    start_total = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_indicator = {executor.submit(execute_single_indicator_test, df, indicator): indicator
                              for indicator in indicators}

        for future in as_completed(future_to_indicator):
            indicator = future_to_indicator[future]
            try:
                result = future.result()
                report.add_result(result)

                completed += 1
                if completed % 10 == 0:
                    logger.info(f"{completed}/{len(indicators)} 個のインジケーター完了")

            except Exception as e:
                logger.error(f"{indicator} テスト実行エラー: {e}")
                error_result = IndicatorTestResult(
                    indicator_name=indicator,
                    category=TestResultCategory.CALCULATION_ERROR,
                    error_message=str(e)
                )
                report.add_result(error_result)

    report.execution_summary['总実行時間'] = time.time() - start_total
    logger.info("すべてのインジケーターテストが完了しました")

    return report


def generate_test_report(report: ComprehensiveTestReport, output_path: Optional[str] = None) -> str:
    """テストレポート生成"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "=" * 80,
        f"包括的テクニカルインジケーターテストレポート",
        f"実行時刻: {timestamp}",
        "=" * 80,
        f"",
        f"総テストインジケーター数: {report.total_indicators}",
        f"",
        f"【結果サマリー】"
    ]

    summary_stats = report.get_summary_stats()
    for key, value in summary_stats.items():
        report_lines.append(f"{key}: {value}")

    report_lines.extend([
        f"",
        f"【カテゴリ別詳細】"
    ])

    for category in TestResultCategory:
        results = report.results_by_category.get(category, [])
        if results:
            report_lines.extend([
                f"",
                f"{category.value} ({len(results)}個):"
            ])

            # アメリカンドル並び順でソート
            sorted_results = sorted(results, key=lambda x: x.indicator_name)

            for result in sorted_results:
                status = "✓" if result.category == TestResultCategory.SUCCESS else "✗"
                duration = f"{result.execution_time:.4f}s"
                info = f"{duration}"

                if result.valid_values_count is not None and result.total_values_count is not None:
                    info += f", 有効値: {result.valid_values_count}/{result.total_values_count}"

                if result.param_count is not None:
                    info += f", パラメータ: {result.param_count}"

                report_lines.append(f"  {status} {result.indicator_name} - {info}")

                if result.error_message:
                    report_lines.append(f"    エラー: {result.error_message}")

                if result.details:
                    for key, value in result.details.items():
                        report_lines.append(f"    {key}: {value}")

    report_lines.extend([
        f"",
        "=" * 80
    ])

    # 文字列形式で取得
    full_report = "\n".join(report_lines)

    # ファイル出力（オプション）
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        logger.info(f"レポートを保存しました: {output_path}")

    return full_report


def main():
    """メイン実行関数"""
    print("=== 全テクニカルインジケータ総合テスト実行開始 ===")

    try:
        # 全インジケーター取得（STC除外済み）
        indicators = get_all_registered_indicators()
        if not indicators:
            print("ERROR: インジケーターが登録されていません")
            return False

        print(f"INFO: {len(indicators)}個のインジケーターに対してテストを実行")

        # STCの除外確認
        if 'STC' in indicators:
            print("WARNING: STCインジケーターがまだテスト対象に含まれています")
        else:
            print("SUCCESS: STCインジケーターが完全に除外されました")

        # バッチテスト実行
        report = execute_batch_tests(indicators, max_workers=4)

        # レポート生成と表示
        import os
        # レポート出力先のディレクトリ作成
        os.makedirs(os.path.dirname("backend/tests/indicators/comprehensive_all_indicators_report.txt"), exist_ok=True)
        output_file = "backend/tests/indicators/comprehensive_all_indicators_report.txt"
        full_report = generate_test_report(report, output_file)

        print("\n" + "="*50)
        print("TEST REPORT")
        print("="*50)
        print(full_report)

        # 全体成功率チェック
        success_count = len(report.results_by_category.get(TestResultCategory.SUCCESS, []))
        success_rate = (success_count / len(indicators) * 100) if indicators else 0

        print("".join(">=" * 50))
        print(f"最終結果: {success_count}/{len(indicators)} 成功 ({success_rate:.1f}%)")

        if success_rate >= 90:
            print("SUCCESS: 90%以上のインジケーターが正常動作")
            return True
        elif success_rate >= 75:
            print("WARNING: 75-90%のインジケーターが正常動作、一部の改善が必要")
            return True
        else:
            print("ERROR: 75%以下のインジケーターのみ正常動作、系統的な問題あり")
            return False

    except Exception as e:
        logger.error(f"テスト実行中に致命的エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)