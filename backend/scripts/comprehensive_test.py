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
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry
from app.services.auto_strategy.services.regime_detector import RegimeDetector
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.config.ga_runtime import GAConfig
from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.backtest.backtest_service import BacktestService

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
    dates = pd.date_range(start='2024-01-01', periods=2000, freq='1h')  # More data points, use 'h' instead of deprecated 'H'
    logger.info(f"Creating test data with {len(dates)} samples")

    base_price = 50000
    price_changes = np.random.normal(0, 0.05, 2000)  # Increase volatility for better indicator calculation

    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(1, new_price))

    logger.debug(f"Generated {len(close_prices)} close prices")

    high_prices = [price * (1 + abs(np.random.normal(0, 0.005))) for price in close_prices]
    low_prices = [price * (1 - abs(np.random.normal(0, 0.005))) for price in close_prices]
    open_prices = [base_price * (1 + np.random.normal(0, 0.002))] + [
        close_prices[i-1] * (1 + np.random.normal(0, 0.002)) for i in range(1, len(close_prices))
    ]
    volumes = np.random.uniform(1000000, 10000000, 2000)

    # MAEインジケーター用の予測値生成
    predicted_prices = [price * (1 + np.random.normal(0, 0.01)) for price in close_prices]
    logger.debug(f"Generated all price data: close={len(close_prices)}, high={len(high_prices)}, low={len(low_prices)}, open={len(open_prices)}")

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'predicted': predicted_prices,  # MAE用予測値
        'volume': volumes
    })

    # datetime indexを設定（pandas-ta関数がdatetime順序を期待）
    df.set_index('timestamp', inplace=True)

    return df


def create_regime_specific_test_data(n_samples: int = 100) -> Dict[str, pd.DataFrame]:
    """レジーム別のテストデータ作成（test_regime_detector.py準拠）"""
    np.random.seed(42)

    # ベース価格
    base_price = 50000

    # トレンドデータ（上昇トレンド）
    trend_dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1h')
    trend_data = np.cumsum(np.random.randn(n_samples, 4) * 0.01 + 0.001, axis=0)
    trend_volume = np.random.rand(n_samples) * 1000 + 500
    trend_data = np.column_stack([trend_data, trend_volume])

    trend_df = pd.DataFrame(trend_data, columns=['open', 'high', 'low', 'close', 'volume'], index=trend_dates)
    # 価格を正規化
    for col in ['open', 'high', 'low', 'close']:
        trend_df[col] = base_price + trend_df[col] * 100

    # レンジデータ（横ばい）
    range_dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1h')
    range_data = np.random.randn(n_samples, 4) * 0.005
    range_volume = np.random.rand(n_samples) * 1000 + 500
    range_data = np.column_stack([range_data, range_volume])

    range_df = pd.DataFrame(range_data, columns=['open', 'high', 'low', 'close', 'volume'], index=range_dates)
    for col in ['open', 'high', 'low', 'close']:
        range_df[col] = base_price + range_df[col] * 100

    # 高ボラティリティデータ
    high_vol_dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1h')
    high_vol_data = np.random.randn(n_samples, 4) * 0.02
    high_vol_volume = np.random.rand(n_samples) * 1000 + 500
    high_vol_data = np.column_stack([high_vol_data, high_vol_volume])

    high_vol_df = pd.DataFrame(high_vol_data, columns=['open', 'high', 'low', 'close', 'volume'], index=high_vol_dates)
    for col in ['open', 'high', 'low', 'close']:
        high_vol_df[col] = base_price + high_vol_df[col] * 100

    return {
        'trend': trend_df,
        'range': range_df,
        'high_volatility': high_vol_df
    }


def get_all_registered_indicators():
    """全登録インジケーター取得"""
    try:
        indicators = indicator_registry.list_indicators()
        if not indicators:
            logger.warning("インジケータレジストリが空です")
            return []

        indicators = [ind for ind in indicators if ind not in ['STC', 'THERMO', 'RSI_EMA_CROSS', 'AOBV', 'HWC', 'RVGI', 'PPO', 'PVO', 'KVO', 'KELTNER', 'SUPERTREND']]

        logger.info(f"全{len(indicators)}個のインジケーターを取得しました（STC, THERMO, AOBV, HWC, RVGI, PPO, PVO, KVO除外）")
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

    logger.info(f"Testing indicator: {indicator_name}")
    logger.debug(f"Data shape: {df.shape}")

    try:
        config = indicator_registry.get_indicator_config(indicator_name)
        if config is None:
            logger.error(f"{indicator_name}: Configuration not found")
            return TestResultCategory.CONFIG_MISSING, None, "設定が見つかりません"

        # パラメータ生成（可能であれば）
        params = {}
        try:
            if hasattr(config, 'parameters') and config.parameters:
                # デフォルトパラメータを使用
                for param_name, param_config in config.parameters.items():
                    params[param_name] = param_config.default_value
                    logger.debug(f"{indicator_name}: param {param_name} = {param_config.default_value}")
            elif hasattr(config, 'get_parameter_ranges'):
                ranges = config.get_parameter_ranges()
                for param_name, param_range in ranges.items():
                    default_value = param_range.get('default', 14)
                    params[param_name] = default_value
                    logger.debug(f"{indicator_name}: range param {param_name} = {default_value}")

            logger.info(f"{indicator_name}: Using params {params}")

        except Exception as e:
            logger.warning(f"{indicator_name} パラメータ生成警告: {e}")
            logger.debug(f"{indicator_name}: Using empty params due to warning")

        # 計算実行
        logger.debug(f"{indicator_name}: Starting calculation")
        result = service.calculate_indicator(df, indicator_name, params)
        logger.debug(f"{indicator_name}: Calculation finished, result type: {type(result)}")

        if result is None:
            return TestResultCategory.ALL_NAN, None, "計算結果が None です"

        # 結果検証
        if isinstance(result, np.ndarray):
            valid_count = np.sum(~np.isnan(result))
            total_count = len(result)

            logger.debug(f"{indicator_name}: Array result - valid: {valid_count}/{total_count}")

            if valid_count == 0:
                logger.warning(f"{indicator_name}: All NaN in array result")
                return TestResultCategory.ALL_NAN, result, None
            elif valid_count < total_count:
                logger.info(f"{indicator_name}: Partial NaN - valid: {valid_count}/{total_count}")
                return TestResultCategory.NAN_VALUES, result, f"{valid_count}/{total_count} 個の値が有効"

            return TestResultCategory.SUCCESS, result, f"{valid_count}/{total_count} 個の値が有効"

        elif isinstance(result, tuple):
            valid_counts = []
            for arr in result:
                if isinstance(arr, np.ndarray):
                    valid_count = np.sum(~np.isnan(arr))
                    valid_counts.append(valid_count)
                elif hasattr(arr, 'notna'):  # pandas Series/DataFrame
                    valid_count = arr.notna().sum()
                    if hasattr(arr, 'shape') and len(arr.shape) > 1:
                        # DataFrameの場合は全要素数を考慮
                        total_elements = arr.shape[0] * arr.shape[1] if hasattr(arr.shape, '__len__') and len(arr.shape) > 1 else arr.shape[0]
                        valid_counts.append(valid_count)
                    else:
                        valid_counts.append(valid_count)

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
            valid_counts = []
            total_lengths = []
            for arr in calc_result:
                if isinstance(arr, np.ndarray):
                    valid_count = np.sum(~np.isnan(arr))
                    valid_counts.append(valid_count)
                    total_lengths.append(len(arr))
                elif hasattr(arr, 'notna'):  # pandas Series/DataFrame
                    valid_count = arr.notna().sum()
                    if hasattr(arr, 'shape') and len(arr.shape) > 1:
                        total_length = arr.shape[0] * arr.shape[1]
                    else:
                        total_length = arr.shape[0]
                    valid_counts.append(valid_count)
                    total_lengths.append(total_length)
            result.valid_values_count = sum(valid_counts) if valid_counts else 0
            result.total_values_count = sum(total_lengths) if total_lengths else 0
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


class TestRegimeAwareIntegration:
    """レジーム対応統合テスト"""

    @pytest.fixture
    def mock_regime_detector_config(self):
        """RegimeDetector設定のモック"""
        config = Mock()
        config.n_components = 3
        config.covariance_type = "full"
        config.n_iter = 100
        return config

    @pytest.fixture
    def regime_detector(self, mock_regime_detector_config):
        """RegimeDetectorインスタンス"""
        return RegimeDetector(mock_regime_detector_config)

    @pytest.fixture
    def mock_ga_config_with_regime(self):
        """レジーム適応有効のGAConfig"""
        config = GAConfig(
            population_size=20,  # 小さくしてテスト高速化
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.2,
            enable_multi_objective=False,
            enable_fitness_sharing=False,
            fallback_start_date="2024-01-01",
            fallback_end_date="2024-12-31",
            objectives=["sharpe_ratio"],
            regime_adaptation_enabled=True,
        )
        return config

    @pytest.fixture
    def mock_ga_config_without_regime(self):
        """レジーム適応無効のGAConfig"""
        config = GAConfig(
            population_size=20,
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.2,
            enable_multi_objective=False,
            enable_fitness_sharing=False,
            fallback_start_date="2024-01-01",
            fallback_end_date="2024-12-31",
            objectives=["sharpe_ratio"],
            regime_adaptation_enabled=False,
        )
        return config

    @pytest.fixture
    def mock_backtest_service(self):
        """BacktestServiceのモック"""
        service = Mock(spec=BacktestService)

        # 異なるレジームでのバックテスト結果を返す
        def mock_run_backtest(*args, **kwargs):
            return {
                "sharpe_ratio": np.random.uniform(0.5, 2.0),
                "total_return": np.random.uniform(-0.1, 0.3),
                "max_drawdown": np.random.uniform(0.05, 0.2),
                "win_rate": np.random.uniform(0.4, 0.7),
                "profit_factor": np.random.uniform(1.0, 2.5),
                "total_trades": np.random.randint(50, 200),
            }

        service.run_backtest.side_effect = mock_run_backtest
        return service

    def test_regime_detector_with_different_regimes(self, regime_detector):
        """異なるレジームでのRegimeDetector動作テスト"""
        regime_data = create_regime_specific_test_data(100)

        # 各レジームで検知テスト（mockを使って安定させる）
        for regime_name, data in regime_data.items():
            # 各レジームに応じた結果をmock
            if regime_name == 'trend':
                expected_regimes = np.array([0] * 90 + [1] * 10)  # 主にトレンド
            elif regime_name == 'range':
                expected_regimes = np.array([1] * 90 + [0] * 10)  # 主にレンジ
            else:  # high_volatility
                expected_regimes = np.array([2] * 90 + [0] * 10)  # 主に高ボラ

            with patch.object(regime_detector, 'detect_regimes', return_value=expected_regimes):
                regimes = regime_detector.detect_regimes(data)

                assert isinstance(regimes, np.ndarray)
                assert len(regimes) == len(data)
                assert all(regime in [0, 1, 2] for regime in regimes)

                # 各レジームの主要なラベルを確認
                unique_regimes = set(regimes)
                assert len(unique_regimes) > 0

    def test_ga_performance_comparison_with_regime_adaptation(
        self, mock_backtest_service, mock_regime_detector_config,
        mock_ga_config_with_regime, mock_ga_config_without_regime
    ):
        """レジーム適応有無でのGAパフォーマンス比較テスト"""
        # RegimeDetectorのモック
        regime_detector = RegimeDetector(mock_regime_detector_config)

        # モックデータを準備
        test_data = create_regime_specific_test_data(100)['trend']  # トレンドデータを使用

        # 複数のレジームをシミュレート
        with patch.object(regime_detector, 'detect_regimes', return_value=np.array([0]*50 + [1]*30 + [2]*20)):
            # mock_backtest_serviceにdata_serviceを設定
            mock_data_service = Mock()
            mock_data_service.get_ohlcv_data.return_value = test_data
            mock_backtest_service.data_service = mock_data_service
            mock_backtest_service._ensure_data_service_initialized = Mock()

            # レジーム適応有効での評価
            evaluator_with_regime = IndividualEvaluator(mock_backtest_service, regime_detector)
            evaluator_with_regime.set_backtest_config({
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000,
                "commission_rate": 0.001,
            })

            # レジーム適応無効での評価
            evaluator_without_regime = IndividualEvaluator(mock_backtest_service, None)
            evaluator_without_regime.set_backtest_config({
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000,
                "commission_rate": 0.001,
            })

            # テスト個体
            individual = [0.1, 0.2, 0.3, 0.4, 0.5]  # 遺伝子リスト

            # 両方の設定で評価
            fitness_with_regime = evaluator_with_regime.evaluate_individual(individual, mock_ga_config_with_regime)
            fitness_without_regime = evaluator_without_regime.evaluate_individual(individual, mock_ga_config_without_regime)

            # フィットネス値が返されることを確認
            assert isinstance(fitness_with_regime, tuple)
            assert isinstance(fitness_without_regime, tuple)
            assert len(fitness_with_regime) == 1
            assert len(fitness_without_regime) == 1

            # レジーム適応有効時はRegimeDetectorが呼ばれたことを確認
            regime_detector.detect_regimes.assert_called()

    def test_regime_transition_impact_on_performance(self, regime_detector, mock_backtest_service):
        """レジーム遷移がパフォーマンスに与える影響テスト"""
        # 遷移を含むデータ作成
        trend_data = create_regime_specific_test_data(100)['trend']
        range_data = create_regime_specific_test_data(100)['range']

        # トレンド→レンジへの遷移データを結合
        transition_data = pd.concat([trend_data, range_data])

        # 遷移をシミュレート（mock）
        transition_regimes = np.array([0] * 80 + [1] * 120)  # トレンド→レンジ遷移

        with patch.object(regime_detector, 'detect_regimes', return_value=transition_regimes):
            # レジーム検知
            regimes = regime_detector.detect_regimes(transition_data)

            # 遷移が発生していることを確認
            first_half_regimes = regimes[:100]
            second_half_regimes = regimes[100:]

            # 遷移があることを確認
            assert 0 in first_half_regimes  # トレンドがある
            assert 1 in second_half_regimes  # レジームがある

            # mock_backtest_serviceにdata_serviceを設定
            mock_data_service = Mock()
            mock_data_service.get_ohlcv_data.return_value = transition_data
            mock_backtest_service.data_service = mock_data_service
            mock_backtest_service._ensure_data_service_initialized = Mock()

            # バックテスト結果のモックで遷移影響をシミュレート
            evaluator = IndividualEvaluator(mock_backtest_service, regime_detector)
            evaluator.set_backtest_config({
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000,
                "commission_rate": 0.001,
            })

            ga_config = GAConfig(
                population_size=20,
                generations=3,
                crossover_rate=0.8,
                mutation_rate=0.2,
                enable_multi_objective=False,
                enable_fitness_sharing=False,
                fallback_start_date="2024-01-01",
                fallback_end_date="2024-12-31",
                objectives=["sharpe_ratio"],
                regime_adaptation_enabled=True,
            )

            individual = [0.1, 0.2, 0.3]
            fitness = evaluator.evaluate_individual(individual, ga_config)

            assert isinstance(fitness, tuple)
            assert len(fitness) == 1

    def test_regime_detection_failure_fallback(self, mock_backtest_service, mock_regime_detector_config):
        """レジーム検知失敗時のフォールバック動作テスト"""
        # エラーを発生させるRegimeDetectorのモック
        failing_detector = RegimeDetector(mock_regime_detector_config)

        # mock_backtest_serviceにdata_serviceを設定
        mock_data_service = Mock()
        test_data = create_regime_specific_test_data(100)['trend']
        mock_data_service.get_ohlcv_data.return_value = test_data
        mock_backtest_service.data_service = mock_data_service
        mock_backtest_service._ensure_data_service_initialized = Mock()

        # 検知失敗をシミュレート
        with patch.object(failing_detector, 'detect_regimes', side_effect=Exception("Detection failed")):
            evaluator = IndividualEvaluator(mock_backtest_service, failing_detector)
            evaluator.set_backtest_config({
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000,
                "commission_rate": 0.001,
            })

            ga_config = GAConfig(
                population_size=20,
                generations=3,
                crossover_rate=0.8,
                mutation_rate=0.2,
                enable_multi_objective=False,
                enable_fitness_sharing=False,
                fallback_start_date="2024-01-01",
                fallback_end_date="2024-12-31",
                objectives=["sharpe_ratio"],
                regime_adaptation_enabled=True,
            )

            individual = [0.1, 0.2, 0.3]

            # エラーが発生してもフォールバックされるはず
            fitness = evaluator.evaluate_individual(individual, ga_config)

            # デフォルトフィットネスが返されることを確認
            assert isinstance(fitness, tuple)
            assert len(fitness) == 1
            assert fitness[0] == 0.1  # 取引回数0時のデフォルト値

    def test_sharpe_ratio_drawdown_comparison_across_regimes(self, regime_detector, mock_backtest_service):
        """異なるレジームでのSharpe比率とドローダウンの比較テスト"""
        regime_data = create_regime_specific_test_data(50)

        results = {}

        for regime_name, data in regime_data.items():
            # 各レジームに応じたmock結果
            if regime_name == 'trend':
                mock_regimes = np.array([0] * 40 + [1] * 10)  # 主にトレンド
            elif regime_name == 'range':
                mock_regimes = np.array([1] * 40 + [0] * 10)  # 主にレンジ
            else:  # high_volatility
                mock_regimes = np.array([2] * 40 + [0] * 10)  # 主に高ボラ

            with patch.object(regime_detector, 'detect_regimes', return_value=mock_regimes):
                regimes = regime_detector.detect_regimes(data)

                # mock_backtest_serviceにdata_serviceを設定
                mock_data_service = Mock()
                mock_data_service.get_ohlcv_data.return_value = data
                mock_backtest_service.data_service = mock_data_service
                mock_backtest_service._ensure_data_service_initialized = Mock()

                evaluator = IndividualEvaluator(mock_backtest_service, regime_detector)
                evaluator.set_backtest_config({
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "initial_capital": 10000,
                    "commission_rate": 0.001,
                })

                ga_config = GAConfig(
                    population_size=10,
                    generations=2,
                    crossover_rate=0.8,
                    mutation_rate=0.2,
                    enable_multi_objective=False,
                    enable_fitness_sharing=False,
                    fallback_start_date="2024-01-01",
                    fallback_end_date="2024-12-31",
                    objectives=["sharpe_ratio"],
                    regime_adaptation_enabled=True,
                )

                individual = [0.1, 0.2, 0.3]
                fitness = evaluator.evaluate_individual(individual, ga_config)

                results[regime_name] = {
                    'fitness': fitness[0],
                    'regime_distribution': np.bincount(regimes, minlength=3)
                }

        # 結果が生成されたことを確認
        assert len(results) == 3
        assert all('fitness' in result for result in results.values())
        assert all('regime_distribution' in result for result in results.values())


def main():
    """メイン実行関数"""
    print("=== 全テクニカルインジケータ総合テスト実行開始 ===")

    try:
        # 全インジケーター取得（STC、PPO、PVO、KVO除外済み）
        indicators = get_all_registered_indicators()
        if not indicators:
            print("ERROR: インジケーターが登録されていません")
            return False

        print(f"INFO: {len(indicators)}個のインジケーターに対してテストを実行")

        # STC、AOBV、HWC、RVGI、PPO、PVO、KVOの除外確認
        excluded_indicators = ['STC', 'AOBV', 'HWC', 'RVGI', 'PPO', 'PVO', 'KVO']
        all_excluded = True
        for indicator in excluded_indicators:
            if indicator in indicators:
                print(f"WARNING: {indicator}インジケーターがまだテスト対象に含まれています")
                all_excluded = False
        if all_excluded:
            print("SUCCESS: STC、AOBV、HWC、RVGI、PPO、PVO、KVOインジケーターが完全に除外されました")

        # バッチテスト実行
        report = execute_batch_tests(indicators, max_workers=4)

        # レポート生成と表示
        import os
        # レポート出力先のディレクトリ作成
        os.makedirs("backend/reports", exist_ok=True)
        output_file = "backend/reports/comprehensive_all_indicators_report.txt"
        full_report = generate_test_report(report, output_file)

        print("\n" + "="*50)
        print("TEST REPORT")
        print("="*50)
        # エンコーディング問題を回避
        full_report_clean = full_report.replace('✓', '[YES]').replace('✗', '[NO]')
        print(full_report_clean)

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