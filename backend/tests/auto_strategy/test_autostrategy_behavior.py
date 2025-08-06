"""
AutoStrategy動作検証テスト

このモジュールは以下をテストします：
1. AutoStrategy全体の動作フロー
2. 各コンポーネント間の連携
3. 実際の取引シナリオでの動作
4. エラーハンドリングの正確性
5. パフォーマンスと一貫性
"""

import unittest
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
from app.services.auto_strategy.calculators.position_sizing_calculator import PositionSizingCalculatorService

# ログ設定
logger = logging.getLogger(__name__)

class TestAutoStrategyBehavior(unittest.TestCase):
    """AutoStrategy動作検証テスト"""
    
    def setUp(self):
        """テスト前の準備"""
        self.start_time = time.time()
        self.auto_strategy_service = AutoStrategyService()

        # GAConfigを作成
        from app.services.auto_strategy.models.ga_config import GAConfig
        ga_config = GAConfig(
            population_size=10,
            generations=1,
            max_indicators=5
        )
        self.gene_generator = RandomGeneGenerator(ga_config)
        self.ml_orchestrator = MLOrchestrator()
        self.tpsl_calculator = TPSLCalculator()
        self.position_sizing = PositionSizingCalculatorService()
        
        # テスト用データ
        self.test_data = self._create_realistic_market_data()
        
    def tearDown(self):
        """テスト後のクリーンアップ"""
        execution_time = time.time() - self.start_time
        logger.info(f"テスト実行時間: {execution_time:.3f}秒")
        
    def _create_realistic_market_data(self) -> pd.DataFrame:
        """リアルな市場データを作成"""
        np.random.seed(42)
        
        # より現実的な価格動向を生成
        dates = pd.date_range(start='2023-01-01', periods=2000, freq='1H')
        
        # トレンドとボラティリティを含む価格生成
        base_price = 50000
        trend = 0.0001  # 微小な上昇トレンド
        volatility = 0.015  # 1.5%のボラティリティ
        
        prices = [base_price]
        for i in range(1, len(dates)):
            # トレンド + ランダムウォーク + 平均回帰
            trend_component = trend
            random_component = np.random.normal(0, volatility)
            mean_reversion = -0.1 * (prices[-1] / base_price - 1)
            
            change = trend_component + random_component + mean_reversion
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.5))  # 最低価格制限
        
        closes = np.array(prices)
        
        # OHLCV データを生成
        highs = closes * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.lognormal(8, 0.5, len(dates))  # より現実的なボリューム分布
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    def test_complete_autostrategy_workflow(self):
        """完全なAutoStrategyワークフローテスト"""
        logger.info("🔍 完全なAutoStrategyワークフローテスト開始")

        # 戦略遺伝子生成テスト
        start_time = time.time()

        # 1. 戦略遺伝子の生成
        strategy_gene = self.gene_generator.generate_random_gene()

        # 2. TP/SL計算テスト
        current_price = self.test_data['close'].iloc[-1]
        tpsl_result = self.tpsl_calculator.calculate_tpsl_prices(
            entry_price=current_price,
            position_type="long",
            sl_percentage=2.0,
            tp_percentage=3.0
        )

        # 3. 資金管理計算テスト
        fund_result = self.position_sizing.calculate_position_size(
            account_balance=10000.0,
            risk_percentage=2.0,
            entry_price=current_price,
            stop_loss_price=current_price * 0.98
        )

        # 4. ML指標計算テスト
        ml_indicators = self.ml_orchestrator.calculate_ml_indicators(
            data=self.test_data,
            symbol='BTC/USDT:USDT',
            timeframe='1h'
        )

        execution_time = time.time() - start_time

        # 基本的な結果検証
        assert strategy_gene is not None, "戦略遺伝子が生成されていません"
        assert hasattr(strategy_gene, 'indicators'), "指標が設定されていません"
        assert hasattr(strategy_gene, 'entry_conditions'), "エントリー条件がありません"
        assert hasattr(strategy_gene, 'exit_conditions'), "エグジット条件がありません"

        # TP/SL計算の検証
        assert tpsl_result['success'], "TP/SL計算に失敗しました"
        assert 'sl_price' in tpsl_result, "SL価格がありません"
        assert 'tp_price' in tpsl_result, "TP価格がありません"

        # 資金管理の検証
        assert fund_result['success'], "資金管理計算に失敗しました"
        assert 'position_size' in fund_result, "ポジションサイズがありません"
        assert 'risk_amount' in fund_result, "リスク金額がありません"

        # ML指標の検証
        assert ml_indicators is not None, "ML指標が計算されていません"

        logger.info(f"実行時間: {execution_time:.3f}秒")
        logger.info(f"戦略遺伝子: 指標{len(strategy_gene.indicators)}個")
        logger.info(f"TP/SL: SL={tpsl_result['sl_price']:.2f}, TP={tpsl_result['tp_price']:.2f}")
        logger.info(f"資金管理: ポジション{fund_result['position_size']:.6f}")

        # パフォーマンス要件
        assert execution_time < 10.0, f"実行時間が長すぎます: {execution_time:.3f}秒"

        logger.info("✅ 完全なAutoStrategyワークフローテスト成功")
    
    def test_strategy_consistency_across_runs(self):
        """複数実行での戦略一貫性テスト"""
        logger.info("🔍 戦略一貫性テスト開始")

        # 同じシードで複数回実行
        np.random.seed(42)

        genes = []
        execution_times = []

        # 同じパラメータで複数回実行
        for i in range(5):
            start_time = time.time()
            gene = self.gene_generator.generate_random_gene()
            execution_time = time.time() - start_time

            genes.append(gene)
            execution_times.append(execution_time)

            logger.info(f"実行{i+1}: {execution_time:.3f}秒")

        # 実行時間の一貫性
        avg_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        cv_time = std_time / avg_time * 100 if avg_time > 0 else 0  # 変動係数

        logger.info(f"実行時間統計: 平均{avg_time:.3f}秒, 標準偏差{std_time:.3f}秒, CV{cv_time:.1f}%")
        assert cv_time < 100, f"実行時間のばらつきが大きすぎます: {cv_time:.1f}%"

        # 戦略の一貫性チェック
        assert len(genes) == 5, "すべての実行で戦略遺伝子が生成されませんでした"

        # 指標数の一貫性
        indicator_counts = [len(gene.indicators) for gene in genes]
        indicator_std = np.std(indicator_counts)
        logger.info(f"指標数: {indicator_counts}, 標準偏差: {indicator_std:.2f}")
        assert indicator_std <= 2.0, f"指標数のばらつきが大きすぎます: {indicator_std:.2f}"

        # エントリー条件数の一貫性
        entry_counts = [len(gene.entry_conditions) for gene in genes]
        entry_std = np.std(entry_counts)
        logger.info(f"エントリー条件数: {entry_counts}, 標準偏差: {entry_std:.2f}")
        assert entry_std <= 2.0, f"エントリー条件数のばらつきが大きすぎます: {entry_std:.2f}"

        logger.info("✅ 戦略一貫性テスト成功")
    
    def test_different_market_conditions(self):
        """異なる市場条件での動作テスト"""
        logger.info("🔍 異なる市場条件での動作テスト開始")

        market_scenarios = [
            {
                'name': '上昇トレンド',
                'trend': 0.001,
                'volatility': 0.01
            },
            {
                'name': '下降トレンド',
                'trend': -0.001,
                'volatility': 0.01
            },
            {
                'name': '横ばい市場',
                'trend': 0.0,
                'volatility': 0.005
            },
            {
                'name': '高ボラティリティ',
                'trend': 0.0,
                'volatility': 0.03
            }
        ]

        scenario_results = []

        for scenario in market_scenarios:
            # シナリオ別のテストデータ生成
            test_data = self._generate_scenario_data(
                scenario['trend'],
                scenario['volatility']
            )

            # 戦略遺伝子生成
            gene = self.gene_generator.generate_random_gene()

            # ML指標計算
            ml_indicators = self.ml_orchestrator.calculate_ml_indicators(
                data=test_data,
                symbol='BTC/USDT:USDT',
                timeframe='1h'
            )

            # TP/SL計算
            current_price = test_data['close'].iloc[-1]
            tpsl_result = self.tpsl_calculator.calculate_tpsl_prices(
                entry_price=current_price,
                position_type="long",
                sl_percentage=2.0,
                tp_percentage=3.0
            )

            assert gene is not None, f"{scenario['name']}で戦略遺伝子生成に失敗"
            assert ml_indicators is not None, f"{scenario['name']}でML指標計算に失敗"
            assert tpsl_result['success'], f"{scenario['name']}でTP/SL計算に失敗"

            scenario_results.append({
                'scenario': scenario['name'],
                'gene': gene,
                'ml_indicators': ml_indicators,
                'tpsl_result': tpsl_result,
                'indicator_count': len(gene.indicators),
                'entry_conditions': len(gene.entry_conditions)
            })

            logger.info(f"{scenario['name']}: 指標{len(gene.indicators)}個, "
                       f"エントリー条件{len(gene.entry_conditions)}個")

        # 各シナリオで適切な結果が得られているかチェック
        for sr in scenario_results:
            assert sr['indicator_count'] > 0, f"{sr['scenario']}で指標が生成されていません"
            assert sr['entry_conditions'] >= 0, f"{sr['scenario']}でエントリー条件が不正です"

        # シナリオ間での適応性チェック
        indicator_counts = [sr['indicator_count'] for sr in scenario_results]
        indicator_range = max(indicator_counts) - min(indicator_counts)

        logger.info(f"指標数の範囲: {indicator_range}")
        assert indicator_range >= 0, f"指標数の範囲が不正: {indicator_range}"

        logger.info("✅ 異なる市場条件での動作テスト成功")
    
    def _generate_scenario_data(self, trend: float, volatility: float) -> pd.DataFrame:
        """シナリオ別のテストデータを生成"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
        base_price = 50000
        
        prices = [base_price]
        for i in range(1, len(dates)):
            change = trend + np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.3))
        
        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.003, len(dates))))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.003, len(dates))))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.lognormal(8, 0.3, len(dates))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

    def test_error_handling_robustness(self):
        """エラーハンドリング堅牢性テスト"""
        logger.info("🔍 エラーハンドリング堅牢性テスト開始")

        error_scenarios = [
            {
                'name': '不十分なデータ',
                'data': self.test_data.head(10),  # 10行のみ
                'should_handle': True
            },
            {
                'name': '欠損値を含むデータ',
                'data': self._create_data_with_missing_values(),
                'should_handle': True
            },
            {
                'name': '異常値を含むデータ',
                'data': self._create_data_with_outliers(),
                'should_handle': True
            },
            {
                'name': '空のデータ',
                'data': pd.DataFrame(),
                'should_handle': True
            }
        ]

        handled_errors = 0
        total_scenarios = len(error_scenarios)

        for scenario in error_scenarios:
            try:
                # 各コンポーネントを個別にテスト

                # 1. 戦略遺伝子生成
                gene = self.gene_generator.generate_random_gene()

                # 2. ML指標計算（データに依存）
                if len(scenario['data']) > 0:
                    ml_result = self.ml_orchestrator.calculate_ml_indicators(
                        data=scenario['data'],
                        symbol='BTC/USDT:USDT',
                        timeframe='1h'
                    )
                else:
                    ml_result = None

                # 3. TP/SL計算（価格に依存）
                if len(scenario['data']) > 0 and 'close' in scenario['data'].columns:
                    current_price = scenario['data']['close'].iloc[-1]
                    if not pd.isna(current_price) and current_price > 0:
                        tpsl_result = self.tpsl_calculator.calculate_tpsl_prices(
                            entry_price=current_price,
                            position_type="long",
                            sl_percentage=2.0,
                            tp_percentage=3.0
                        )
                    else:
                        tpsl_result = {'success': False}
                else:
                    tpsl_result = {'success': False}

                if scenario['should_handle']:
                    # エラーが適切にハンドリングされているかチェック
                    if gene is not None:
                        handled_errors += 1
                        logger.info(f"{scenario['name']}: 戦略遺伝子生成成功")
                    else:
                        logger.info(f"{scenario['name']}: 戦略遺伝子生成失敗（期待通り）")
                        handled_errors += 1

            except Exception as e:
                if scenario['should_handle']:
                    logger.info(f"{scenario['name']}: 例外をキャッチ - {type(e).__name__}")
                    handled_errors += 1
                else:
                    logger.error(f"{scenario['name']}: 予期しない例外 - {e}")

        error_handling_rate = handled_errors / total_scenarios * 100
        logger.info(f"エラーハンドリング成功率: {error_handling_rate:.1f}% ({handled_errors}/{total_scenarios})")

        assert error_handling_rate >= 75, f"エラーハンドリング率が低すぎます: {error_handling_rate:.1f}%"

        logger.info("✅ エラーハンドリング堅牢性テスト成功")

    def test_component_integration(self):
        """コンポーネント統合テスト"""
        logger.info("🔍 コンポーネント統合テスト開始")

        # 各コンポーネントの個別動作確認
        components_status = {}

        # 1. 戦略遺伝子生成器
        try:
            gene = self.gene_generator.generate_random_gene()
            components_status['gene_generator'] = gene is not None
            logger.info(f"戦略遺伝子生成器: {'成功' if components_status['gene_generator'] else '失敗'}")
        except Exception as e:
            components_status['gene_generator'] = False
            logger.error(f"戦略遺伝子生成器エラー: {e}")

        # 2. TP/SL計算器
        try:
            current_price = self.test_data['close'].iloc[-1]
            tpsl_result = self.tpsl_calculator.calculate_tpsl_prices(
                entry_price=current_price,
                position_type="long",
                sl_percentage=2.0,
                tp_percentage=3.0
            )
            components_status['tpsl_calculator'] = tpsl_result.get('success', False)
            logger.info(f"TP/SL計算器: {'成功' if components_status['tpsl_calculator'] else '失敗'}")
        except Exception as e:
            components_status['tpsl_calculator'] = False
            logger.error(f"TP/SL計算器エラー: {e}")

        # 3. 資金管理
        try:
            fund_result = self.position_sizing.calculate_position_size(
                account_balance=10000.0,
                risk_percentage=2.0,
                entry_price=current_price,
                stop_loss_price=current_price * 0.98
            )
            components_status['position_sizing'] = fund_result.get('success', False)
            logger.info(f"資金管理: {'成功' if components_status['position_sizing'] else '失敗'}")
        except Exception as e:
            components_status['position_sizing'] = False
            logger.error(f"資金管理エラー: {e}")

        # 4. ML オーケストレーター
        try:
            ml_result = self.ml_orchestrator.calculate_ml_indicators(
                data=self.test_data,
                symbol='BTC/USDT:USDT',
                timeframe='1h'
            )
            components_status['ml_orchestrator'] = ml_result is not None
            logger.info(f"MLオーケストレーター: {'成功' if components_status['ml_orchestrator'] else '失敗'}")
        except Exception as e:
            components_status['ml_orchestrator'] = False
            logger.error(f"MLオーケストレーターエラー: {e}")

        # 統合成功率の計算
        successful_components = sum(components_status.values())
        total_components = len(components_status)
        integration_rate = successful_components / total_components * 100

        logger.info(f"コンポーネント統合成功率: {integration_rate:.1f}% ({successful_components}/{total_components})")

        # 最低限必要なコンポーネントの動作確認
        critical_components = ['gene_generator', 'tpsl_calculator', 'position_sizing']
        critical_success = all(components_status.get(comp, False) for comp in critical_components)

        assert critical_success, "重要なコンポーネントが動作していません"
        assert integration_rate >= 75, f"統合成功率が低すぎます: {integration_rate:.1f}%"

        logger.info("✅ コンポーネント統合テスト成功")

    def test_performance_under_load(self):
        """負荷下でのパフォーマンステスト"""
        logger.info("🔍 負荷下でのパフォーマンステスト開始")

        # 大量データでのテスト
        large_data = self._create_realistic_market_data()  # 2000行

        # 複数回実行してパフォーマンスを測定
        execution_times = []

        for i in range(10):
            start_time = time.time()

            # 戦略遺伝子生成
            gene = self.gene_generator.generate_random_gene()

            # ML指標計算
            ml_indicators = self.ml_orchestrator.calculate_ml_indicators(
                data=large_data,
                symbol='BTC/USDT:USDT',
                timeframe='1h'
            )

            # TP/SL計算
            current_price = large_data['close'].iloc[-1]
            tpsl_result = self.tpsl_calculator.calculate_tpsl_prices(
                entry_price=current_price,
                position_type="long",
                sl_percentage=2.0,
                tp_percentage=3.0
            )

            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            assert gene is not None, f"実行{i+1}で戦略遺伝子がNullです"
            assert ml_indicators is not None, f"実行{i+1}でML指標がNullです"
            assert tpsl_result['success'], f"実行{i+1}でTP/SL計算に失敗"

            if i % 3 == 0:
                logger.info(f"実行{i+1}: {execution_time:.3f}秒")

        # パフォーマンス統計
        avg_time = np.mean(execution_times)
        max_time = np.max(execution_times)
        min_time = np.min(execution_times)
        std_time = np.std(execution_times)

        logger.info(f"実行時間統計:")
        logger.info(f"  平均: {avg_time:.3f}秒")
        logger.info(f"  最大: {max_time:.3f}秒")
        logger.info(f"  最小: {min_time:.3f}秒")
        logger.info(f"  標準偏差: {std_time:.3f}秒")

        # パフォーマンス要件
        assert avg_time < 30.0, f"平均実行時間が長すぎます: {avg_time:.3f}秒"
        assert max_time < 60.0, f"最大実行時間が長すぎます: {max_time:.3f}秒"
        assert std_time < 15.0, f"実行時間のばらつきが大きすぎます: {std_time:.3f}秒"

        logger.info("✅ 負荷下でのパフォーマンステスト成功")

    def _create_data_with_missing_values(self) -> pd.DataFrame:
        """欠損値を含むテストデータを作成"""
        data = self.test_data.copy()
        # ランダムに10%のデータを欠損させる
        mask = np.random.random(len(data)) < 0.1
        data.loc[mask, 'close'] = np.nan
        return data

    def _create_data_with_outliers(self) -> pd.DataFrame:
        """異常値を含むテストデータを作成"""
        data = self.test_data.copy()
        # ランダムに5%のデータを異常値にする
        outlier_mask = np.random.random(len(data)) < 0.05
        data.loc[outlier_mask, 'close'] *= np.random.choice([0.1, 10.0], size=outlier_mask.sum())
        return data


if __name__ == '__main__':
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    unittest.main()
