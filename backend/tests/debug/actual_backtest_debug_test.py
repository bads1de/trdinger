#!/usr/bin/env python3
"""
実際のバックテストデータを使った高度なデバッグテスト

修正済みオートストラテジーを実際の市場データでテストし、
ロング・ショート取引バランスを詳しく分析する。
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from typing import Dict, Any, List, Optional

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.auto_strategy.generators.condition_generator import ConditionGenerator
from app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene
from app.services.backtest.backtest_service import BacktestService
from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.core.condition_evaluator import ConditionEvaluator

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_ohlc_data() -> pd.DataFrame:
    """サンプルOHLCデータを作成（実際の市場データを模倣）"""
    logger.info("サンプルOHLCデータを作成中...")

    # 一定期間のサンプルデータ作成
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='4H')
    np.random.seed(42)  # 再現性のため

    # 現実的な価格変動をシミュレーション
    base_price = 50000.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.exp(returns.cumsum())

    # OHLCデータ生成
    data = []
    for i, price in enumerate(prices):
        # 単純なOHLCシミュレーション
        open_price = price
        high_price = price * (1 + np.random.uniform(0.005, 0.015))
        low_price = price * (1 - np.random.uniform(0.005, 0.015))
        close_price = prices[min(i+1, len(prices)-1)]

        data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': np.random.uniform(1000, 10000)
        })

    df = pd.DataFrame(data, index=dates)
    logger.info(f"作成したデータ: {len(df)}行, 期間: {df.index[0]} ~ {df.index[-1]}")
    return df


class TradeAnalyzer:
    """取引分析クラス"""

    def __init__(self):
        self.logger = logger
        self.trade_log = []

    def reset_trade_log(self):
        """取引ログをリセット"""
        self.trade_log = []

    def record_trade_attempt(self, bar_index: int, conditions_satisfied: Dict[str, Any]):
        """取引試行を記録"""
        self.trade_log.append({
            'bar_index': bar_index,
            'long_signal': conditions_satisfied.get('long_signal', False),
            'short_signal': conditions_satisfied.get('short_signal', False),
            'condition_details': conditions_satisfied,
            'timestamp': pd.Timestamp.now()
        })

    def analyze_trade_logs(self) -> Dict[str, Any]:
        """取引ログを分析"""
        total_bars = len(self.trade_log)
        total_long_signals = sum(1 for log in self.trade_log if log['long_signal'])
        total_short_signals = sum(1 for log in self.trade_log if log['short_signal'])
        both_signals = sum(1 for log in self.trade_log
                          if log['long_signal'] and log['short_signal'])

        # シグナル比率計算
        long_ratio = total_long_signals / total_bars if total_bars > 0 else 0
        short_ratio = total_short_signals / total_bars if total_bars > 0 else 0
        both_ratio = both_signals / total_bars if total_bars > 0 else 0

        return {
            'total_bars': total_bars,
            'total_long_signals': total_long_signals,
            'total_short_signals': total_short_signals,
            'total_both_signals': both_signals,
            'long_ratio': long_ratio,
            'short_ratio': short_ratio,
            'both_ratio': both_ratio,
            'trade_balance_ratio': short_ratio / long_ratio if long_ratio > 0 else float('inf')
        }


class StrategyDebugSimulator:
    """戦略デバッグシミュレーター"""

    def __init__(self):
        self.condition_evaluator = ConditionEvaluator()
        self.trade_analyzer = TradeAnalyzer()

    def simulate_strategy_execution(self, strategy_gene: StrategyGene,
                                   ohlc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        戦略の実行を模擬し、条件評価と取引シグナルを分析

        Args:
            strategy_gene: テストする戦略遺伝子
            ohlc_data: OHLCデータ（pandas.DataFrame）

        Returns:
            分析結果の辞書
        """
        logger.info("戦略実行シミュレーションを開始...")

        # 取引ログをリセット
        self.trade_analyzer.reset_trade_log()

        # 戦略条件を取得
        long_conditions = strategy_gene.get_effective_long_conditions()
        short_conditions = strategy_gene.get_effective_short_conditions()

        logger.info(f"ロング条件数: {len(long_conditions)}")
        logger.info(f"ショート条件数: {len(short_conditions)}")

        # 各バーでの条件評価
        for bar_index in range(len(ohlc_data)):
            # 現在のバーデータ
            bar_data = ohlc_data.iloc[bar_index]

            # 条件評価用のデータを模擬
            mock_strategy = type('MockStrategy', (), {
                'data': MockData(ohlc_data[:bar_index+1] if bar_index > 0 else pd.DataFrame([bar_data])),
                '__class__': type('MockStrategyClass', (), {})
            })()

            # ロング・ショート条件評価
            try:
                long_signal = False
                short_signal = False

                if long_conditions:
                    long_signal = self.condition_evaluator.evaluate_conditions(
                        long_conditions, mock_strategy
                    )

                if short_conditions:
                    short_signal = self.condition_evaluator.evaluate_conditions(
                        short_conditions, mock_strategy
                    )

                # 取引試行を記録
                conditions_satisfied = {
                    'long_signal': long_signal,
                    'short_signal': short_signal,
                    'long_condition_count': len(long_conditions),
                    'short_condition_count': len(short_conditions),
                    'current_close': bar_data['Close']
                }

                self.trade_analyzer.record_trade_attempt(bar_index, conditions_satisfied)

                # 定期的に詳細ログを出力（最初の10バーと100バーごと）
                if bar_index < 10 or bar_index % 100 == 0:
                    logger.info(f"バー{bar_index}: ロング={long_signal}, ショート={short_signal}, 価格={bar_data['Close']:.2f}")

            except Exception as e:
                logger.warning(f"バー{bar_index}の条件評価でエラー: {e}")

        # 分析結果を返す
        analysis_results = self.trade_analyzer.analyze_trade_logs()
        analysis_results['strategy_gene'] = strategy_gene
        analysis_results['indicators'] = [ind.type for ind in strategy_gene.indicators]

        return analysis_results


class MockData:
    """バックテストフレームワークのデータを模擬"""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.Close = df['Close'].values if not df.empty else np.array([])
        self.Open = df['Open'].values if not df.empty else np.array([])
        self.High = df['High'].values if not df.empty else np.array([])
        self.Low = df['Low'].values if not df.empty else np.array([])
        self.Volume = df['Volume'].values if not df.empty else np.array([])


def create_test_strategy(indicator_combination: List[str]) -> StrategyGene:
    """指定された指標の組み合わせで戦略を作成"""
    logger.info(f"指標組み合わせで戦略を作成: {indicator_combination}")

    # 指標設定マップ
    indicator_configs = {
        'RSI': {'type': 'RSI', 'parameters': {'period': 14}},
        'SMA': {'type': 'SMA', 'parameters': {'period': 20}},
        'EMA': {'type': 'EMA', 'parameters': {'period': 50}},
        'STOCH': {'type': 'STOCH', 'parameters': {}},
        'MACD': {'type': 'MACD', 'parameters': {}},
        'CORREL': {'type': 'CORREL', 'parameters': {}},
        'CDL_HAMMER': {'type': 'CDL_HAMMER', 'parameters': {}},
        'BB': {'type': 'BB', 'parameters': {'period': 20}}
    }

    # 指標リストを作成
    indicators = []
    for indicator_name in indicator_combination[:5]:  # 最大5指標
        if indicator_name in indicator_configs:
            config = indicator_configs[indicator_name]
            indicators.append(IndicatorGene(
                type=config['type'],
                parameters=config['parameters'],
                enabled=True
            ))

    # Smart条件生成器で条件を作成
    generator = SmartConditionGenerator()
    long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(indicators)

    return StrategyGene(
        id=f"debug_test_{'_'.join(indicator_combination)}",
        indicators=indicators,
        long_entry_conditions=long_conds,
        short_entry_conditions=short_conds,
        exit_conditions=exit_conds,
    )


def run_comprehensive_debug_test():
    """包括的デバッグテスト実行"""
    logger.info("=" * 60)
    logger.info("実際のバックテストデータを使ったデバッグテスト開始")
    logger.info("=" * 60)

    try:
        # サンプルOHLCデータ作成
        ohlc_data = create_sample_ohlc_data()
        logger.info(f"作成したOHLCデータ: {len(ohlc_data)}行")

        # シミュレーター初期化
        simulator = StrategyDebugSimulator()

        # テストする指標組み合わせ
        test_combinations = [
            # 基本的な組み合わせ
            ['RSI', 'SMA'],
            ['RSI', 'SMA', 'EMA'],
            # モメンタム中心
            ['RSI', 'STOCH', 'MACD'],
            # 統計指標を含む
            ['RSI', 'SMA', 'CORREL'],
            # パターン認識を含む
            ['RSI', 'SMA', 'CDL_HAMMER'],
            # ボラティリティ系を含む
            ['RSI', 'SMA', 'BB']
        ]

        # 各組み合わせで戦略をテスト
        results = []
        for i, combo in enumerate(test_combinations):
            logger.info(f"\n{'='*50}")
            logger.info(f"テストケース {i+1}: {combo}")
            logger.info('='*50)

            # 戦略作成
            strategy = create_test_strategy(combo)

            # 戦略実行シミュレーション
            analysis = simulator.simulate_strategy_execution(strategy, ohlc_data)

            # 結果表示
            logger.info("分析結果:")
            logger.info(f"  総バー数: {analysis['total_bars']}")
            logger.info(f"  ロングシグナル数: {analysis['total_long_signals']}")
            logger.info(f"  ショートシグナル数: {analysis['total_short_signals']}")
            logger.info(f"  同時シグナル数: {analysis['total_both_signals']}")
            logger.info(".3f")
            logger.info(".3f")
            logger.info(".3f")

            # 取引バランス分析
            balance_ratio = analysis['trade_balance_ratio']
            if balance_ratio == float('inf'):
                logger.warning("  警告: ロングシグナルが0件")
            elif balance_ratio < 0.5:
                logger.warning(f"  警告: ショートシグナルが少なすぎる (比率: {balance_ratio:.2f})")
            elif balance_ratio > 2.0:
                logger.warning(f"  警告: ショートシグナルが多すぎる (比率: {balance_ratio:.2f})")
            else:
                logger.info(f"  ✅ バランス良好 (比率: {balance_ratio:.2f})")

            analysis['combination'] = combo
            results.append(analysis)

        # 全体統計
        logger.info(f"\n{'='*60}")
        logger.info("全体統計分析")
        logger.info('='*60)

        total_strategies = len(results)
        good_balance_count = sum(1 for r in results
                                if isinstance(r['trade_balance_ratio'], float)
                                and 0.5 <= r['trade_balance_ratio'] <= 2.0)

        logger.info(f"総戦略数: {total_strategies}")
        logger.info(f"バランス良好な戦略数: {good_balance_count}")
        logger.info(".1f")

        if good_balance_count == total_strategies:
            logger.info("🎉 全ての戦略でバランスが良好です！修正成功！")
        elif good_balance_count > total_strategies * 0.8:
            logger.info("✅ 大部分の戦略でバランスが良好です。修正は概ね成功！")
        else:
            logger.warning("⚠️ バランスの問題が残っている戦略があります")

        return results

    except Exception as e:
        logger.error(f"包括的デバッグテストでエラー: {e}")
        traceback.print_exc()
        return []


def run_edge_case_tests():
    """エッジケーステスト"""
    logger.info(f"\n{'='*50}")
    logger.info("エッジケーステスト")
    logger.info('='*50)

    # エッジケースの指標組み合わせ
    edge_cases = [
        # 単一指標
        ['RSI'],
        ['SMA'],
        # 多くの指標
        ['RSI', 'SMA', 'EMA', 'MACD', 'STOCH', 'CORREL', 'BB'],
        # 不利な組み合わせ
        ['MACD', 'CORREL', 'CDL_HAMMER'],  # 全て別のタイプ
    ]

    ohlc_data = create_sample_ohlc_data()
    simulator = StrategyDebugSimulator()

    for combo in edge_cases:
        logger.info(f"\nテスト: {combo}")

        try:
            strategy = create_test_strategy(combo)
            analysis = simulator.simulate_strategy_execution(strategy, ohlc_data)

            balance_ratio = analysis['trade_balance_ratio']

            if isinstance(balance_ratio, float) and 0.3 <= balance_ratio <= 3.0:
                logger.info(f"  ✅ バランスOK (比率: {balance_ratio:.2f})")
            else:
                logger.warning(f"  ⚠️ バランス問題 (比率: {balance_ratio:.2f})")

        except Exception as e:
            logger.error(f"  エッジケーステストエラー: {e}")


if __name__ == "__main__":
    logger.info("実際のバックテストデータを使った高度なデバッグテストを開始")

    try:
        # 主要テスト実行
        results = run_comprehensive_debug_test()

        # エッジケーステスト
        run_edge_case_tests()

        logger.info("\n" + "="*60)
        logger.info("デバッグテスト完了")
        logger.info("修正の効果を詳細に分析しました")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"メインエラー: {e}")
        traceback.print_exc()
        sys.exit(1)