#!/usr/bin/env python3
"""
簡易版実際のバックテストデータを使ったデバッグテスト

修正済みオートストラテジーを実際の条件評価でテスト。
"""

import logging
import sys
from pathlib import Path
import numpy as np
import traceback

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene
from app.services.auto_strategy.core.condition_evaluator import ConditionEvaluator

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockStrategy:
    """バックテスト用モック戦略"""

    def __init__(self, price_data, volume_data=None):
        self.data = self
        self.Close = price_data
        self.Open = price_data  # 簡易化
        self.High = price_data * 1.01  # 簡易化
        self.Low = price_data * 0.99   # 簡易化
        self.Volume = volume_data or np.ones_like(price_data)


def create_synthetic_price_data(length=200):
    """合成価格データを作成（簡易版）"""
    base_price = 50000.0
    np.random.seed(42)

    # ランダムウォーク
    returns = np.random.normal(0.001, 0.02, length)
    prices = base_price * np.exp(returns.cumsum())

    return prices


def test_strategy_conditions_with_real_data():
    """実際のデータを使った戦略条件テスト"""
    logger.info("="*60)
    logger.info("実際の価格データを使った戦略条件テスト開始")
    logger.info("="*60)

    # 合成価格データを作成
    price_data = create_synthetic_price_data(200)
    mock_strategy = MockStrategy(price_data)

    # 条件評価器
    evaluator = ConditionEvaluator()

    # テスト戦略生成器
    generator = SmartConditionGenerator()

    # テスト指標組み合わせ
    test_cases = [
        {
            "name": "RSI + SMA",
            "indicators": [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ]
        },
        {
            "name": "RSI + EMA + MACD",
            "indicators": [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="EMA", parameters={"period": 50}, enabled=True),
                IndicatorGene(type="MACD", parameters={}, enabled=True),
            ]
        },
        {
            "name": "RSI + STCOH + BB",
            "indicators": [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="STOCH", parameters={}, enabled=True),
                IndicatorGene(type="BB", parameters={"period": 20}, enabled=True),
            ]
        }
    ]

    results = []

    for case in test_cases:
        logger.info(f"\n--- テストケース: {case['name']} ---")

        # 戦略生成
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(
            case['indicators']
        )

        logger.info(f"ロング条件数: {len(long_conds)}")
        logger.info(f"ショート条件数: {len(short_conds)}")

        # 実際の条件評価（適切なバーのみ）
        test_count = min(50, len(price_data) - 50)  # 指標計算を考慮
        long_signals = 0
        short_signals = 0
        both_signals = 0

        for i in range(50, 50 + test_count):
            bar_prices = price_data[:i+1]
            mock = MockStrategy(bar_prices)

            try:
                long_signal = bool(long_conds) and evaluator.evaluate_conditions(long_conds, mock)
                short_signal = bool(short_conds) and evaluator.evaluate_conditions(short_conds, mock)

                if long_signal:
                    long_signals += 1
                if short_signal:
                    short_signals += 1
                if long_signal and short_signal:
                    both_signals += 1

            except Exception as e:
                logger.warning(f"バー{i}評価エラー: {e}")
                continue

        # 結果分析
        long_ratio = long_signals / test_count if test_count > 0 else 0
        short_ratio = short_signals / test_count if test_count > 0 else 0

        logger.info("条件評価結果:")
        logger.info(f"  テストバー数: {test_count}")
        logger.info(f"  ロングシグナル数: {long_signals}")
        logger.info(f"  ショートシグナル数: {short_signals}")
        logger.info(f"  同時シグナル数: {both_signals}")
        logger.info(".3f")
        logger.info(".3f")

        # バランスチェック
        balance_ratio = short_ratio / long_ratio if long_ratio > 0 else float('inf')

        if balance_ratio == float('inf'):
            logger.warning("  ⚠️ ロングシグナルが0件")
            balance_status = "NO_LONG"
        elif balance_ratio < 0.5:
            logger.warning(f"  ⚠️ ショートシグナルが少なすぎる (バランス比: {balance_ratio:.2f})")
            balance_status = "SHORT_LOW"
        elif balance_ratio > 2.0:
            logger.warning(f"  ⚠️ ショートシグナルが多すぎる (バランス比: {balance_ratio:.2f})")
            balance_status = "SHORT_HIGH"
        else:
            logger.info(f"  ✅ バランス良好 (バランス比: {balance_ratio:.2f})")
            balance_status = "GOOD"

        # 結果保存
        results.append({
            'case': case['name'],
            'long_signals': long_signals,
            'short_signals': short_signals,
            'both_signals': both_signals,
            'test_count': test_count,
            'long_ratio': long_ratio,
            'short_ratio': short_ratio,
            'balance_ratio': balance_ratio,
            'balance_status': balance_status,
            'long_conditions': len(long_conds),
            'short_conditions': len(short_conds)
        })

    # 全体分析
    logger.info(f"\n{'='*60}")
    logger.info("全体テスト結果分析")
    logger.info('='*60)

    good_balance_count = sum(1 for r in results if r['balance_status'] == "GOOD")
    total_cases = len(results)

    logger.info(f"総テストケース数: {total_cases}")
    logger.info(f"バランス良好ケース数: {good_balance_count}")

    for result in results:
        status_emoji = {
            'GOOD': '✅',
            'SHORT_LOW': '⚠️',
            'SHORT_HIGH': '⚠️',
            'NO_LONG': '❌'
        }.get(result['balance_status'], '❓')

        logger.info(f"{status_emoji} {result['case']}: バランス比={result['balance_ratio']:.2f}")

    success_rate = good_balance_count / total_cases if total_cases > 0 else 0
    logger.info(".1f")

    if success_rate >= 0.8:
        logger.info("🎉 修正成功！戦略条件のバランスが良好です！")
    else:
        logger.warning("⚠️ 一部のケースでバランス調整が必要です")

    return results


def run_real_world_scenario_tests():
    """現実的な市場シナリオテスト"""
    logger.info(f"\n{'='*50}")
    logger.info("現実的な市場シナリオテスト")
    logger.info('='*50)

    # さまざまな市場状況でのテスト
    scenarios = [
        ("コアルレンジ相場", lambda x: 50000 + 2000 * np.sin(0.1 * np.arange(len(x)))),
        ("上昇トレンド相場", lambda x: 50000 * (1.0002 ** np.arange(len(x)))),
        ("下降トレンド相場", lambda x: 50000 * (0.9998 ** np.arange(len(x)))),
        ("高ボラティリティ相場", lambda x: 50000 * np.exp(np.random.normal(0, 0.005, len(x)).cumsum())),
    ]

    generator = SmartConditionGenerator()

    # 固定指標セット
    test_indicators = [
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="EMA", parameters={"period": 50}, enabled=True),
    ]

    evaluator = ConditionEvaluator()

    scenario_results = []

    for scenario_name, price_generator in scenarios:
        logger.info(f"\n--- シナリオ: {scenario_name} ---")

        # シナリオ別価格データ生成
        prices = price_generator(range(200))

        long_conds, short_conds, _ = generator.generate_balanced_conditions(test_indicators)

        # シグナル分析
        signals = 0
        long_count = 0
        short_count = 0

        test_start = 50  # 指標計算のためのウォームアップ期間
        for i in range(test_start, len(prices)):
            price_subset = prices[:i+1]
            mock = MockStrategy(price_subset)

            try:
                long_sig = bool(long_conds) and evaluator.evaluate_conditions(long_conds, mock)
                short_sig = bool(short_conds) and evaluator.evaluate_conditions(short_conds, mock)

                signals += 1
                if long_sig:
                    long_count += 1
                if short_sig:
                    short_count += 1

            except Exception as e:
                continue

        balance_ratio = short_count / long_count if long_count > 0 else float('inf')

        logger.info(f"  総シグナル数: {signals}")
        logger.info(f"  ロングシグナル: {long_count}")
        logger.info(f"  ショートシグナル: {short_count}")

        if balance_ratio != float('inf'):
            logger.info(".2f")

            if 0.5 <= balance_ratio <= 2.0:
                logger.info("  ✅ バランス良好")
                status = "GOOD"
            else:
                logger.warning("  ⚠️ バランス問題")
                status = "BAD"
        else:
            logger.warning("  ❌ ロングシグナル無し")
            status = "NO_LONG"

        scenario_results.append({
            'scenario': scenario_name,
            'balance_ratio': balance_ratio,
            'status': status
        })

    # シナリオ別成功率
    good_scenarios = sum(1 for r in scenario_results if r['status'] == "GOOD")
    total_scenarios = len(scenarios)
    success_rate = good_scenarios / total_scenarios if total_scenarios > 0 else 0
    logger.info(".1f")

if __name__ == "__main__":
    logger.info("簡易版実際のバックテストデータを使ったデバッグテストを開始")

    try:
        # 戦略条件テスト
        condition_results = test_strategy_conditions_with_real_data()

        # リアルダシナリオテスト
        scenario_results = run_real_world_scenario_tests()

        logger.info("\n" + "="*60)
        logger.info("デバッグテスト完了")
        logger.info("🐛 デバッグしながら修正効果を検証しました")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"メインエラー: {e}")
        traceback.print_exc()
        sys.exit(1)