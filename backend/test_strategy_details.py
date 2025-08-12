#!/usr/bin/env python3
"""
戦略詳細表示テスト

生成された戦略の詳細を確認
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.models.ga_config import GAConfig

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_condition(cond, prefix=""):
    """条件を分析して表示"""
    if hasattr(cond, 'left_operand'):
        # 単一条件
        return f"{prefix}{cond.left_operand} {cond.operator} {cond.right_operand}"
    elif hasattr(cond, 'conditions'):
        # ConditionGroup
        sub_conditions = []
        for sub_cond in cond.conditions:
            sub_conditions.append(analyze_condition(sub_cond, ""))
        return f"{prefix}({' OR '.join(sub_conditions)})"
    else:
        return f"{prefix}{str(cond)}"


def display_strategy_details(gene):
    """戦略の詳細を表示"""
    print("\n" + "="*80)
    print("🎯 生成された戦略の詳細")
    print("="*80)

    # 基本情報
    print(f"戦略ID: {gene.id}")
    print(f"指標数: {len(gene.indicators)}")

    # 指標詳細
    print(f"\n📊 使用指標:")
    for i, ind in enumerate(gene.indicators, 1):
        params_str = ", ".join([f"{k}={v}" for k, v in ind.parameters.items()])
        print(f"  {i}. {ind.type} ({params_str})")

    # ロングエントリー条件
    print(f"\n📈 ロングエントリー条件 ({len(gene.long_entry_conditions)}個):")
    for i, cond in enumerate(gene.long_entry_conditions, 1):
        condition_str = analyze_condition(cond)
        print(f"  {i}. {condition_str}")

    # ショートエントリー条件
    print(f"\n📉 ショートエントリー条件 ({len(gene.short_entry_conditions)}個):")
    for i, cond in enumerate(gene.short_entry_conditions, 1):
        condition_str = analyze_condition(cond)
        print(f"  {i}. {condition_str}")

    # TP/SL設定
    if gene.tpsl_gene and gene.tpsl_gene.enabled:
        print(f"\n🎯 TP/SL設定:")
        print(f"  方式: {gene.tpsl_gene.method}")
        print(f"  ストップロス: {gene.tpsl_gene.stop_loss_pct*100:.2f}%")
        print(f"  テイクプロフィット: {gene.tpsl_gene.take_profit_pct*100:.2f}%")
        if hasattr(gene.tpsl_gene, 'risk_reward_ratio'):
            print(f"  リスクリワード比: {gene.tpsl_gene.risk_reward_ratio}")

    # ポジションサイジング
    if gene.position_sizing_gene and gene.position_sizing_gene.enabled:
        print(f"\n💰 ポジションサイジング:")
        print(f"  方式: {gene.position_sizing_gene.method}")
        print(f"  リスク/取引: {gene.position_sizing_gene.risk_per_trade*100:.2f}%")

    print("="*80)


def generate_multiple_strategies():
    """複数の戦略を生成して表示"""
    logger.info("=== 複数戦略生成テスト ===")

    # 再現性のためのシード設定
    random.seed(456)
    np.random.seed(456)

    # GA設定
    ga_config = GAConfig(
        population_size=3,
        generations=2,
        max_indicators=3,
        min_indicators=2,
        max_conditions=3,
        min_conditions=1,
        indicator_mode="technical_only",
        allowed_indicators=["SMA", "EMA", "RSI", "MACD", "BB", "ATR"],
        log_level="INFO",
    )

    # 遺伝子生成器を作成
    gene_generator = RandomGeneGenerator(ga_config, enable_smart_generation=True)

    strategies = []
    for i in range(3):
        logger.info(f"戦略 {i+1} を生成中...")
        try:
            gene = gene_generator.generate_random_gene()
            strategies.append(gene)
            logger.info(f"✅ 戦略 {i+1} 生成成功")
        except Exception as e:
            logger.error(f"❌ 戦略 {i+1} 生成失敗: {e}")

    return strategies


def analyze_strategy_types(strategies):
    """戦略タイプの分析"""
    print("\n" + "="*80)
    print("📊 戦略タイプ分析")
    print("="*80)

    for i, strategy in enumerate(strategies, 1):
        print(f"\n--- 戦略 {i} ---")
        
        # 指標の組み合わせ
        indicators = [ind.type for ind in strategy.indicators]
        print(f"指標組み合わせ: {' + '.join(indicators)}")
        
        # 戦略の特徴を分析
        has_trend = any(ind in ['SMA', 'EMA', 'WMA'] for ind in indicators)
        has_momentum = any(ind in ['RSI', 'MACD', 'CCI'] for ind in indicators)
        has_volatility = any(ind in ['BB', 'ATR'] for ind in indicators)
        
        strategy_type = []
        if has_trend:
            strategy_type.append("トレンド系")
        if has_momentum:
            strategy_type.append("モメンタム系")
        if has_volatility:
            strategy_type.append("ボラティリティ系")
        
        print(f"戦略タイプ: {' + '.join(strategy_type) if strategy_type else '基本系'}")
        
        # 条件の複雑さ
        long_complexity = len(strategy.long_entry_conditions)
        short_complexity = len(strategy.short_entry_conditions)
        print(f"条件複雑度: ロング{long_complexity}条件, ショート{short_complexity}条件")

    print("="*80)


def main():
    """メイン実行関数"""
    logger.info("🚀 戦略詳細分析テスト開始")

    try:
        # 戦略生成
        strategies = generate_multiple_strategies()
        if not strategies:
            logger.error("❌ 戦略生成に失敗しました")
            return

        logger.info(f"✅ {len(strategies)}個の戦略を生成しました")

        # 各戦略の詳細表示
        for i, strategy in enumerate(strategies):
            print(f"\n{'='*20} 戦略 {i+1} {'='*20}")
            display_strategy_details(strategy)

        # 戦略タイプ分析
        analyze_strategy_types(strategies)

        # 戦略の例を表示
        if strategies:
            example_strategy = strategies[0]
            print(f"\n🎉 戦略生成成功！")
            print(f"例: {[ind.type for ind in example_strategy.indicators]} を使用した戦略")
            
            # 簡単な戦略説明
            print(f"\n📝 戦略の概要:")
            indicators = [ind.type for ind in example_strategy.indicators]
            if 'SMA' in indicators or 'EMA' in indicators:
                print("- トレンドフォロー要素を含む")
            if 'RSI' in indicators:
                print("- RSIによるオーバーボート/オーバーソールド判定")
            if 'MACD' in indicators:
                print("- MACDによるモメンタム分析")
            if 'BB' in indicators:
                print("- ボリンジャーバンドによるボラティリティ分析")

        logger.info("✅ すべての分析が完了しました！")

    except Exception as e:
        logger.error(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
