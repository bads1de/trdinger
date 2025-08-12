#!/usr/bin/env python3
"""
オートストラテジーのテクニカルオンリーモードテスト

リファクタリング後の動作確認として、実際に戦略を生成してテストします。
"""

import json
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from app.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.models.gene_serialization import GeneSerializer
from app.services.auto_strategy.orchestration.auto_strategy_orchestration_service import (
    AutoStrategyOrchestrationService,
)
from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.backtest_service import BacktestService
from database.connection import SessionLocal

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_backtest_service():
    """テスト用のバックテストサービスを作成"""
    try:
        with SessionLocal() as db:
            from database.repositories.funding_rate_repository import (
                FundingRateRepository,
            )
            from database.repositories.ohlcv_repository import OHLCVRepository
            from database.repositories.open_interest_repository import (
                OpenInterestRepository,
            )

            ohlcv_repo = OHLCVRepository(db)
            oi_repo = OpenInterestRepository(db)
            fr_repo = FundingRateRepository(db)

            data_service = BacktestDataService(
                ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
            )
            return BacktestService(data_service)
    except Exception as e:
        logger.error(f"バックテストサービス作成エラー: {e}")
        return None


def test_technical_only_strategy_generation():
    """テクニカルオンリーモードでの戦略生成テスト"""
    logger.info("=== テクニカルオンリー戦略生成テスト開始 ===")

    # 再現性のためのシード設定
    random.seed(42)
    np.random.seed(42)

    # テクニカルオンリーモードのGA設定
    ga_config = GAConfig(
        population_size=5,  # 小さめで高速テスト
        generations=3,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=1,
        max_indicators=3,
        min_indicators=2,
        max_conditions=4,
        min_conditions=2,
        indicator_mode="technical_only",  # テクニカルオンリー
        allowed_indicators=[
            "SMA",
            "EMA",
            "RSI",
            "MACD",
            "BB",
            "ATR",
            "CCI",
            "STOCH",
        ],
        enable_multi_objective=False,
        objectives=["total_return"],
        log_level="INFO",
    )

    logger.info(f"GA設定: {ga_config.indicator_mode}, 指標数: {ga_config.max_indicators}")

    # 遺伝子生成器を作成
    gene_generator = RandomGeneGenerator(ga_config, enable_smart_generation=True)

    # 複数の戦略を生成してテスト
    strategies = []
    for i in range(5):
        logger.info(f"戦略 {i+1} を生成中...")
        gene = gene_generator.generate_random_gene()

        # 戦略の詳細を確認
        logger.info(f"  指標数: {len(gene.indicators)}")
        logger.info(f"  指標: {[ind.type for ind in gene.indicators]}")
        logger.info(f"  ロング条件数: {len(gene.long_entry_conditions)}")
        logger.info(f"  ショート条件数: {len(gene.short_entry_conditions)}")
        logger.info(f"  TP/SL有効: {gene.tpsl_gene.enabled if gene.tpsl_gene else False}")

        # ML指標が含まれていないことを確認
        ml_indicators = [ind for ind in gene.indicators if ind.type.startswith("ML_")]
        assert len(ml_indicators) == 0, f"ML指標が含まれています: {ml_indicators}"

        strategies.append(gene)

    logger.info(f"✅ {len(strategies)}個の戦略を正常に生成しました")
    return strategies


def test_strategy_execution():
    """生成された戦略の実行テスト"""
    logger.info("=== 戦略実行テスト開始 ===")

    # 戦略を生成
    strategies = test_technical_only_strategy_generation()
    if not strategies:
        logger.error("戦略が生成されませんでした")
        return None

    # 最初の戦略をテスト
    test_strategy = strategies[0]
    logger.info("テスト戦略の詳細:")
    logger.info(f"  ID: {test_strategy.id}")
    logger.info(f"  指標: {[f'{ind.type}({ind.parameters})' for ind in test_strategy.indicators]}")

    # 戦略ファクトリーで戦略クラスを作成
    factory = StrategyFactory()
    try:
        strategy_class = factory.create_strategy_class(test_strategy)
        logger.info("✅ 戦略クラスの作成に成功しました")

        # 戦略の妥当性検証
        is_valid, errors = factory.validate_gene(test_strategy)
        if is_valid:
            logger.info("✅ 戦略の妥当性検証に成功しました")
        else:
            logger.warning(f"⚠️ 戦略の妥当性検証で警告: {errors}")

        return test_strategy

    except Exception as e:
        logger.error(f"❌ 戦略クラス作成エラー: {e}")
        return None


def test_strategy_serialization():
    """戦略のシリアライゼーションテスト"""
    logger.info("=== 戦略シリアライゼーションテスト開始 ===")

    # 戦略を生成
    strategies = test_technical_only_strategy_generation()
    if not strategies:
        return None

    test_strategy = strategies[0]
    serializer = GeneSerializer()

    try:
        # 辞書形式にシリアライズ
        strategy_dict = serializer.strategy_gene_to_dict(test_strategy)
        logger.info("✅ 戦略の辞書シリアライゼーションに成功しました")

        # 辞書から復元
        restored_strategy = serializer.dict_to_strategy_gene(strategy_dict, type(test_strategy))
        logger.info("✅ 戦略の辞書デシリアライゼーションに成功しました")

        # リスト形式にエンコード
        encoded_list = serializer.to_list(test_strategy)
        logger.info(f"✅ 戦略のリストエンコードに成功しました (長さ: {len(encoded_list)})")

        # リストから復元
        decoded_strategy = serializer.from_list(encoded_list, type(test_strategy))
        logger.info("✅ 戦略のリストデコードに成功しました")

        return strategy_dict

    except Exception as e:
        logger.error(f"❌ シリアライゼーションエラー: {e}")
        return None


def display_strategy_details(strategy_dict):
    """戦略の詳細を表示"""
    logger.info("=== 生成された戦略の詳細 ===")

    print("\n" + "="*60)
    print("🎯 生成された戦略の詳細")
    print("="*60)

    # 基本情報
    print(f"戦略ID: {strategy_dict.get('id', 'N/A')}")
    print(f"生成者: {strategy_dict.get('metadata', {}).get('generated_by', 'N/A')}")

    # 指標情報
    indicators = strategy_dict.get('indicators', [])
    print(f"\n📊 使用指標 ({len(indicators)}個):")
    for i, ind in enumerate(indicators, 1):
        params_str = ", ".join([f"{k}={v}" for k, v in ind.get('parameters', {}).items()])
        print(f"  {i}. {ind.get('type', 'N/A')} ({params_str})")

    # エントリー条件
    long_conditions = strategy_dict.get('long_entry_conditions', [])
    short_conditions = strategy_dict.get('short_entry_conditions', [])

    print(f"\n📈 ロングエントリー条件 ({len(long_conditions)}個):")
    for i, cond in enumerate(long_conditions, 1):
        print(f"  {i}. {cond.get('left_operand', 'N/A')} {cond.get('operator', 'N/A')} {cond.get('right_operand', 'N/A')}")

    print(f"\n📉 ショートエントリー条件 ({len(short_conditions)}個):")
    for i, cond in enumerate(short_conditions, 1):
        print(f"  {i}. {cond.get('left_operand', 'N/A')} {cond.get('operator', 'N/A')} {cond.get('right_operand', 'N/A')}")

    # TP/SL設定
    tpsl_gene = strategy_dict.get('tpsl_gene', {})
    if tpsl_gene and tpsl_gene.get('enabled'):
        print(f"\n🎯 TP/SL設定:")
        print(f"  方式: {tpsl_gene.get('method', 'N/A')}")
        print(f"  ストップロス: {tpsl_gene.get('stop_loss_pct', 0)*100:.2f}%")
        print(f"  テイクプロフィット: {tpsl_gene.get('take_profit_pct', 0)*100:.2f}%")
        print(f"  リスクリワード比: {tpsl_gene.get('risk_reward_ratio', 'N/A')}")

    # ポジションサイジング
    ps_gene = strategy_dict.get('position_sizing_gene', {})
    if ps_gene and ps_gene.get('enabled'):
        print(f"\n💰 ポジションサイジング:")
        print(f"  方式: {ps_gene.get('method', 'N/A')}")
        print(f"  リスク/取引: {ps_gene.get('risk_per_trade', 0)*100:.2f}%")

    print("="*60)


def main():
    """メイン実行関数"""
    logger.info("🚀 オートストラテジー テクニカルオンリーモード テスト開始")

    try:
        # 1. 戦略生成テスト
        strategies = test_technical_only_strategy_generation()
        if not strategies:
            logger.error("❌ 戦略生成に失敗しました")
            return

        # 2. 戦略実行テスト
        test_strategy = test_strategy_execution()
        if not test_strategy:
            logger.error("❌ 戦略実行テストに失敗しました")
            return

        # 3. シリアライゼーションテスト
        strategy_dict = test_strategy_serialization()
        if not strategy_dict:
            logger.error("❌ シリアライゼーションテストに失敗しました")
            return

        # 4. 戦略詳細表示
        display_strategy_details(strategy_dict)

        logger.info("✅ すべてのテストが正常に完了しました！")

    except Exception as e:
        logger.error(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
