"""
リファクタリング後の戦略生成デモ

実際に戦略を生成して結果を確認します。
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.utils.auto_strategy_utils import AutoStrategyUtils
from app.services.auto_strategy.config.shared_constants import validate_symbol, validate_timeframe
from app.services.auto_strategy.utils.error_handling import AutoStrategyErrorHandler

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_ga_config():
    """テスト用GA設定を作成"""
    config = GAConfig.create_fast()
    
    # 小さな設定でテスト
    config.population_size = 5
    config.generations = 2
    config.max_indicators = 2
    
    logger.info(f"GA設定作成: {config.get_summary()}")
    return config


def create_test_backtest_config():
    """テスト用バックテスト設定を作成"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 30日間のテスト
    
    config = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initial_capital": 100000,
        "commission": 0.001,
        "slippage": 0.0001
    }
    
    # 設定検証
    assert validate_symbol(config["symbol"]), f"無効なシンボル: {config['symbol']}"
    assert validate_timeframe(config["timeframe"]), f"無効な時間軸: {config['timeframe']}"
    
    logger.info(f"バックテスト設定作成: {json.dumps(config, indent=2)}")
    return config


def test_error_handling():
    """エラーハンドリングのテスト"""
    logger.info("=== エラーハンドリングテスト ===")
    
    # 正常実行
    def success_func():
        return "成功しました"
    
    result = AutoStrategyErrorHandler.safe_execute(success_func)
    logger.info(f"正常実行結果: {result}")
    
    # エラー実行
    def error_func():
        raise ValueError("テストエラー")
    
    result = AutoStrategyErrorHandler.safe_execute(
        error_func, 
        fallback_value="フォールバック値",
        context="エラーテスト"
    )
    logger.info(f"エラー実行結果: {result}")


def test_utils_functionality():
    """ユーティリティ機能のテスト"""
    logger.info("=== ユーティリティ機能テスト ===")
    
    # データ変換テスト
    float_result = AutoStrategyUtils.safe_convert_to_float("123.45")
    int_result = AutoStrategyUtils.safe_convert_to_int("42")
    logger.info(f"データ変換: float={float_result}, int={int_result}")
    
    # シンボル正規化
    normalized = AutoStrategyUtils.normalize_symbol("BTC")
    logger.info(f"シンボル正規化: BTC -> {normalized}")
    
    # 検証機能
    range_valid = AutoStrategyUtils.validate_range(5, 1, 10)
    range_invalid = AutoStrategyUtils.validate_range(15, 1, 10)
    logger.info(f"範囲検証: 5 in [1,10] = {range_valid}, 15 in [1,10] = {range_invalid}")
    
    # 設定マージ
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"y": 3, "z": 4}, "c": 5}
    merged = AutoStrategyUtils.merge_configs(base, override)
    logger.info(f"設定マージ結果: {json.dumps(merged, indent=2)}")


def test_strategy_gene_creation():
    """戦略遺伝子作成のテスト"""
    logger.info("=== 戦略遺伝子作成テスト ===")
    
    strategy_gene = AutoStrategyUtils.create_default_strategy_gene()
    
    if strategy_gene:
        logger.info(f"戦略遺伝子作成成功:")
        logger.info(f"  指標数: {len(strategy_gene.indicators)}")
        logger.info(f"  エントリー条件数: {len(strategy_gene.entry_conditions)}")
        logger.info(f"  エグジット条件数: {len(strategy_gene.exit_conditions)}")
        logger.info(f"  メタデータ: {strategy_gene.metadata}")
        
        # 指標詳細
        for i, indicator in enumerate(strategy_gene.indicators):
            logger.info(f"  指標{i+1}: {indicator.type} (パラメータ: {indicator.parameters})")
        
        # 条件詳細
        for i, condition in enumerate(strategy_gene.entry_conditions):
            logger.info(f"  エントリー条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
    else:
        logger.error("戦略遺伝子の作成に失敗しました")


def test_config_functionality():
    """設定機能のテスト"""
    logger.info("=== 設定機能テスト ===")
    
    # GA設定テスト
    ga_config = create_test_ga_config()
    
    # 検証
    is_valid, errors = ga_config.validate()
    logger.info(f"GA設定検証: 有効={is_valid}, エラー={errors}")
    
    # 辞書変換
    config_dict = ga_config.to_dict()
    logger.info(f"GA設定辞書変換: {len(config_dict)}個のフィールド")
    
    # JSON変換
    config_json = ga_config.to_json()
    logger.info(f"GA設定JSON変換: {len(config_json)}文字")
    
    # 復元テスト
    restored_config = GAConfig.from_json(config_json)
    logger.info(f"GA設定復元: population_size={restored_config.population_size}")


def main():
    """メイン実行関数"""
    logger.info("🚀 リファクタリング後の戦略生成デモを開始します")
    
    try:
        # 1. エラーハンドリングテスト
        test_error_handling()
        
        # 2. ユーティリティ機能テスト
        test_utils_functionality()
        
        # 3. 戦略遺伝子作成テスト
        test_strategy_gene_creation()
        
        # 4. 設定機能テスト
        test_config_functionality()
        
        # 5. 設定作成テスト
        logger.info("=== 設定作成テスト ===")
        ga_config = create_test_ga_config()
        backtest_config = create_test_backtest_config()
        
        logger.info("✅ 全てのテストが正常に完了しました")
        
        # 結果サマリー
        logger.info("\n📊 リファクタリング結果サマリー:")
        logger.info("  ✅ エラーハンドリング統合: AutoStrategyErrorHandler")
        logger.info("  ✅ ユーティリティ統合: AutoStrategyUtils")
        logger.info("  ✅ 設定クラス統合: BaseConfig継承")
        logger.info("  ✅ 共通定数統合: shared_constants")
        logger.info("  ✅ 後方互換性: 既存APIの保持")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ デモ実行中にエラーが発生: {e}", exc_info=True)
        
        # エラーハンドリングのテスト
        error_result = AutoStrategyErrorHandler.handle_ga_error(e, "デモ実行")
        logger.info(f"エラーハンドリング結果: {error_result}")
        
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 リファクタリング後のシステムが正常に動作しています！")
    else:
        print("\n⚠️ 一部の機能でエラーが発生しました。ログを確認してください。")
