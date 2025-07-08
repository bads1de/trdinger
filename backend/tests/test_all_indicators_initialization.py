#!/usr/bin/env python3
"""
全インジケーター初期化テストスクリプト

オートストラテジー機能で全てのテクニカルインジケーターの初期化をテストし、
問題がないか包括的に確認します。
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.calculators.indicator_calculator import (
    IndicatorCalculator,
)
from app.core.services.indicators import TechnicalIndicatorService
import pandas as pd
import numpy as np

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data():
    """テスト用のOHLCVデータを作成"""
    # 100日分のテストデータを作成
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)  # 再現性のため
    
    # 価格データを生成（ランダムウォーク）
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 100)  # 2%の標準偏差
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # 価格が負にならないように
    
    # OHLCV データを作成
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close + np.random.normal(0, close * 0.005)
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'Open': open_price,
            'High': max(open_price, high, close),
            'Low': min(open_price, low, close),
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_indicator_direct(indicator_type, params):
    """インジケーターを直接計算してテスト"""
    try:
        df = create_test_data()
        
        # TechnicalIndicatorServiceを使用して計算
        service = TechnicalIndicatorService()
        
        logger.info(f"{indicator_type}計算開始: パラメータ={params}")
        result = service.calculate_indicator(df, indicator_type, params)
        
        logger.info(f"{indicator_type}計算結果: タイプ={type(result)}")
        if isinstance(result, tuple):
            logger.info(f"{indicator_type}出力数: {len(result)}")
            for i, output in enumerate(result):
                logger.info(f"出力{i}: 形状={output.shape}, 非NaN数={np.sum(~np.isnan(output))}")
        else:
            logger.info(f"{indicator_type}: 形状={result.shape}, 非NaN数={np.sum(~np.isnan(result))}")
        
        return True
        
    except Exception as e:
        logger.error(f"{indicator_type}直接計算エラー: {e}")
        return False

def test_indicator_in_strategy(indicator_type, params):
    """戦略初期化でのインジケーターテスト"""
    try:
        df = create_test_data()
        
        # インジケーター遺伝子を作成
        indicator_gene = IndicatorGene(
            type=indicator_type,
            parameters=params,
            enabled=True
        )
        
        # ダミーの条件を作成
        dummy_entry_condition = Condition(
            left_operand="close",
            operator=">",
            right_operand="open"
        )
        dummy_exit_condition = Condition(
            left_operand="close",
            operator="<",
            right_operand="open"
        )
        
        # 戦略遺伝子を作成
        strategy_gene = StrategyGene(
            indicators=[indicator_gene],
            entry_conditions=[dummy_entry_condition],
            long_entry_conditions=[],
            short_entry_conditions=[],
            exit_conditions=[dummy_exit_condition],
            risk_management={},
            tpsl_gene=None,
        )
        
        # StrategyFactoryで戦略を作成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)
        
        # backtesting.pyのデータオブジェクトをシミュレート
        class MockData:
            def __init__(self, df):
                self.df = df
                # backtesting.pyが期待する属性を追加
                self.Close = df["Close"]
                self.Open = df["Open"]
                self.High = df["High"]
                self.Low = df["Low"]
                self.Volume = df["Volume"]
                self.index = df.index
        
        mock_data = MockData(df)
        
        # 戦略インスタンスを作成して初期化
        strategy_instance = strategy_class(data=mock_data)
        strategy_instance.init()
        
        # インジケーター関連属性が正しく設定されているか確認
        indicator_attrs = [attr for attr in dir(strategy_instance) if indicator_type in attr]
        logger.info(f"{indicator_type}関連属性: {indicator_attrs}")
        
        return True
        
    except Exception as e:
        logger.error(f"{indicator_type}戦略初期化エラー: {e}")
        return False

def main():
    """メイン関数"""
    logger.info("全インジケーター初期化テスト開始")
    
    # テスト対象のインジケーターとパラメータ
    test_indicators = [
        ("RSI", {"period": 14}),
        ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
        ("SMA", {"period": 20}),
        ("EMA", {"period": 20}),
        ("ATR", {"period": 14}),
        ("BB", {"period": 20, "std_dev": 2.0}),
        ("STOCH", {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3}),
    ]
    
    results = {}
    
    for indicator_type, params in test_indicators:
        logger.info(f"\n=== {indicator_type} テスト ===")
        
        # 直接計算テスト
        direct_success = test_indicator_direct(indicator_type, params)
        
        # 戦略初期化テスト
        strategy_success = test_indicator_in_strategy(indicator_type, params)
        
        results[indicator_type] = {
            'direct': direct_success,
            'strategy': strategy_success,
            'overall': direct_success and strategy_success
        }
    
    # 結果サマリー
    logger.info("\n=== テスト結果サマリー ===")
    all_success = True
    for indicator_type, result in results.items():
        overall_status = "成功" if result['overall'] else "失敗"
        direct_status = "成功" if result['direct'] else "失敗"
        strategy_status = "成功" if result['strategy'] else "失敗"
        
        logger.info(f"{indicator_type}: {overall_status} (直接: {direct_status}, 戦略: {strategy_status})")
        
        if not result['overall']:
            all_success = False
    
    if all_success:
        logger.info("全てのインジケーターテストが成功しました")
    else:
        logger.error("一部のインジケーターテストが失敗しました")
        
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
