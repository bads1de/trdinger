#!/usr/bin/env python3
"""
指標計算テスト
"""

from app.services.auto_strategy.calculators.indicator_calculator import (
    IndicatorCalculator,
)
from app.services.auto_strategy.models.gene_strategy import IndicatorGene
import pandas as pd
import numpy as np

# 簡単なテストデータ作成
data = type("Data", (), {})()
df = pd.DataFrame(
    {
        "Open": np.linspace(50, 60, 100),
        "High": np.linspace(51, 61, 100),
        "Low": np.linspace(49, 59, 100),
        "Close": np.linspace(50, 60, 100),
        "Volume": np.full(100, 1000),
    }
)
data.df = df


# 偽の戦略インスタンス作成
class FakeStrategy:
    def __init__(self, data):
        self.data = data

    def I(self, func):
        return func()


strategy = FakeStrategy(data)

# IndicatorCalculatorテスト
calc = IndicatorCalculator()

# SMAテスト
print("=== SMA テスト ===")
sma_gene = IndicatorGene(type="SMA", parameters={"period": 14}, enabled=True)
try:
    calc.init_indicator(sma_gene, strategy)
    print(f'SMA登録後の属性確認: hasattr(strategy, "SMA") = {hasattr(strategy, "SMA")}')
    if hasattr(strategy, "SMA"):
        print(f"SMA値: {strategy.SMA}")
except Exception as e:
    print(f"SMA初期化失敗: {e}")

# EMAテスト
print("\n=== EMA テスト ===")
ema_gene = IndicatorGene(type="EMA", parameters={"period": 14}, enabled=True)
try:
    calc.init_indicator(ema_gene, strategy)
    print(f'EMA登録後の属性確認: hasattr(strategy, "EMA") = {hasattr(strategy, "EMA")}')
    if hasattr(strategy, "EMA"):
        print(f"EMA値: {strategy.EMA}")
except Exception as e:
    print(f"EMA初期化失敗: {e}")

# WMAテスト
print("\n=== WMA テスト ===")
wma_gene = IndicatorGene(type="WMA", parameters={"period": 14}, enabled=True)
try:
    calc.init_indicator(wma_gene, strategy)
    print(f'WMA登録後の属性確認: hasattr(strategy, "WMA") = {hasattr(strategy, "WMA")}')
    if hasattr(strategy, "WMA"):
        print(f"WMA値: {strategy.WMA}")
except Exception as e:
    print(f"WMA初期化失敗: {e}")

# 全属性確認
print(f"\n=== 戦略インスタンスの全属性 ===")
attrs = [attr for attr in dir(strategy) if not attr.startswith("_")]
print(f"利用可能属性: {attrs}")
