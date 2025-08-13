#!/usr/bin/env python3
"""
ZLMA指標の詳細デバッグ - 指標登録ロジックの実行確認
"""

print("🚀 テストスクリプト実行開始！")
print("Python実行確認")

from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.gene_strategy import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.services.indicators.config import indicator_registry
import pandas as pd
import numpy as np
import backtesting
import logging

# ログレベルを詳細に設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("=== ZLMA指標の詳細デバッグ ===")

# テストデータ作成
np.random.seed(42)
n_periods = 100
df = pd.DataFrame(
    {
        "Open": np.linspace(50, 60, n_periods),
        "High": np.linspace(51, 61, n_periods),
        "Low": np.linspace(49, 59, n_periods),
        "Close": np.linspace(50, 60, n_periods),
        "Volume": np.full(n_periods, 1000),
    }
)

print(f"テストデータ作成完了: {len(df)}期間")

# ZLMA指標の設定確認
config = indicator_registry.get_indicator_config("ZLMA")
if config:
    print(
        f"ZLMA設定: カテゴリ={config.category}, 結果タイプ={config.result_type.value}"
    )
    print(f"必要データ: {config.required_data}")
    print(f"パラメータ: {config.parameters}")
else:
    print("ZLMA設定なし")
    exit(1)


# デフォルトパラメータ取得
def get_default_parameters(indicator_name):
    config = indicator_registry.get_indicator_config(indicator_name)
    if not config or not config.parameters:
        return {}

    params = {}
    for param_name, param_config in config.parameters.items():
        params[param_name] = param_config.default_value
    return params


default_params = get_default_parameters("ZLMA")
print(f"デフォルトパラメータ: {default_params}")

# 指標遺伝子作成
indicator_gene = IndicatorGene(type="ZLMA", parameters=default_params, enabled=True)

# 条件作成
condition = Condition(left_operand="ZLMA", operator=">", right_operand="close")

# ポジションサイジング遺伝子
position_sizing_gene = PositionSizingGene(
    method=PositionSizingMethod.FIXED_RATIO, fixed_ratio=0.1
)

# 戦略遺伝子作成
strategy_gene = StrategyGene(
    indicators=[indicator_gene],
    long_entry_conditions=[condition],
    short_entry_conditions=[],
    exit_conditions=[condition],
    risk_management={"stop_loss": 0.02, "take_profit": 0.04},
    position_sizing_gene=position_sizing_gene,
)

print(f"戦略遺伝子作成成功")

# 戦略クラス作成
print(f"\n=== 戦略クラス作成開始 ===")
factory = StrategyFactory()
print(f"StrategyFactory作成成功: {factory}")

print(f"create_strategy_class呼び出し開始...")
strategy_class = factory.create_strategy_class(strategy_gene)
print(f"create_strategy_class呼び出し完了")

print(f"戦略クラス作成成功: {strategy_class}")
print(f"戦略クラス型: {type(strategy_class)}")
print(f"戦略クラス名: {strategy_class.__name__}")
print(f"戦略クラスMRO: {strategy_class.__mro__}")

# バックテスト実行前の確認
print(f"\n=== バックテスト実行前の確認 ===")
print(f"戦略クラス: {strategy_class}")
print(f"戦略クラスのMRO: {strategy_class.__mro__}")

# バックテスト実行
print(f"\n=== バックテスト実行 ===")
bt = backtesting.Backtest(df, strategy_class, cash=1000000)

# 詳細ログを有効にしてバックテスト実行
print("バックテスト実行開始...")
result = bt.run(strategy_gene=strategy_gene)
print("バックテスト実行完了")

# 戦略インスタンス詳細確認
strategy_instance = bt._strategy
print(f"\n=== 戦略インスタンス詳細確認 ===")
print(f"戦略インスタンス型: {type(strategy_instance)}")
print(f"戦略インスタンスID: {id(strategy_instance)}")

# 全属性確認
all_attrs = dir(strategy_instance)
public_attrs = [attr for attr in all_attrs if not attr.startswith("_")]
print(f"パブリック属性: {public_attrs}")

# ZLMA確認
if hasattr(strategy_instance, "ZLMA"):
    print(f"✅ ZLMA登録成功")
    zlma_value = strategy_instance.ZLMA
    print(f"ZLMA型: {type(zlma_value)}")
    if hasattr(zlma_value, "__len__"):
        print(f"ZLMA長さ: {len(zlma_value)}")
else:
    print(f"❌ ZLMA登録失敗")

# indicators辞書確認
if hasattr(strategy_instance, "indicators"):
    print(f"indicators辞書存在: True")
    print(f"indicators辞書キー: {list(strategy_instance.indicators.keys())}")
    if "ZLMA" in strategy_instance.indicators:
        print(f"✅ indicators辞書にZLMA存在")
    else:
        print(f"❌ indicators辞書にZLMA不存在")
else:
    print(f"indicators辞書存在: False")

# __dict__確認
if "ZLMA" in strategy_instance.__dict__:
    print(f"✅ __dict__にZLMA存在")
else:
    print(f"❌ __dict__にZLMA不存在")

print(f"__dict__キー: {list(strategy_instance.__dict__.keys())}")

# 戦略遺伝子確認
if hasattr(strategy_instance, "strategy_gene"):
    print(f"戦略遺伝子存在: True")
    print(
        f"指標遺伝子: {[ind.type for ind in strategy_instance.strategy_gene.indicators]}"
    )
else:
    print(f"戦略遺伝子存在: False")

print(f"\n=== デバッグ完了 ===")
