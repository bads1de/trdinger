#!/usr/bin/env python3
"""
インポートテスト
"""

print("🚀 インポートテスト開始！")

try:
    print("StrategyFactoryインポート開始...")
    from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
    print("✅ StrategyFactoryインポート成功")
except Exception as e:
    print(f"❌ StrategyFactoryインポートエラー: {e}")
    exit(1)

try:
    print("gene_strategyインポート開始...")
    from app.services.auto_strategy.models.gene_strategy import (
        StrategyGene, IndicatorGene, Condition
    )
    print("✅ gene_strategyインポート成功")
except Exception as e:
    print(f"❌ gene_strategyインポートエラー: {e}")
    exit(1)

try:
    print("gene_position_sizingインポート開始...")
    from app.services.auto_strategy.models.gene_position_sizing import (
        PositionSizingGene, PositionSizingMethod
    )
    print("✅ gene_position_sizingインポート成功")
except Exception as e:
    print(f"❌ gene_position_sizingインポートエラー: {e}")
    exit(1)

try:
    print("indicator_registryインポート開始...")
    from app.services.indicators.config import indicator_registry
    print("✅ indicator_registryインポート成功")
except Exception as e:
    print(f"❌ indicator_registryインポートエラー: {e}")
    exit(1)

try:
    print("その他のインポート開始...")
    import pandas as pd
    import numpy as np
    import backtesting
    import logging
    print("✅ その他のインポート成功")
except Exception as e:
    print(f"❌ その他のインポートエラー: {e}")
    exit(1)

print("🎉 全てのインポート成功！")
