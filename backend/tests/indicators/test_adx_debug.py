#!/usr/bin/env python3
"""
ADX threshold デバッグテスト
"""

import logging
import sys

from app.services.auto_strategy.generators.condition_generator import ConditionGenerator

# ログレベル設定
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def test_adx():
    print("=== ADX threshold取得テスト ===")

    generator = ConditionGenerator()

    # ADX設定取得
    config = generator._get_indicator_config_from_yaml('ADX')
    print(f"ADX config: {config}")

    if config:
        # long条件のthreshold取得
        threshold = generator._get_threshold_from_yaml(config, 'long')
        print(f"ADX long threshold: {threshold}")

        # short条件のthreshold取得
        threshold_short = generator._get_threshold_from_yaml(config, 'short')
        print(f"ADX short threshold: {threshold_short}")

    print("=== 完了 ===")

if __name__ == "__main__":
    test_adx()