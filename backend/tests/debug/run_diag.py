#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','app'))

# minimal imports
from pathlib import Path

# Load condition_generator dynamically to avoid circular imports
import importlib.util

spec = importlib.util.spec_from_file_location(
    "condition_generator",
    os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'services', 'auto_strategy', 'generators', 'condition_generator.py')
)

if spec and spec.loader:
    condition_generator_module = importlib.util.module_from_spec(spec)

    # Pre-load minimal dependencies
    try:
        # Minimal constants import
        from app.services.auto_strategy.config.constants import IndicatorType, StrategyType
        sys.modules['app.services.auto_strategy.config.constants'] = importlib.import_module('app.services.auto_strategy.config.constants')

        # Minimal models import
        from app.services.auto_strategy.models.strategy_models import Condition, IndicatorGene
        sys.modules['app.services.auto_strategy.models.strategy_models'] = importlib.import_module('app.services.auto_strategy.models.strategy_models')

        # Load the module
        spec.loader.exec_module(condition_generator_module)

        # Run diagnostic
        if hasattr(condition_generator_module, 'run_diagnostic'):
            condition_generator_module.run_diagnostic()
        else:
            print("run_diagnostic関数が見つからない")

    except ImportError as e:
        print(f"必要なモジュールのインポートに失敗: {e}")

else:
    print("condition_generatorモジュールの読み込みに失敗")