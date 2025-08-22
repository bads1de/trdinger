#!/usr/bin/env python3

import inspect
from app.services.indicators.parameter_manager import normalize_params
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from unittest.mock import Mock

def debug_normalize_params():
    # モック設定を作成
    config = Mock()
    config.indicator_name = 'LINEARREG'
    config.parameters = {}
    config.param_map = {}

    # LINEARREGのパラメータテスト
    params = {'period': 14}
    result = normalize_params('LINEARREG', params, config)
    print('LINEARREG result:', result)

    # STOCHRSIのパラメータテスト
    config.indicator_name = 'STOCHRSI'
    params = {'period': 14, 'fastk_period': 5, 'fastd_period': 3}
    result = normalize_params('STOCHRSI', params, config)
    print('STOCHRSI result:', result)

    # KSTのパラメータテスト
    config.indicator_name = 'KST'
    params = {'r1': 10, 'r2': 15, 'r3': 20, 'r4': 30}
    result = normalize_params('KST', params, config)
    print('KST result:', result)

    # STOCHRSI関数のシグネチャを確認
    print('\nSTOCHRSI function signature:')
    sig = inspect.signature(MomentumIndicators.stochrsi)
    print('Parameters:', list(sig.parameters.keys()))
    for name, param in sig.parameters.items():
        print(f'  {name}: {param}')

if __name__ == "__main__":
    debug_normalize_params()