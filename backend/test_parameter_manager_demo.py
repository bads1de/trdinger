#!/usr/bin/env python3
"""
ParameterManagerのデモンストレーション
"""

import sys
sys.path.append('.')

from app.core.services.indicators.parameter_manager import IndicatorParameterManager
from app.core.services.indicators.config.indicator_config import (
    IndicatorConfig, 
    ParameterConfig, 
    IndicatorResultType
)

def main():
    print("=== ParameterManagerのデモンストレーション ===\n")
    
    manager = IndicatorParameterManager()

    # MACD設定を作成
    print("1. MACD パラメータ生成テスト")
    macd_config = IndicatorConfig(
        indicator_name="MACD",
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="fast_period",
            default_value=12,
            min_value=2,
            max_value=20,
            description="短期期間",
        )
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="slow_period",
            default_value=26,
            min_value=15,
            max_value=50,
            description="長期期間",
        )
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="signal_period",
            default_value=9,
            min_value=2,
            max_value=20,
            description="シグナル期間",
        )
    )

    # 複数回生成してfast_period < slow_periodを確認
    print("複数回生成して制約が適用されることを確認:")
    for i in range(5):
        params = manager.generate_parameters("MACD", macd_config)
        print(f"  生成{i+1}: {params}")
        print(f"    fast < slow: {params['fast_period'] < params['slow_period']}")
        print(f"    バリデーション: {manager.validate_parameters('MACD', params, macd_config)}")
    print()

    # Stochastic設定を作成
    print("2. Stochastic パラメータ生成テスト")
    stoch_config = IndicatorConfig(
        indicator_name="STOCH",
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="fastk_period",
            default_value=5,
            min_value=2,
            max_value=20,
            description="Fast %K期間",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="slowk_period",
            default_value=3,
            min_value=2,
            max_value=10,
            description="Slow %K期間",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="slowk_matype",
            default_value=0,
            min_value=0,
            max_value=20,  # 制約エンジンが0-8に制限する
            description="Slow %K MA種別",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="slowd_period",
            default_value=3,
            min_value=2,
            max_value=10,
            description="Slow %D期間",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="slowd_matype",
            default_value=0,
            min_value=0,
            max_value=20,  # 制約エンジンが0-8に制限する
            description="Slow %D MA種別",
        )
    )

    print("複数回生成してMA種別制約が適用されることを確認:")
    for i in range(5):
        params = manager.generate_parameters("STOCH", stoch_config)
        print(f"  生成{i+1}: {params}")
        print(f"    slowk_matype範囲: 0 <= {params['slowk_matype']} <= 8: {0 <= params['slowk_matype'] <= 8}")
        print(f"    slowd_matype範囲: 0 <= {params['slowd_matype']} <= 8: {0 <= params['slowd_matype'] <= 8}")
        print(f"    バリデーション: {manager.validate_parameters('STOCH', params, stoch_config)}")
    print()

    # 標準的なインディケーター（制約なし）
    print("3. RSI パラメータ生成テスト（制約なし）")
    rsi_config = IndicatorConfig(
        indicator_name="RSI",
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
    )
    rsi_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="RSI計算期間",
        )
    )

    for i in range(3):
        params = manager.generate_parameters("RSI", rsi_config)
        print(f"  生成{i+1}: {params}")
        print(f"    バリデーション: {manager.validate_parameters('RSI', params, rsi_config)}")
    print()

    print("=== デモ完了 ===")

if __name__ == "__main__":
    main()
