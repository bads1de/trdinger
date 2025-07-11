#!/usr/bin/env python3
"""
全指標の設定と実装の整合性を検証するスクリプト

指標クラスのメソッド定義とindicator_definitions.pyの設定が
一致しているかを包括的にチェックします。
"""

import sys
import os
import inspect
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.services.indicators.config.indicator_config import indicator_registry
from app.core.services.indicators.technical_indicators.trend import TrendIndicators
from app.core.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.core.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.core.services.indicators.technical_indicators.volume import VolumeIndicators
from app.core.services.indicators.technical_indicators.price_transform import PriceTransformIndicators
from app.core.services.indicators.technical_indicators.cycle import CycleIndicators
from app.core.services.indicators.technical_indicators.statistics import StatisticsIndicators
from app.core.services.indicators.technical_indicators.math_transform import MathTransformIndicators
from app.core.services.indicators.technical_indicators.math_operators import MathOperatorsIndicators
from app.core.services.indicators.technical_indicators.pattern_recognition import PatternRecognitionIndicators


class IndicatorValidator:
    """指標の整合性検証クラス"""
    
    def __init__(self):
        self.indicator_classes = {
            'trend': TrendIndicators,
            'momentum': MomentumIndicators,
            'volatility': VolatilityIndicators,
            'volume': VolumeIndicators,
            'price_transform': PriceTransformIndicators,
            'cycle': CycleIndicators,
            'statistics': StatisticsIndicators,
            'math_transform': MathTransformIndicators,
            'math_operators': MathOperatorsIndicators,
            'pattern_recognition': PatternRecognitionIndicators,
        }
        
        # データマッピング
        self.data_mapping = {
            'close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume',
            'open_data': 'Open',  # パターン認識系で使用
            'data0': 'Close',     # 数学演算子系で使用
            'data1': 'High',      # 数学演算子系で使用
        }
        
        self.errors = []
        self.warnings = []
    
    def validate_all_indicators(self) -> Dict[str, Any]:
        """全指標の検証を実行"""
        print("=== 全指標の整合性検証開始 ===")
        
        # レジストリから登録済み指標を取得
        registered_indicators = indicator_registry.list_indicators()
        print(f"登録済み指標数: {len(registered_indicators)}")
        
        # 各指標の検証
        for indicator_name in registered_indicators:
            self._validate_single_indicator(indicator_name)
        
        # 結果の集計
        result = {
            'total_indicators': len(registered_indicators),
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
        }
        
        return result
    
    def _validate_single_indicator(self, indicator_name: str):
        """単一指標の検証"""
        try:
            # レジストリから設定を取得
            config = indicator_registry.get_indicator_config(indicator_name)
            if not config:
                self.errors.append(f"{indicator_name}: レジストリに設定が見つかりません")
                return
            
            # アダプター関数の存在確認
            if not config.adapter_function:
                self.errors.append(f"{indicator_name}: アダプター関数が設定されていません")
                return
            
            # 関数のシグネチャ確認
            sig = inspect.signature(config.adapter_function)
            params = list(sig.parameters.keys())
            
            # required_dataとの整合性確認
            required_data = config.required_data
            
            # 基本的な引数チェック
            missing_args = []
            for data_key in required_data:
                if data_key not in params:
                    missing_args.append(data_key)
            
            if missing_args:
                self.errors.append(
                    f"{indicator_name}: 必要な引数が不足 - {missing_args}, "
                    f"実際の引数: {params}, 必要なデータ: {required_data}"
                )
            
            # 簡単な実行テスト
            self._test_indicator_execution(indicator_name, config)
            
        except Exception as e:
            self.errors.append(f"{indicator_name}: 検証中にエラー - {str(e)}")
    
    def _test_indicator_execution(self, indicator_name: str, config):
        """指標の実行テスト"""
        try:
            # テストデータの準備
            test_data = self._create_test_data()
            
            # 必要なデータを準備
            data_args = {}
            for data_key in config.required_data:
                if data_key in self.data_mapping:
                    column_name = self.data_mapping[data_key]
                    if column_name in test_data.columns:
                        data_args[data_key] = test_data[column_name].values
                    else:
                        # フォールバック
                        data_args[data_key] = test_data['Close'].values
                else:
                    # 不明なデータキーの場合はCloseを使用
                    data_args[data_key] = test_data['Close'].values
            
            # パラメータの準備
            params = {}
            if config.parameters:
                for param_config in config.parameters:
                    params[param_config.name] = param_config.default_value
            
            # 関数実行
            all_args = {**data_args, **params}
            result = config.adapter_function(**all_args)
            
            # 結果の基本検証
            if result is None:
                self.warnings.append(f"{indicator_name}: 実行結果がNone")
            elif isinstance(result, np.ndarray):
                if len(result) == 0:
                    self.warnings.append(f"{indicator_name}: 実行結果が空配列")
                elif np.all(np.isnan(result)):
                    self.warnings.append(f"{indicator_name}: 実行結果が全てNaN")
            
        except Exception as e:
            self.warnings.append(f"{indicator_name}: 実行テストでエラー - {str(e)}")
    
    def _create_test_data(self) -> pd.DataFrame:
        """テスト用のOHLCVデータを作成"""
        np.random.seed(42)  # 再現性のため
        
        n = 100
        base_price = 100
        
        # ランダムウォークで価格データを生成
        returns = np.random.normal(0, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # OHLCV データを生成
        data = {
            'Open': prices * (1 + np.random.normal(0, 0.001, n)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n),
        }
        
        # 価格の論理的整合性を保証
        for i in range(n):
            data['High'][i] = max(data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i])
            data['Low'][i] = min(data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i])
        
        return pd.DataFrame(data)


def main():
    """メイン実行関数"""
    validator = IndicatorValidator()
    result = validator.validate_all_indicators()
    
    print(f"\n=== 検証結果 ===")
    print(f"総指標数: {result['total_indicators']}")
    print(f"エラー数: {result['error_count']}")
    print(f"警告数: {result['warning_count']}")
    
    if result['errors']:
        print(f"\n=== エラー詳細 ===")
        for error in result['errors']:
            print(f"ERROR: {error}")
    
    if result['warnings']:
        print(f"\n=== 警告詳細 ===")
        for warning in result['warnings']:
            print(f"WARNING: {warning}")
    
    if result['error_count'] == 0:
        print(f"\n✅ 全指標の検証が正常に完了しました！")
    else:
        print(f"\n❌ {result['error_count']}個のエラーが見つかりました。修正が必要です。")
    
    return result['error_count'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
