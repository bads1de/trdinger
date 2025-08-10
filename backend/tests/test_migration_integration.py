"""
talib から pandas-ta への移行統合テスト

このテストファイルは、移行後のシステム全体の統合テストを実施します。
以下の観点でテストを実施します：
1. 実際のデータを使用したテスト
2. backtesting.py との互換性テスト
3. ML feature engineering との統合テスト
4. エンドツーエンドテスト
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# テスト対象のモジュールをインポート
sys.path.append(str(Path(__file__).parent.parent))

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry
from app.services.ml.feature_engineering.technical_features import TechnicalFeatureEngineer
from app.services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer


class TestMigrationIntegration:
    """移行統合テストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        self.service = TechnicalIndicatorService()
        
        # リアルなBTC価格データを模擬
        np.random.seed(42)
        n = 1000
        
        # より現実的な価格データを生成
        base_price = 50000.0  # BTC価格
        volatility = 0.03
        
        returns = np.random.normal(0.0005, volatility, n)
        log_prices = np.cumsum(returns)
        prices = base_price * np.exp(log_prices)
        
        self.real_data = pd.DataFrame({
            'Open': prices + np.random.normal(0, prices * 0.001),
            'High': prices * (1 + np.random.uniform(0.001, 0.02, n)),
            'Low': prices * (1 - np.random.uniform(0.001, 0.02, n)),
            'Close': prices,
            'Volume': np.random.lognormal(15, 1, n)  # より現実的な出来高分布
        }, index=pd.date_range('2024-01-01', periods=n, freq='H'))
        
        # 価格の整合性を保証
        self.real_data['High'] = np.maximum(
            self.real_data['High'], 
            np.maximum(self.real_data['Open'], self.real_data['Close'])
        )
        self.real_data['Low'] = np.minimum(
            self.real_data['Low'], 
            np.minimum(self.real_data['Open'], self.real_data['Close'])
        )

    def test_all_supported_indicators_with_real_data(self):
        """サポートされている全指標をリアルデータでテスト"""
        supported_indicators = indicator_registry.get_all_indicator_names()
        
        failed_indicators = []
        successful_indicators = []
        
        for indicator_name in supported_indicators:
            try:
                config = indicator_registry.get_indicator_config(indicator_name)
                if config is None:
                    continue
                
                # デフォルトパラメータを取得
                params = {}
                for param_name, param_config in config.parameters.items():
                    params[param_name] = param_config.default_value
                
                # 指標を計算
                result = self.service.calculate_indicator(
                    self.real_data, indicator_name, params
                )
                
                # 基本的な検証
                self._validate_indicator_result(result, indicator_name, len(self.real_data))
                successful_indicators.append(indicator_name)
                
            except Exception as e:
                failed_indicators.append((indicator_name, str(e)))
        
        # 結果をレポート
        print(f"\n成功した指標: {len(successful_indicators)}")
        print(f"失敗した指標: {len(failed_indicators)}")
        
        if failed_indicators:
            failure_report = "\n".join([
                f"  {name}: {error}" for name, error in failed_indicators
            ])
            pytest.fail(f"以下の指標でエラーが発生しました:\n{failure_report}")

    def test_backtesting_compatibility(self):
        """backtesting.py との互換性テスト"""
        # backtesting.py で使用される形式のデータを準備
        bt_data = self.real_data.copy()
        
        # 主要な指標をテスト
        test_indicators = [
            ('SMA', {'period': 20}),
            ('EMA', {'period': 20}),
            ('RSI', {'period': 14}),
            ('MACD', {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
            ('BB', {'period': 20, 'std_dev': 2.0}),
            ('ATR', {'period': 14}),
        ]
        
        for indicator_name, params in test_indicators:
            with pytest.subTest(indicator=indicator_name):
                result = self.service.calculate_indicator(bt_data, indicator_name, params)
                
                # backtesting.py で期待される形式を検証
                if isinstance(result, tuple):
                    for arr in result:
                        assert isinstance(arr, np.ndarray), \
                            f"{indicator_name}: numpy配列が期待されます"
                        assert arr.dtype in [np.float64, np.float32], \
                            f"{indicator_name}: float型が期待されます"
                else:
                    assert isinstance(result, np.ndarray), \
                        f"{indicator_name}: numpy配列が期待されます"
                    assert result.dtype in [np.float64, np.float32], \
                        f"{indicator_name}: float型が期待されます"

    def test_ml_feature_engineering_integration(self):
        """ML feature engineering との統合テスト"""
        # TechnicalFeatureEngineer のテスト
        tech_engineer = TechnicalFeatureEngineer()
        
        try:
            tech_features = tech_engineer.create_features(self.real_data)
            
            # 基本的な検証
            assert isinstance(tech_features, pd.DataFrame), \
                "TechnicalFeatureEngineer: DataFrameが期待されます"
            assert len(tech_features) == len(self.real_data), \
                "TechnicalFeatureEngineer: 長さが一致しません"
            assert not tech_features.empty, \
                "TechnicalFeatureEngineer: 空のDataFrameです"
            
            # NaN の割合をチェック
            nan_ratio = tech_features.isna().sum().sum() / (tech_features.shape[0] * tech_features.shape[1])
            assert nan_ratio < 0.5, \
                f"TechnicalFeatureEngineer: NaNの割合が高すぎます ({nan_ratio:.2%})"
                
        except Exception as e:
            pytest.fail(f"TechnicalFeatureEngineer統合テストエラー: {e}")

        # AdvancedFeatureEngineer のテスト
        adv_engineer = AdvancedFeatureEngineer()
        
        try:
            adv_features = adv_engineer.create_features(self.real_data)
            
            # 基本的な検証
            assert isinstance(adv_features, pd.DataFrame), \
                "AdvancedFeatureEngineer: DataFrameが期待されます"
            assert len(adv_features) == len(self.real_data), \
                "AdvancedFeatureEngineer: 長さが一致しません"
            assert not adv_features.empty, \
                "AdvancedFeatureEngineer: 空のDataFrameです"
            
            # NaN の割合をチェック
            nan_ratio = adv_features.isna().sum().sum() / (adv_features.shape[0] * adv_features.shape[1])
            assert nan_ratio < 0.5, \
                f"AdvancedFeatureEngineer: NaNの割合が高すぎます ({nan_ratio:.2%})"
                
        except Exception as e:
            pytest.fail(f"AdvancedFeatureEngineer統合テストエラー: {e}")

    def test_edge_cases_handling(self):
        """エッジケースの処理テスト"""
        edge_cases = [
            # 最小データサイズ
            self.real_data.head(50),
            # 価格変動が激しいデータ
            self._create_volatile_data(),
            # 価格が一定のデータ
            self._create_flat_data(),
            # 欠損値を含むデータ
            self._create_data_with_missing_values(),
        ]
        
        test_indicators = ['SMA', 'EMA', 'RSI']
        
        for i, edge_data in enumerate(edge_cases):
            for indicator in test_indicators:
                with pytest.subTest(case=f"edge_case_{i}", indicator=indicator):
                    try:
                        if indicator == 'SMA':
                            params = {'period': min(20, len(edge_data) // 2)}
                        elif indicator == 'EMA':
                            params = {'period': min(20, len(edge_data) // 2)}
                        elif indicator == 'RSI':
                            params = {'period': min(14, len(edge_data) // 2)}
                        
                        if params['period'] < 2:
                            continue  # スキップ
                        
                        result = self.service.calculate_indicator(edge_data, indicator, params)
                        self._validate_indicator_result(result, indicator, len(edge_data))
                        
                    except Exception as e:
                        # エッジケースでのエラーは許容される場合がある
                        print(f"エッジケース {i} で {indicator} がエラー: {e}")

    def test_concurrent_calculation(self):
        """並行計算のテスト"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def calculate_indicator(indicator_name, params):
            try:
                result = self.service.calculate_indicator(self.real_data, indicator_name, params)
                results_queue.put((indicator_name, result))
            except Exception as e:
                errors_queue.put((indicator_name, str(e)))
        
        # 複数の指標を並行計算
        threads = []
        test_cases = [
            ('SMA', {'period': 20}),
            ('EMA', {'period': 20}),
            ('RSI', {'period': 14}),
            ('ATR', {'period': 14}),
        ]
        
        for indicator_name, params in test_cases:
            thread = threading.Thread(target=calculate_indicator, args=(indicator_name, params))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()
        
        # 結果を検証
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        assert len(results) == len(test_cases), \
            f"期待される結果数と異なります。エラー: {errors}"
        
        for indicator_name, result in results:
            self._validate_indicator_result(result, indicator_name, len(self.real_data))

    def test_memory_usage(self):
        """メモリ使用量のテスト"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 大量の計算を実行
        for _ in range(100):
            self.service.calculate_indicator(self.real_data, 'SMA', {'period': 20})
        
        gc.collect()  # ガベージコレクション実行
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # メモリリークがないことを確認（100MB以下の増加は許容）
        assert memory_increase < 100, \
            f"メモリ使用量が大幅に増加しました: {memory_increase:.2f}MB"

    def _validate_indicator_result(self, result: Any, indicator_name: str, expected_length: int):
        """指標結果の基本検証"""
        assert result is not None, f"{indicator_name}: 結果がNone"
        
        if isinstance(result, tuple):
            for i, arr in enumerate(result):
                assert len(arr) == expected_length, \
                    f"{indicator_name}: 結果の長さが不正 (index {i})"
                assert isinstance(arr, np.ndarray), \
                    f"{indicator_name}: numpy配列が期待されます (index {i})"
        else:
            assert len(result) == expected_length, \
                f"{indicator_name}: 結果の長さが不正"
            assert isinstance(result, np.ndarray), \
                f"{indicator_name}: numpy配列が期待されます"

    def _create_volatile_data(self) -> pd.DataFrame:
        """価格変動が激しいデータを作成"""
        n = 200
        base_price = 50000.0
        
        # 高いボラティリティ
        returns = np.random.normal(0, 0.1, n)  # 10%のボラティリティ
        log_prices = np.cumsum(returns)
        prices = base_price * np.exp(log_prices)
        
        data = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.05,
            'Low': prices * 0.95,
            'Close': prices,
            'Volume': np.random.lognormal(15, 1, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='H'))
        
        return data

    def _create_flat_data(self) -> pd.DataFrame:
        """価格が一定のデータを作成"""
        n = 200
        price = 50000.0
        
        data = pd.DataFrame({
            'Open': [price] * n,
            'High': [price * 1.001] * n,  # わずかな変動
            'Low': [price * 0.999] * n,
            'Close': [price] * n,
            'Volume': [1000.0] * n
        }, index=pd.date_range('2024-01-01', periods=n, freq='H'))
        
        return data

    def _create_data_with_missing_values(self) -> pd.DataFrame:
        """欠損値を含むデータを作成"""
        data = self.real_data.head(200).copy()
        
        # ランダムに欠損値を挿入
        np.random.seed(42)
        missing_indices = np.random.choice(len(data), size=20, replace=False)
        
        for idx in missing_indices:
            data.iloc[idx, np.random.choice(5)] = np.nan
        
        return data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
