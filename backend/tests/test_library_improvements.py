"""
ライブラリ置き換え修正のテスト

3.1と3.3の問題修正が正しく動作することを確認するテストファイル
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

# テスト対象のモジュールをインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer
from utils.label_generation import LabelGenerator, ThresholdMethod


class TestAdvancedFeatureEngineer:
    """AdvancedFeatureEngineerの修正テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.engineer = AdvancedFeatureEngineer()
        
        # テスト用のOHLCVデータを作成
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)
        
        self.test_data = pd.DataFrame({
            'Open': 50000 + np.random.randn(100) * 1000,
            'High': 50000 + np.random.randn(100) * 1000 + 500,
            'Low': 50000 + np.random.randn(100) * 1000 - 500,
            'Close': 50000 + np.random.randn(100) * 1000,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Closeが正の値になるように調整
        self.test_data['Close'] = np.abs(self.test_data['Close'])
        self.test_data['High'] = np.maximum(self.test_data['High'], self.test_data['Close'])
        self.test_data['Low'] = np.minimum(self.test_data['Low'], self.test_data['Close'])
    
    def test_trend_strength_calculation(self):
        """トレンド強度計算の修正テスト"""
        # 時系列特徴量を追加（トレンド強度を含む）
        result = self.engineer._add_time_series_features(self.test_data.copy())
        
        # トレンド強度の列が存在することを確認
        trend_columns = [col for col in result.columns if 'Trend_strength' in col]
        assert len(trend_columns) == 3  # window=[10, 20, 50]
        
        # 各トレンド強度列をチェック
        for col in trend_columns:
            # NaNでない値が存在することを確認
            non_nan_values = result[col].dropna()
            assert len(non_nan_values) > 0, f"{col}にNaNでない値が存在しません"
            
            # 値が数値であることを確認
            assert all(isinstance(val, (int, float)) for val in non_nan_values), \
                f"{col}に数値でない値が含まれています"
    
    def test_no_scipy_stats_import(self):
        """scipy.statsがインポートされていないことを確認"""
        import services.ml.feature_engineering.advanced_features as module
        
        # モジュールのソースコードを確認
        import inspect
        source = inspect.getsource(module)
        
        # scipy.statsのインポートがないことを確認
        assert 'from scipy import stats' not in source, \
            "scipy.statsのインポートが残っています"
        assert 'import scipy.stats' not in source, \
            "scipy.statsのインポートが残っています"
    
    def test_feature_engineering_performance(self):
        """特徴量エンジニアリングのパフォーマンステスト"""
        import time
        
        start_time = time.time()
        result = self.engineer.create_advanced_features(self.test_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 実行時間が合理的であることを確認（10秒以内）
        assert execution_time < 10, f"実行時間が長すぎます: {execution_time:.2f}秒"
        
        # 結果が適切な形状であることを確認
        assert isinstance(result, pd.DataFrame), "結果がDataFrameではありません"
        assert len(result) == len(self.test_data), "行数が一致しません"
        assert len(result.columns) > len(self.test_data.columns), "特徴量が追加されていません"


class TestLabelGenerator:
    """LabelGeneratorの修正テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.generator = LabelGenerator()
        
        # テスト用の価格データを作成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        
        # トレンドのある価格データを生成
        trend = np.linspace(50000, 55000, 1000)
        noise = np.random.randn(1000) * 500
        self.price_data = pd.Series(trend + noise, index=dates, name='Close')
    
    def test_kbins_discretizer_method(self):
        """KBinsDiscretizerメソッドのテスト"""
        # KBinsDiscretizerを使ったラベル生成
        labels, info = self.generator.generate_labels(
            self.price_data,
            method=ThresholdMethod.KBINS_DISCRETIZER,
            strategy='quantile'
        )
        
        # 基本的な検証
        assert isinstance(labels, pd.Series), "ラベルがSeriesではありません"
        assert isinstance(info, dict), "情報が辞書ではありません"
        
        # ラベルが0, 1, 2の値を持つことを確認
        unique_labels = set(labels.unique())
        expected_labels = {0, 1, 2}
        assert unique_labels == expected_labels, f"期待されるラベル{expected_labels}と異なります: {unique_labels}"
        
        # 情報辞書の内容を確認
        assert info['method'] == 'kbins_discretizer', "メソッド名が正しくありません"
        assert 'threshold_up' in info, "threshold_upが含まれていません"
        assert 'threshold_down' in info, "threshold_downが含まれていません"
        assert 'bin_edges' in info, "bin_edgesが含まれていません"
        assert 'actual_distribution' in info, "actual_distributionが含まれていません"
    
    def test_kbins_discretizer_strategies(self):
        """異なる戦略でのKBinsDiscretizerテスト"""
        strategies = ['uniform', 'quantile', 'kmeans']
        
        for strategy in strategies:
            labels, info = self.generator.generate_labels(
                self.price_data,
                method=ThresholdMethod.KBINS_DISCRETIZER,
                strategy=strategy
            )
            
            # 各戦略で適切にラベルが生成されることを確認
            assert len(labels) > 0, f"{strategy}戦略でラベルが生成されませんでした"
            assert info['strategy'] == strategy, f"戦略が正しく設定されていません: {info['strategy']}"
            
            # 分布が合理的であることを確認（各クラスに最低5%のデータ）
            distribution = info['actual_distribution']
            for class_name, ratio in distribution.items():
                assert ratio >= 0.05, f"{strategy}戦略の{class_name}クラスの比率が低すぎます: {ratio}"
    
    def test_convenience_method(self):
        """便利メソッドのテスト"""
        labels, info = self.generator.generate_labels_with_kbins_discretizer(
            self.price_data,
            strategy='quantile'
        )
        
        # 基本的な動作確認
        assert isinstance(labels, pd.Series), "ラベルがSeriesではありません"
        assert info['method'] == 'kbins_discretizer', "メソッドが正しくありません"
        assert info['strategy'] == 'quantile', "戦略が正しくありません"
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 空のデータでテスト
        empty_data = pd.Series([], dtype=float)
        
        # エラーが適切に処理されることを確認
        try:
            labels, info = self.generator.generate_labels(
                empty_data,
                method=ThresholdMethod.KBINS_DISCRETIZER
            )
            # フォールバックが動作することを確認
            assert info['method'] != 'kbins_discretizer', "フォールバックが動作していません"
        except Exception as e:
            # 適切なエラーメッセージが含まれることを確認
            assert "有効な価格変化率データがありません" in str(e) or "価格変化率" in str(e)
    
    def test_comparison_with_existing_methods(self):
        """既存メソッドとの比較テスト"""
        # 複数の方法でラベルを生成
        methods_to_test = [
            (ThresholdMethod.QUANTILE, {}),
            (ThresholdMethod.STD_DEVIATION, {'std_multiplier': 0.5}),
            (ThresholdMethod.KBINS_DISCRETIZER, {'strategy': 'quantile'})
        ]
        
        results = {}
        for method, params in methods_to_test:
            labels, info = self.generator.generate_labels(
                self.price_data,
                method=method,
                **params
            )
            results[method.value] = {
                'labels': labels,
                'info': info,
                'distribution': info.get('actual_distribution', {})
            }
        
        # すべての方法で適切にラベルが生成されることを確認
        for method_name, result in results.items():
            assert len(result['labels']) > 0, f"{method_name}でラベルが生成されませんでした"
            
            # 分布の合理性を確認
            if 'distribution' in result and result['distribution']:
                total_ratio = sum(result['distribution'].values())
                assert abs(total_ratio - 1.0) < 0.01, f"{method_name}の分布の合計が1に近くありません: {total_ratio}"


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v"])
