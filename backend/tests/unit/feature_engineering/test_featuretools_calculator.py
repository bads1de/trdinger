"""
FeaturetoolsCalculatorのテスト

Featuretools Deep Feature Synthesis計算クラスのテストを実装します。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from app.services.ml.feature_engineering.automl_features.featuretools_calculator import (
    FeaturetoolsCalculator,
    FEATURETOOLS_AVAILABLE
)
from app.services.ml.feature_engineering.automl_features.automl_config import FeaturetoolsConfig


class TestFeaturetoolsCalculator:
    """FeaturetoolsCalculatorのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.config = FeaturetoolsConfig(
            enabled=True,
            max_depth=2,
            max_features=20  # テスト用に少なく設定
        )
        self.calculator = FeaturetoolsCalculator(self.config)

    def create_test_ohlcv_data(self, rows: int = 100) -> pd.DataFrame:
        """テスト用のOHLCVデータを作成"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=rows, freq='1h')
        
        # 現実的な価格データを生成
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, rows)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))
        
        prices = np.array(prices)
        
        data = {
            'Open': prices * (1 + np.random.normal(0, 0.001, rows)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, rows))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, rows))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 1, rows)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # High >= Close >= Low の制約を満たす
        df['High'] = np.maximum(df['High'], df[['Open', 'Close']].max(axis=1))
        df['Low'] = np.minimum(df['Low'], df[['Open', 'Close']].min(axis=1))
        
        return df

    def test_initialization(self):
        """初期化テスト"""
        # デフォルト設定での初期化
        calculator = FeaturetoolsCalculator()
        assert calculator.config is not None
        assert calculator.entityset is None
        assert calculator.feature_defs is None

        # カスタム設定での初期化
        custom_config = FeaturetoolsConfig(max_depth=3, max_features=50)
        calculator_custom = FeaturetoolsCalculator(custom_config)
        assert calculator_custom.config.max_depth == 3
        assert calculator_custom.config.max_features == 50

    def test_has_ohlcv_data(self):
        """OHLCVデータ検出テスト"""
        test_data = self.create_test_ohlcv_data(50)
        
        assert self.calculator._has_ohlcv_data(test_data) is True
        
        # OHLCVデータがない場合
        non_ohlcv_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        assert self.calculator._has_ohlcv_data(non_ohlcv_data) is False

    def test_create_price_groups(self):
        """価格グループ作成テスト"""
        test_data = self.create_test_ohlcv_data(50)
        
        price_groups = self.calculator._create_price_groups(test_data)
        
        assert isinstance(price_groups, pd.Series)
        assert len(price_groups) == len(test_data)
        
        # グループが文字列であることを確認
        assert all(isinstance(group, str) for group in price_groups)

    def test_create_price_entity(self):
        """価格エンティティ作成テスト"""
        test_data = self.create_test_ohlcv_data(50)
        test_data['price_group'] = self.calculator._create_price_groups(test_data)
        
        price_entity = self.calculator._create_price_entity(test_data)
        
        assert isinstance(price_entity, pd.DataFrame)
        if not price_entity.empty:
            assert 'price_group' in price_entity.columns

    def test_add_categorical_features(self):
        """カテゴリ特徴量追加テスト"""
        test_data = self.create_test_ohlcv_data(50)
        
        result_df = self.calculator._add_categorical_features(test_data)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(test_data)
        
        # 元の列が保持されているか確認
        for col in test_data.columns:
            assert col in result_df.columns

    def test_preprocess_data_for_entityset(self):
        """エンティティセット用データ前処理テスト"""
        test_data = self.create_test_ohlcv_data(50)
        
        processed_df = self.calculator._preprocess_data_for_entityset(test_data)
        
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(test_data)
        
        # 無限値がないことを確認
        assert not np.isinf(processed_df.select_dtypes(include=[np.number])).any().any()

    def test_get_primitives(self):
        """プリミティブ取得テスト"""
        agg_primitives, trans_primitives = self.calculator._get_primitives()
        
        assert isinstance(agg_primitives, list)
        assert isinstance(trans_primitives, list)
        assert len(agg_primitives) > 0
        assert len(trans_primitives) > 0

    def test_get_primitives_custom(self):
        """カスタムプリミティブ取得テスト"""
        custom_primitives = {
            'agg_primitives': ['mean', 'std'],
            'trans_primitives': ['add', 'subtract']
        }
        
        agg_primitives, trans_primitives = self.calculator._get_primitives(custom_primitives)
        
        assert agg_primitives == ['mean', 'std']
        assert trans_primitives == ['add', 'subtract']

    def test_clean_feature_names(self):
        """特徴量名クリーンアップテスト"""
        test_df = pd.DataFrame({
            'feature(1)': [1, 2, 3],
            'feature 2': [4, 5, 6],
            'feature,3': [7, 8, 9]
        })
        
        cleaned_df = self.calculator._clean_feature_names(test_df)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        
        # 特徴量名がクリーンアップされているか確認
        for col in cleaned_df.columns:
            assert col.startswith('FT_')
            assert '(' not in col
            assert ')' not in col
            assert ' ' not in col or '_' in col

    def test_merge_features_with_original(self):
        """特徴量結合テスト"""
        original_df = self.create_test_ohlcv_data(50)
        features = pd.DataFrame(np.random.randn(50, 3), columns=['feat1', 'feat2', 'feat3'])
        
        result_df = self.calculator._merge_features_with_original(original_df, features)
        
        assert len(result_df) == len(original_df)
        
        # 元の列が保持されているか確認
        for col in original_df.columns:
            assert col in result_df.columns
        
        # 新しい特徴量が追加されているか確認
        for col in features.columns:
            assert col in result_df.columns

    def test_get_feature_names_empty(self):
        """特徴量名取得テスト（空の場合）"""
        feature_names = self.calculator.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) == 0

    def test_get_synthesis_info_empty(self):
        """合成情報取得テスト（空の場合）"""
        synthesis_info = self.calculator.get_synthesis_info()
        assert isinstance(synthesis_info, dict)

    def test_get_entityset_info_empty(self):
        """エンティティセット情報取得テスト（空の場合）"""
        entityset_info = self.calculator.get_entityset_info()
        assert isinstance(entityset_info, dict)
        assert "message" in entityset_info

    def test_clear_entityset(self):
        """エンティティセットクリアテスト"""
        # エンティティセットにダミーデータを設定
        self.calculator.entityset = "dummy"
        self.calculator.feature_defs = ["dummy"]
        
        self.calculator.clear_entityset()
        
        assert self.calculator.entityset is None
        assert self.calculator.feature_defs is None

    def test_create_custom_primitives(self):
        """カスタムプリミティブ作成テスト"""
        custom_primitives = self.calculator.create_custom_primitives()
        
        assert isinstance(custom_primitives, dict)
        assert "agg_primitives" in custom_primitives
        assert "trans_primitives" in custom_primitives
        assert isinstance(custom_primitives["agg_primitives"], list)
        assert isinstance(custom_primitives["trans_primitives"], list)

    def test_optimize_feature_matrix(self):
        """特徴量マトリックス最適化テスト"""
        # テスト用の特徴量マトリックスを作成
        test_matrix = pd.DataFrame({
            'normal_feature': [1, 2, 3, 4, 5],
            'constant_feature': [1, 1, 1, 1, 1],  # 定数列
            'inf_feature': [1, 2, np.inf, 4, 5],  # 無限値を含む
            'corr_feature1': [1, 2, 3, 4, 5],
            'corr_feature2': [1.001, 2.001, 3.001, 4.001, 5.001]  # 高相関
        })
        
        optimized_matrix = self.calculator.optimize_feature_matrix(test_matrix)
        
        assert isinstance(optimized_matrix, pd.DataFrame)
        assert len(optimized_matrix) == len(test_matrix)
        
        # 定数列が除去されているか確認
        assert 'constant_feature' not in optimized_matrix.columns

    @pytest.mark.skipif(not FEATURETOOLS_AVAILABLE, reason="Featuretoolsライブラリが利用できません")
    def test_calculate_featuretools_features_basic(self):
        """基本的なFeaturetools特徴量計算テスト"""
        test_data = self.create_test_ohlcv_data(50)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.calculator.calculate_featuretools_features(
                test_data, max_depth=1, max_features=5
            )
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(test_data)
        
        # 元の列が保持されているか確認
        for col in test_data.columns:
            assert col in result_df.columns

    def test_calculate_featuretools_features_empty_input(self):
        """空データでのFeaturetools特徴量計算テスト"""
        empty_df = pd.DataFrame()
        
        result_df = self.calculator.calculate_featuretools_features(empty_df)
        
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty

    def test_calculate_featuretools_features_none_input(self):
        """Noneデータでの特徴量計算テスト"""
        result_df = self.calculator.calculate_featuretools_features(None)
        
        assert result_df is None

    @patch('app.services.ml.feature_engineering.automl_features.featuretools_calculator.FEATURETOOLS_AVAILABLE', False)
    def test_calculate_featuretools_features_library_unavailable(self):
        """Featuretoolsライブラリが利用できない場合のテスト"""
        test_data = self.create_test_ohlcv_data(50)
        
        result_df = self.calculator.calculate_featuretools_features(test_data)
        
        # 元のDataFrameがそのまま返されることを確認
        pd.testing.assert_frame_equal(result_df, test_data)
