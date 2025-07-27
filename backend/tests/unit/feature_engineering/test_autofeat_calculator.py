"""
AutoFeatCalculatorのテスト

AutoFeat遺伝的特徴量選択クラスのテストを実装します。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from app.services.ml.feature_engineering.automl_features.autofeat_calculator import (
    AutoFeatCalculator,
    AUTOFEAT_AVAILABLE
)
from app.services.ml.feature_engineering.automl_features.automl_config import AutoFeatConfig


class TestAutoFeatCalculator:
    """AutoFeatCalculatorのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.config = AutoFeatConfig(
            enabled=True,
            max_features=10,  # テスト用に少なく設定
            generations=2,    # テスト用に少なく設定
            population_size=10,
            tournament_size=3
        )
        self.calculator = AutoFeatCalculator(self.config)

    def create_test_data(self, rows: int = 100) -> tuple:
        """テスト用のデータセットを作成"""
        np.random.seed(42)
        
        # 特徴量データ
        features = pd.DataFrame({
            'feature1': np.random.randn(rows),
            'feature2': np.random.randn(rows),
            'feature3': np.random.randn(rows),
            'feature4': np.random.randn(rows),
            'feature5': np.random.randn(rows)
        })
        
        # ターゲット変数（回帰用）
        target_regression = pd.Series(
            features['feature1'] * 2 + features['feature2'] + np.random.randn(rows) * 0.1,
            name='target'
        )
        
        # ターゲット変数（分類用）
        target_classification = pd.Series(
            np.random.choice([0, 1, 2], size=rows),
            name='target'
        )
        
        return features, target_regression, target_classification

    def test_initialization(self):
        """初期化テスト"""
        # デフォルト設定での初期化
        calculator = AutoFeatCalculator()
        assert calculator.config is not None
        assert calculator.autofeat_model is None
        assert calculator.selected_features is None

        # カスタム設定での初期化
        custom_config = AutoFeatConfig(max_features=20, generations=5)
        calculator_custom = AutoFeatCalculator(custom_config)
        assert calculator_custom.config.max_features == 20
        assert calculator_custom.config.generations == 5

    def test_preprocess_data(self):
        """データ前処理テスト"""
        features, target_regression, _ = self.create_test_data(50)
        
        # 無限値とNaNを含むデータを作成
        features_with_issues = features.copy()
        features_with_issues.loc[0, 'feature1'] = np.inf
        features_with_issues.loc[1, 'feature2'] = np.nan
        
        target_with_issues = target_regression.copy()
        target_with_issues.loc[2] = np.nan
        
        processed_df, processed_target = self.calculator._preprocess_data(
            features_with_issues, target_with_issues
        )
        
        assert isinstance(processed_df, pd.DataFrame)
        assert isinstance(processed_target, pd.Series)
        
        # 無限値とNaNが除去されているか確認
        assert not np.isinf(processed_df.values).any()
        assert not processed_df.isna().any().any()
        assert not processed_target.isna().any()

    def test_preprocess_data_constant_columns(self):
        """定数列除去テスト"""
        features, target_regression, _ = self.create_test_data(50)
        
        # 定数列を追加
        features_with_constant = features.copy()
        features_with_constant['constant_col'] = 1.0
        
        processed_df, processed_target = self.calculator._preprocess_data(
            features_with_constant, target_regression
        )
        
        # 定数列が除去されているか確認
        assert 'constant_col' not in processed_df.columns

    def test_preprocess_data_too_many_features(self):
        """特徴量数制限テスト"""
        # 101個の特徴量を作成
        large_features = pd.DataFrame(
            np.random.randn(50, 101),
            columns=[f'feature_{i}' for i in range(101)]
        )
        
        _, target_regression, _ = self.create_test_data(50)
        
        processed_df, processed_target = self.calculator._preprocess_data(
            large_features, target_regression
        )
        
        # 特徴量数が100個以下に制限されているか確認
        assert len(processed_df.columns) <= 100

    def test_calculate_feature_scores(self):
        """特徴量スコア計算テスト"""
        features, target_regression, _ = self.create_test_data(50)
        
        # AutoFeat特徴量を模擬
        autofeat_features = features.copy()
        autofeat_features.columns = [f'AF_{i}' for i in range(len(features.columns))]
        
        scores = self.calculator._calculate_feature_scores(
            autofeat_features, target_regression, "regression"
        )
        
        assert isinstance(scores, dict)
        assert len(scores) == len(autofeat_features.columns)
        
        for col, score in scores.items():
            assert col.startswith('AF_')
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_get_feature_names_empty(self):
        """特徴量名取得テスト（空の場合）"""
        feature_names = self.calculator.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) == 0

    def test_get_selection_info_empty(self):
        """選択情報取得テスト（空の場合）"""
        selection_info = self.calculator.get_selection_info()
        assert isinstance(selection_info, dict)

    def test_get_feature_scores_empty(self):
        """特徴量スコア取得テスト（空の場合）"""
        feature_scores = self.calculator.get_feature_scores()
        assert isinstance(feature_scores, dict)
        assert len(feature_scores) == 0

    def test_clear_model(self):
        """モデルクリアテスト"""
        # モデルにダミーデータを設定
        self.calculator.autofeat_model = "dummy"
        self.calculator.selected_features = ["dummy"]
        self.calculator.feature_scores = {"dummy": 0.5}
        
        self.calculator.clear_model()
        
        assert self.calculator.autofeat_model is None
        assert self.calculator.selected_features is None
        assert self.calculator.feature_scores == {}

    def test_evaluate_selected_features_empty(self):
        """特徴量評価テスト（AutoFeat特徴量なし）"""
        features, target_regression, _ = self.create_test_data(50)
        
        evaluation = self.calculator.evaluate_selected_features(
            features, target_regression, "regression"
        )
        
        assert isinstance(evaluation, dict)
        assert "error" in evaluation

    def test_evaluate_selected_features_insufficient_data(self):
        """特徴量評価テスト（データ不足）"""
        # 少ないデータを作成
        small_features = pd.DataFrame({
            'AF_1': [1, 2, 3],
            'AF_2': [4, 5, 6]
        })
        small_target = pd.Series([1, 2, 3])
        
        evaluation = self.calculator.evaluate_selected_features(
            small_features, small_target, "regression"
        )
        
        assert isinstance(evaluation, dict)
        assert "error" in evaluation

    def test_select_optimal_features_empty_input(self):
        """空データでの特徴量選択テスト"""
        empty_df = pd.DataFrame()
        empty_target = pd.Series(dtype=float)
        
        result_df, selection_info = self.calculator.select_optimal_features(
            empty_df, empty_target
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(selection_info, dict)
        assert "error" in selection_info

    def test_select_optimal_features_none_input(self):
        """Noneデータでの特徴量選択テスト"""
        result_df, selection_info = self.calculator.select_optimal_features(
            None, None
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(selection_info, dict)
        assert "error" in selection_info

    @patch('app.services.ml.feature_engineering.automl_features.autofeat_calculator.AUTOFEAT_AVAILABLE', False)
    def test_select_optimal_features_library_unavailable(self):
        """AutoFeatライブラリが利用できない場合のテスト"""
        features, target_regression, _ = self.create_test_data(50)
        
        result_df, selection_info = self.calculator.select_optimal_features(
            features, target_regression
        )
        
        # 元のDataFrameがそのまま返されることを確認
        pd.testing.assert_frame_equal(result_df, features)
        assert "error" in selection_info

    @pytest.mark.skipif(not AUTOFEAT_AVAILABLE, reason="AutoFeatライブラリが利用できません")
    def test_select_optimal_features_basic(self):
        """基本的な特徴量選択テスト"""
        features, target_regression, _ = self.create_test_data(30)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result_df, selection_info = self.calculator.select_optimal_features(
                df=features,
                target=target_regression,
                task_type="regression",
                max_features=5,
                generations=2,
                population_size=10
            )
        
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(selection_info, dict)
        
        if "error" not in selection_info:
            # 成功した場合の検証
            assert "original_features" in selection_info
            assert "selected_features" in selection_info
            assert "selection_time" in selection_info
            assert selection_info["original_features"] == len(features.columns)

    @pytest.mark.skipif(not AUTOFEAT_AVAILABLE, reason="AutoFeatライブラリが利用できません")
    def test_select_optimal_features_classification(self):
        """分類タスクでの特徴量選択テスト"""
        features, _, target_classification = self.create_test_data(30)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result_df, selection_info = self.calculator.select_optimal_features(
                df=features,
                target=target_classification,
                task_type="classification",
                max_features=5,
                generations=2,
                population_size=10
            )
        
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(selection_info, dict)
        
        if "error" not in selection_info:
            assert selection_info["task_type"] == "classification"

    def test_extract_selected_features_no_model(self):
        """モデルなしでの特徴量抽出テスト"""
        features, _, _ = self.create_test_data(50)
        
        result_df = self.calculator._extract_selected_features(features, features)
        
        # モデルがない場合は元のDataFrameが返される
        pd.testing.assert_frame_equal(result_df, features)

    @pytest.mark.skipif(not AUTOFEAT_AVAILABLE, reason="AutoFeatライブラリが利用できません")
    def test_evaluate_selected_features_with_autofeat_features(self):
        """AutoFeat特徴量での評価テスト"""
        # AutoFeat特徴量を模擬
        autofeat_features = pd.DataFrame({
            'AF_0': np.random.randn(50),
            'AF_1': np.random.randn(50),
            'AF_2': np.random.randn(50)
        })
        
        target = pd.Series(np.random.randn(50))
        
        evaluation = self.calculator.evaluate_selected_features(
            autofeat_features, target, "regression"
        )
        
        assert isinstance(evaluation, dict)
        
        if "error" not in evaluation:
            assert "cv_r2_mean" in evaluation
            assert "cv_r2_std" in evaluation
            assert "feature_count" in evaluation
            assert evaluation["feature_count"] == 3
