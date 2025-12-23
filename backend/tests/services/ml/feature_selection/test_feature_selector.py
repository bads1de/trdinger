"""
FeatureSelector のテスト

scikit-learn互換の特徴量選択器をテストします。
- Pipeline統合
- 各選択手法の動作確認
- エッジケースのハンドリング
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from app.services.ml.feature_selection.feature_selector import (
    FeatureSelector,
    FeatureSelectionConfig,
    SelectionMethod,
    VarianceStrategy,
    UnivariateStrategy,
    LassoStrategy,
    TreeBasedStrategy,
    PermutationStrategy,
    ShadowFeatureStrategy,
    StagedStrategy,
    RFECVStrategy,
    create_feature_selector,
)


class TestFeatureSelectionConfig:
    """設定クラスのテスト"""

    def test_default_config(self):
        """デフォルト設定の確認"""
        config = FeatureSelectionConfig()

        assert config.method == SelectionMethod.STAGED
        assert config.variance_threshold == 0.0
        assert config.correlation_threshold == 0.90
        assert config.min_features == 5
        assert config.cv_folds == 5

    def test_custom_config(self):
        """カスタム設定の確認"""
        config = FeatureSelectionConfig(
            method=SelectionMethod.SHADOW,
            target_k=10,
            cv_folds=3,
        )

        assert config.method == SelectionMethod.SHADOW
        assert config.target_k == 10
        assert config.cv_folds == 3


class TestFeatureSelectorBasics:
    """FeatureSelector の基本機能テスト"""

    @pytest.fixture
    def sample_data(self):
        """
        テスト用データを生成。
        f1, f2 は目的変数と強い相関、f3, f4 はノイズ、f5 は定数、f6 は f1 と高い相関。
        """
        np.random.seed(42)
        n = 100
        f1 = np.random.randn(n)
        f2 = np.random.randn(n)
        target = (f1 + f2 + np.random.normal(0, 0.5, n) > 0).astype(int)

        df = pd.DataFrame(
            {
                "f1": f1,
                "f2": f2,
                "f3": np.random.randn(n),  # Noise
                "f4": np.random.randn(n),  # Noise
                "f5": np.ones(n),  # Constant
                "f6": f1 * 0.99,  # High correlation with f1
            }
        )
        return df, pd.Series(target)

    def test_sklearn_compatibility(self, sample_data):
        """sklearn互換性の確認（fit/transform/get_support）"""
        X, y = sample_data
        selector = FeatureSelector(method="variance")

        # fit
        selector.fit(X, y)

        # transform
        X_transformed = selector.transform(X)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] < X.shape[1]  # 定数列が削除される

        # get_support
        support = selector.get_support()
        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]

    def test_pipeline_integration(self, sample_data):
        """Pipelineとの統合テスト（手動連携）"""
        X, y = sample_data
    
        # FeatureSelectorはfit_transformがDataFrameを返すように修正された
        selector = FeatureSelector(method="variance")
        X_selected = selector.fit_transform(X, y)
        
        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape[1] < X.shape[1]

    def test_feature_names_out(self, sample_data):
        """特徴量名の取得"""
        X, y = sample_data
        selector = FeatureSelector(method="variance")
        selector.fit(X, y)

        names = selector.get_feature_names_out()

        assert isinstance(names, np.ndarray)
        # 定数列 f5 が削除されているはず
        assert "f5" not in names

    def test_removes_constant_features(self, sample_data):
        """定数特徴量が削除されることを確認"""
        X, y = sample_data
        selector = FeatureSelector(method="variance", variance_threshold=0.0)
        selector.fit(X, y)

        support = selector.get_support()
        # f5 (index 4) は定数なので False
        assert not support[4]

    def test_removes_highly_correlated_features(self, sample_data):
        """高相関特徴量が削除されることを確認"""
        X, y = sample_data
        selector = FeatureSelector(method="variance", correlation_threshold=0.9)
        selector.fit(X, y)

        support = selector.get_support()
        # f1 か f6 のどちらかが削除される（相関 > 0.9）
        # 両方が True になることはない
        f1_selected = support[0]
        f6_selected = support[5]
        # 少なくとも片方は False（高相関で削除）
        # ただし f5 は定数なので別理由で削除
        assert not (f1_selected and f6_selected) or not f6_selected


class TestSelectionStrategies:
    """各選択戦略のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        np.random.seed(42)
        n = 100
        f1 = np.random.randn(n)
        f2 = np.random.randn(n)
        target = (f1 + f2 > 0).astype(int)

        X = np.column_stack(
            [
                f1,
                f2,
                np.random.randn(n),
                np.random.randn(n),
            ]
        )
        feature_names = ["f1", "f2", "f3", "f4"]
        return X, target, feature_names

    @pytest.fixture
    def config(self):
        """基本設定"""
        return FeatureSelectionConfig(
            min_features=2,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
        )

    def test_variance_strategy(self, sample_data, config):
        """分散ベース選択"""
        X, y, names = sample_data
        strategy = VarianceStrategy()

        mask, details = strategy.select(X, y, names, config)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert "variances" in details

    def test_univariate_strategy_f_classif(self, sample_data, config):
        """F統計量ベース選択"""
        X, y, names = sample_data
        strategy = UnivariateStrategy("f_classif")

        mask, details = strategy.select(X, y, names, config)

        assert mask.sum() >= config.min_features
        assert "scores" in details

    def test_univariate_strategy_mutual_info(self, sample_data, config):
        """相互情報量ベース選択"""
        X, y, names = sample_data
        strategy = UnivariateStrategy("mutual_info")

        mask, details = strategy.select(X, y, names, config)

        assert mask.sum() >= 1
        assert "scores" in details

    def test_lasso_strategy(self, sample_data, config):
        """Lasso正則化ベース選択"""
        X, y, names = sample_data
        strategy = LassoStrategy()

        mask, details = strategy.select(X, y, names, config)

        assert mask.sum() >= config.min_features
        assert "coefficients" in details

    def test_tree_based_strategy(self, sample_data, config):
        """ツリーベース（LightGBM/RandomForest）重要度ベース選択"""
        X, y, names = sample_data
        strategy = TreeBasedStrategy()

        mask, details = strategy.select(X, y, names, config)

        assert mask.sum() >= config.min_features
        assert "importances" in details

    def test_permutation_strategy(self, sample_data, config):
        """Permutation Importanceベース選択"""
        X, y, names = sample_data
        strategy = PermutationStrategy()

        mask, details = strategy.select(X, y, names, config)

        assert mask.sum() >= config.min_features
        assert "importances_mean" in details

    def test_rfecv_strategy(self, sample_data, config):
        """RFECVベース選択"""
        X, y, names = sample_data
        strategy = RFECVStrategy()

        mask, details = strategy.select(X, y, names, config)

        assert mask.sum() >= config.min_features
        assert "ranking" in details

    def test_shadow_strategy(self, sample_data, config):
        """シャドウ特徴量ベース選択（Boruta風）"""
        X, y, names = sample_data
        config.shadow_iterations = 5  # テスト高速化
        strategy = ShadowFeatureStrategy()

        mask, details = strategy.select(X, y, names, config)

        assert isinstance(mask, np.ndarray)
        assert "hit_counts" in details
        assert "confirmed_count" in details

    def test_staged_strategy(self, sample_data, config):
        """段階的選択"""
        X, y, names = sample_data
        config.staged_methods = [
            SelectionMethod.VARIANCE,
            SelectionMethod.MUTUAL_INFO,
        ]
        strategy = StagedStrategy()

        mask, details = strategy.select(X, y, names, config)

        assert mask.sum() >= 1
        assert "stages" in details
        assert len(details["stages"]) >= 1


class TestFeatureSelectorMethods:
    """FeatureSelector の各手法テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        np.random.seed(42)
        n = 100
        f1 = np.random.randn(n)
        f2 = np.random.randn(n)
        target = (f1 + f2 + np.random.normal(0, 0.3, n) > 0).astype(int)

        df = pd.DataFrame(
            {
                "signal_1": f1,
                "signal_2": f2,
                "noise_1": np.random.randn(n),
                "noise_2": np.random.randn(n),
            }
        )
        return df, pd.Series(target)

    @pytest.mark.parametrize(
        "method",
        [
            "variance",
            "univariate_f",
            "mutual_info",
            "lasso",
            "random_forest",
            "permutation",
            "rfecv",
            "shadow",
            "staged",
        ],
    )
    def test_all_methods_smoke(self, sample_data, method):
        """全手法のスモークテスト"""
        X, y = sample_data

        selector = FeatureSelector(
            method=method,
            min_features=2,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
            shadow_iterations=3,  # 高速化
        )

        selector.fit(X, y)
        X_sel = selector.transform(X)

        # 最低限の特徴量が残る
        assert X_sel.shape[1] >= 1
        # サンプル数は変わらない
        assert X_sel.shape[0] == X.shape[0]


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_dataframe_raises_error(self):
        """空のDataFrameでエラーが発生"""
        selector = FeatureSelector()
        X_empty = pd.DataFrame()
        y_empty = pd.Series([], dtype=int)

        with pytest.raises(ValueError, match="Empty input"):
            selector.fit(X_empty, y_empty)

    def test_mismatched_samples_raises_error(self):
        """サンプル数不一致でエラーが発生"""
        selector = FeatureSelector()
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1])

        with pytest.raises(ValueError, match="inconsistent samples"):
            selector.fit(X, y)

    def test_handles_nan_values(self):
        """NaN値を適切に処理"""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 4.0, 5.0],
                "b": [2.0, 3.0, np.nan, 5.0, 6.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        selector = FeatureSelector(method="variance")
        selector.fit(X, y)

        # エラーなく完了
        assert selector.get_support().sum() >= 1

    def test_handles_inf_values(self):
        """無限値を適切に処理"""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": [1.0, np.inf, 3.0, 4.0, 5.0],
                "b": [2.0, 3.0, -np.inf, 5.0, 6.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        selector = FeatureSelector(method="variance")
        selector.fit(X, y)

        # エラーなく完了
        assert selector.get_support().sum() >= 1

    def test_numpy_array_input(self):
        """NumPy配列入力の対応"""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        selector = FeatureSelector(method="variance")
        selector.fit(X, y)
        X_sel = selector.transform(X)

        assert X_sel.shape[1] >= 1


class TestBackwardCompatibility:
    """後方互換性のテスト"""

    def test_create_feature_selector_factory(self):
        """ファクトリー関数の動作確認"""
        selector = create_feature_selector(method="variance")

        assert isinstance(selector, FeatureSelector)
        assert selector.method == "variance"

    def test_selection_details_available(self):
        """選択詳細が取得可能"""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(50),
                "b": np.random.randn(50),
            }
        )
        y = pd.Series(np.random.randint(0, 2, 50))

        selector = FeatureSelector(method="variance")
        selector.fit(X, y)

        # selection_details_ 属性が存在
        assert hasattr(selector, "selection_details_")
        assert isinstance(selector.selection_details_, dict)
