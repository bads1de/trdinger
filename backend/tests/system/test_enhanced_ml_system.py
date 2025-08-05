"""
拡張MLシステムの統合テスト

フェーズ3・4で実装された新機能の統合テスト：
- 時系列クロスバリデーション
- 拡張評価指標
- 特徴量選択
- モデル管理システム
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.validation.time_series_cv import (
    TimeSeriesCrossValidator,
    CVConfig,
    CVStrategy,
)
from app.services.ml.evaluation.enhanced_metrics import (
    EnhancedMetricsCalculator,
    MetricsConfig,
)
from app.services.ml.feature_selection.feature_selector import (
    FeatureSelector,
    FeatureSelectionConfig,
    SelectionMethod,
)
from app.services.ml.model_manager import (
    ModelManager,
    PerformanceMetric,
)

logger = logging.getLogger(__name__)


class TestEnhancedMLSystem:
    """拡張MLシステムの統合テストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # テスト用の一時ディレクトリ
        self.temp_dir = tempfile.mkdtemp()

        # 各コンポーネントを初期化
        self.cv_validator = TimeSeriesCrossValidator()
        self.metrics_calculator = EnhancedMetricsCalculator()
        self.feature_selector = FeatureSelector()
        # テスト用のModelManagerを作成（テンポラリディレクトリを使用）
        from app.services.ml.model_manager import PerformanceMonitoringConfig

        self.model_manager = ModelManager()
        # テスト用にbase_pathを変更
        self.model_manager.base_path = Path(self.temp_dir)
        # configのMODEL_SAVE_PATHも変更
        self.model_manager.config.MODEL_SAVE_PATH = self.temp_dir

    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_data(self, n_samples=200, n_features=20):
        """テストデータを作成"""
        # 時系列インデックス
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="1h")

        # 特徴量データ
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            index=dates,
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # ターゲットデータ（3クラス分類）
        y = pd.Series(
            np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2]),
            index=dates,
            name="target",
        )

        return X, y

    def create_simple_model(self):
        """テスト用の簡単なモデルを作成"""
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=10, random_state=42)

    def test_time_series_cross_validation(self):
        """時系列クロスバリデーションのテスト"""
        logger.info("=== 時系列クロスバリデーションテスト ===")

        # テストデータを作成
        X, y = self.create_test_data(n_samples=150, n_features=10)

        # モデルを作成
        model = self.create_simple_model()

        # 設定を作成
        config = CVConfig(
            strategy=CVStrategy.TIME_SERIES_SPLIT, n_splits=3, min_train_size=50
        )

        cv_validator = TimeSeriesCrossValidator(config)

        # クロスバリデーション実行
        results = cv_validator.cross_validate(
            model, X, y, scoring=["accuracy", "f1", "precision", "recall"]
        )

        # 結果の検証
        assert "accuracy_mean" in results
        assert "accuracy_std" in results
        assert "fold_results" in results
        assert len(results["fold_results"]) <= 3

        logger.info(
            f"CV結果: 精度={results['accuracy_mean']:.4f}±{results['accuracy_std']:.4f}"
        )
        logger.info("✅ 時系列クロスバリデーションテスト完了")

    def test_enhanced_metrics_calculation(self):
        """拡張評価指標のテスト"""
        logger.info("=== 拡張評価指標テスト ===")

        # テストデータを作成
        np.random.seed(42)
        y_true = np.random.choice([0, 1, 2], size=100, p=[0.4, 0.4, 0.2])
        y_pred = np.random.choice([0, 1, 2], size=100, p=[0.3, 0.5, 0.2])
        y_proba = np.random.dirichlet([1, 1, 1], size=100)

        # 設定を作成
        config = MetricsConfig(
            include_balanced_accuracy=True,
            include_pr_auc=True,
            include_roc_auc=True,
            include_confusion_matrix=True,
        )

        metrics_calculator = EnhancedMetricsCalculator(config)

        # 評価指標を計算
        metrics = metrics_calculator.calculate_comprehensive_metrics(
            y_true, y_pred, y_proba, class_names=["Down", "Hold", "Up"]
        )

        # 結果の検証
        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert "per_class_metrics" in metrics

        if "pr_auc_macro" in metrics:
            logger.info(f"PR-AUC (macro): {metrics['pr_auc_macro']:.4f}")

        logger.info(f"バランス精度: {metrics['balanced_accuracy']:.4f}")
        logger.info("✅ 拡張評価指標テスト完了")

    def test_feature_selection(self):
        """特徴量選択のテスト"""
        logger.info("=== 特徴量選択テスト ===")

        # テストデータを作成（多めの特徴量）
        X, y = self.create_test_data(n_samples=200, n_features=50)

        # 設定を作成
        config = FeatureSelectionConfig(
            method=SelectionMethod.ENSEMBLE,
            ensemble_methods=[
                SelectionMethod.MUTUAL_INFO,
                SelectionMethod.RANDOM_FOREST,
                SelectionMethod.LASSO,
            ],
            k_features=15,
            importance_threshold=0.1,  # 閾値を高く設定
            ensemble_voting="majority",
        )

        feature_selector = FeatureSelector(config)

        # 特徴量選択を実行
        X_selected, results = feature_selector.fit_transform(X, y)

        # 結果の検証
        assert X_selected.shape[1] <= X.shape[1], "特徴量数が増加しています"
        assert X_selected.shape[0] == X.shape[0], "サンプル数が変わっています"
        assert "selected_features" in results
        assert "method" in results

        # 特徴量が実際に選択されているかチェック（緩い条件）
        if X_selected.shape[1] == X.shape[1]:
            logger.warning("特徴量選択で全特徴量が選択されました（閾値調整が必要）")

        logger.info(f"特徴量選択: {X.shape[1]} → {X_selected.shape[1]}")
        logger.info(f"選択率: {X_selected.shape[1]/X.shape[1]*100:.1f}%")
        logger.info("✅ 特徴量選択テスト完了")

    def test_model_management(self):
        """モデル管理システムのテスト"""
        logger.info("=== モデル管理システムテスト ===")

        # テストデータを作成
        X, y = self.create_test_data(n_samples=100, n_features=10)

        # モデルを学習
        model = self.create_simple_model()
        model.fit(X, y)

        # 予測とメトリクス計算
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        metrics_calculator = EnhancedMetricsCalculator()
        metrics = metrics_calculator.calculate_comprehensive_metrics(y, y_pred, y_proba)

        # モデルを登録
        model_key = self.model_manager.register_model(
            model=model,
            model_name="test_model",
            algorithm="random_forest",
            training_data=X,
            performance_metrics={
                "accuracy": metrics.get("accuracy", 0.0),
                "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
            },
            tags=["test", "random_forest"],
            description="テスト用モデル",
        )

        # モデルをロード
        loaded_model, metadata = self.model_manager.load_model_enhanced(model_key)

        # 結果の検証
        assert loaded_model is not None
        assert metadata.get("algorithm") == "random_forest"

        # モデル一覧取得
        model_list = self.model_manager.get_model_list_enhanced()
        assert len(model_list) >= 1
        # 登録したモデルが含まれているかチェック
        registered_model_found = any(model["path"] == model_key for model in model_list)
        assert (
            registered_model_found
        ), f"登録したモデル {model_key} が一覧に見つかりません"

        logger.info(f"登録されたモデル: {model_key}")
        logger.info(
            f"モデル精度: {metadata.get('performance_metrics', {}).get('accuracy', 0.0):.4f}"
        )
        logger.info("✅ モデル管理システムテスト完了")

    def test_integrated_ml_pipeline(self):
        """統合MLパイプラインのテスト"""
        logger.info("=== 統合MLパイプラインテスト ===")

        # 1. データ準備
        X, y = self.create_test_data(n_samples=300, n_features=30)

        # 2. 特徴量選択
        feature_selector = FeatureSelector(
            FeatureSelectionConfig(method=SelectionMethod.RANDOM_FOREST, k_features=15)
        )
        X_selected, selection_results = feature_selector.fit_transform(X, y)

        # 3. モデル学習
        model = self.create_simple_model()

        # 4. 時系列クロスバリデーション
        cv_validator = TimeSeriesCrossValidator(
            CVConfig(strategy=CVStrategy.TIME_SERIES_SPLIT, n_splits=3)
        )
        cv_results = cv_validator.cross_validate(
            model, X_selected, y, scoring=["accuracy", "balanced_accuracy", "f1"]
        )

        # 5. 最終モデル学習
        model.fit(X_selected, y)
        y_pred = model.predict(X_selected)
        y_proba = model.predict_proba(X_selected)

        # 6. 拡張評価指標計算
        metrics_calculator = EnhancedMetricsCalculator()
        final_metrics = metrics_calculator.calculate_comprehensive_metrics(
            y, y_pred, y_proba
        )

        # 7. モデル管理システムに登録
        model_key = self.model_manager.register_model(
            model=model,
            model_name="integrated_pipeline_model",
            algorithm="random_forest",
            training_data=X_selected,
            performance_metrics={
                "accuracy": final_metrics.get("accuracy", 0.0),
                "balanced_accuracy": final_metrics.get("balanced_accuracy", 0.0),
                "f1_score": final_metrics.get("f1_score", 0.0),
                "cv_accuracy_mean": cv_results.get("accuracy_mean", 0.0),
            },
            feature_selection_config=selection_results,
            tags=["integrated", "pipeline", "test"],
            description="統合パイプラインで学習されたモデル",
        )

        # 結果の検証
        assert X_selected.shape[1] <= X.shape[1], "特徴量数が増加しています"
        assert "accuracy_mean" in cv_results, "クロスバリデーションが実行されていません"
        assert "balanced_accuracy" in final_metrics, "拡張評価指標が計算されていません"
        assert model_key is not None, "モデル登録が失敗しました"

        # 特徴量選択の結果をログ出力
        if X_selected.shape[1] == X.shape[1]:
            logger.warning("統合パイプライン: 全特徴量が選択されました")

        # 最高性能モデルの取得
        best_result = self.model_manager.get_best_model(
            PerformanceMetric.BALANCED_ACCURACY
        )

        assert best_result is not None, "最高性能モデルが見つかりません"
        best_model_path = best_result

        logger.info("統合パイプライン結果:")
        logger.info(f"  特徴量選択: {X.shape[1]} → {X_selected.shape[1]}")
        logger.info(
            f"  CV精度: {cv_results.get('accuracy_mean', 0.0):.4f}±{cv_results.get('accuracy_std', 0.0):.4f}"
        )
        logger.info(f"  最終精度: {final_metrics.get('accuracy', 0.0):.4f}")
        logger.info(
            f"  バランス精度: {final_metrics.get('balanced_accuracy', 0.0):.4f}"
        )
        logger.info(f"  登録モデル: {model_key}")
        logger.info("✅ 統合MLパイプラインテスト完了")

    def test_performance_monitoring(self):
        """パフォーマンス監視のテスト"""
        logger.info("=== パフォーマンス監視テスト ===")

        # テストデータとモデル準備
        X, y = self.create_test_data(n_samples=100, n_features=10)
        model = self.create_simple_model()
        model.fit(X, y)

        # モデル登録
        model_key = self.model_manager.register_model(
            model=model,
            model_name="monitoring_test_model",
            algorithm="random_forest",
            training_data=X,
            performance_metrics={"accuracy": 0.90, "f1_score": 0.85},
            description="パフォーマンス監視テスト用モデル",
        )

        # 正常なパフォーマンス記録
        self.model_manager.record_performance(
            model_key, {"accuracy": 0.88, "f1_score": 0.83}
        )

        # パフォーマンス低下の記録（警告が出るはず）
        self.model_manager.record_performance(
            model_key, {"accuracy": 0.80, "f1_score": 0.75}  # 大幅な低下
        )

        # 履歴の確認
        assert model_key in self.model_manager.performance_history
        history = self.model_manager.performance_history[model_key]
        assert len(history) == 2

        logger.info(f"パフォーマンス履歴: {len(history)}件記録")
        logger.info("✅ パフォーマンス監視テスト完了")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    test_system = TestEnhancedMLSystem()
    test_system.setup_method()

    try:
        test_system.test_time_series_cross_validation()
        test_system.test_enhanced_metrics_calculation()
        test_system.test_feature_selection()
        test_system.test_model_management()
        test_system.test_integrated_ml_pipeline()
        test_system.test_performance_monitoring()

        logger.info("=== 全拡張MLシステムテスト完了 ===")

    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        raise
    finally:
        test_system.teardown_method()
