"""
OOF予測のデータリーク検証テスト

修正後のOOF予測が正しく生成されているか、
データリークが発生していないかを検証します。
"""

import numpy as np
import pandas as pd
import pytest


class TestOOFDataLeakValidation:
    """OOF予測のデータリーク検証"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        # 3クラス分類
        y = pd.Series(np.random.randint(0, 3, n_samples))

        return X, y

    def test_oof_predictions_differ_from_in_fold(self, sample_data):
        """
        OOF予測とIn-Fold予測が異なることを確認

        OOF予測: 各サンプルを学習に使用していないモデルで予測
        In-Fold予測: 全データで学習したモデルで全データを予測

        → 異なるはず（同じ場合はデータリーク）
        """
        from app.services.ml.ensemble.stacking import StackingEnsemble

        X, y = sample_data

        # StackingEnsembleを設定
        config = {
            "base_models": ["lightgbm", "xgboost"],
            "meta_model": "logistic_regression",
            "cv_folds": 5,
            "random_state": 42,
            "n_jobs": 1,
            "cv_strategy": "kfold",  # TimeSeriesSplitではcross_val_predictが使えないためKFoldを使用
        }

        stacking = StackingEnsemble(config=config)

        # モデルを学習
        print("StackingEnsembleを学習中...")
        stacking.fit(X, y)

        # OOF予測を取得
        oof_predictions = stacking.get_oof_predictions()
        assert oof_predictions is not None, "OOF予測が生成されていません"

        # In-Fold予測（全データで学習したモデルで全データを予測）
        in_fold_predictions = stacking.predict_proba(X)[:, 1]

        # 差分を計算
        diff = np.abs(oof_predictions - in_fold_predictions)
        mean_diff = diff.mean()
        max_diff = diff.max()

        print(f"\nOOF vs In-Fold 予測の差分:")
        print(f"  平均差分: {mean_diff:.6f}")
        print(f"  最大差分: {max_diff:.6f}")
        print(f"  標準偏差: {diff.std():.6f}")

        # OOF予測とIn-Fold予測が有意に異なることを確認
        # データリークがなければ、pred > 0.01 程度の差があるはず
        assert mean_diff > 0.005, (
            f"OOF予測とIn-Fold予測の平均差が小さすぎます ({mean_diff:.6f})。"
            "データリークの可能性があります。"
        )

        print(f"✅ OOF予測は正しく生成されています（平均差={mean_diff:.6f} > 0.005）")

    def test_oof_base_model_predictions_differ(self, sample_data):
        """
        各ベースモデルのOOF予測も正しく生成されているか確認
        """
        from app.services.ml.ensemble.stacking import StackingEnsemble

        X, y = sample_data

        config = {
            "base_models": ["lightgbm", "xgboost"],
            "meta_model": "logistic_regression",
            "cv_folds": 5,
            "random_state": 42,
            "n_jobs": 1,
            "cv_strategy": "kfold",
        }

        stacking = StackingEnsemble(config=config)
        stacking.fit(X, y)

        # 各ベースモデルのOOF予測を取得
        oof_base_predictions = stacking.get_oof_base_model_predictions()
        assert (
            oof_base_predictions is not None
        ), "ベースモデルのOOF予測が生成されていません"
        assert len(oof_base_predictions.columns) > 0, "ベースモデルのOOF予測が空です"

        print(f"\nベースモデル数: {len(oof_base_predictions.columns)}")
        print(f"ベースモデル: {list(oof_base_predictions.columns)}")

        # 各ベースモデルについてOOF vs In-Foldを比較
        all_valid = True
        for model_name in oof_base_predictions.columns:
            oof_pred = oof_base_predictions[model_name].values

            # In-Fold予測を取得（学習済みモデルで全データを予測）
            base_model = stacking._fitted_base_models[model_name]
            in_fold_pred = base_model.predict_proba(X)[:, 1]

            diff = np.abs(oof_pred - in_fold_pred).mean()

            print(f"  {model_name}: 平均差={diff:.6f}")

            if diff <= 0.005:
                all_valid = False
                print(f"    ⚠️ 差が小さすぎます（データリークの可能性）")

        assert all_valid, "一部のベースモデルでOOF予測が正しく生成されていません"
        print("✅ 全ベースモデルのOOF予測が正しく生成されています")

    def test_oof_predictions_coverage(self, sample_data):
        """
        OOF予測が全サンプルをカバーしているか確認
        """
        from app.services.ml.ensemble.stacking import StackingEnsemble

        X, y = sample_data

        config = {
            "base_models": ["lightgbm"],
            "meta_model": "logistic_regression",
            "cv_folds": 3,
            "random_state": 42,
            "n_jobs": 1,
            "cv_strategy": "kfold",
        }

        stacking = StackingEnsemble(config=config)
        stacking.fit(X, y)

        oof_predictions = stacking.get_oof_predictions()

        # 全サンプルに対してOOF予測が存在することを確認
        assert len(oof_predictions) == len(X), (
            f"OOF予測のサンプル数({len(oof_predictions)})が"
            f"元データのサンプル数({len(X)})と一致しません"
        )

        # NaNがないことを確認
        nan_count = np.isnan(oof_predictions).sum()
        assert nan_count == 0, f"OOF予測に{nan_count}個のNaNが含まれています"

        print(f"✅ OOF予測が全{len(X)}サンプルをカバーしています（NaNなし）")

    def test_oof_predictions_probability_range(self, sample_data):
        """
        OOF予測が確率の範囲[0, 1]内にあることを確認
        """
        from app.services.ml.ensemble.stacking import StackingEnsemble

        X, y = sample_data

        config = {
            "base_models": ["lightgbm"],
            "meta_model": "logistic_regression",
            "cv_folds": 3,
            "random_state": 42,
            "n_jobs": 1,
            "cv_strategy": "kfold",
        }

        stacking = StackingEnsemble(config=config)
        stacking.fit(X, y)

        oof_predictions = stacking.get_oof_predictions()

        # 確率の範囲チェック
        assert np.all(oof_predictions >= 0), "OOF予測に負の値が含まれています"
        assert np.all(oof_predictions <= 1), "OOF予測に1を超える値が含まれています"

        print(f"✅ OOF予測が確率の範囲[0, 1]内にあります")
        print(f"  最小値: {oof_predictions.min():.6f}")
        print(f"  最大値: {oof_predictions.max():.6f}")
        print(f"  平均値: {oof_predictions.mean():.6f}")


class TestOOFPredictionQuality:
    """OOF予測の品質検証"""

    @pytest.fixture
    def realistic_data(self):
        """より現実的なテストデータ"""
        np.random.seed(42)
        n_samples = 2000
        n_features = 20

        # 特徴量を生成
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # ターゲットを特徴量から生成（現実的な関係性）
        y_continuous = (
            X.iloc[:, 0] * 0.5
            + X.iloc[:, 1] * 0.3
            + X.iloc[:, 2] * 0.2
            + np.random.randn(n_samples) * 0.5
        )

        # 3クラスに分割
        y = pd.qcut(y_continuous, q=3, labels=[0, 1, 2])

        return X, y

    def test_oof_predictions_reasonable_performance(self, realistic_data):
        """
        OOF予測が合理的な性能を示すことを確認

        過学習していない（In-Foldより性能が低い）ことを確認
        """
        from app.services.ml.ensemble.stacking import StackingEnsemble
        from sklearn.metrics import roc_auc_score

        X, y = realistic_data
        y = y.astype(int)

        config = {
            "base_models": ["lightgbm", "xgboost"],
            "meta_model": "logistic_regression",
            "cv_folds": 5,
            "random_state": 42,
            "n_jobs": 1,
            "cv_strategy": "kfold",
        }

        stacking = StackingEnsemble(config=config)
        stacking.fit(X, y)

        # OOF予測でのスコア
        oof_predictions = stacking.get_oof_predictions()

        # In-Fold予測でのスコア
        in_fold_proba = stacking.predict_proba(X)

        # 精度を比較
        # ROC-AUCは2クラス分類用なので、まずバイナリ化
        # 今回は簡易的にクラス1 vs その他で計算
        y_binary = (y == 1).astype(int)

        try:
            print(f"OOF shape: {getattr(oof_predictions, 'shape', 'unknown')}, type: {type(oof_predictions)}")
            oof_auc = roc_auc_score(y_binary, oof_predictions)
            in_fold_auc = roc_auc_score(y_binary, in_fold_proba[:, 1])

            print(f"\nROC-AUC比較:")
            print(f"  OOF予測: {oof_auc:.4f}")
            print(f"  In-Fold予測: {in_fold_auc:.4f}")
            print(f"  差分: {in_fold_auc - oof_auc:.4f}")

            # OOF性能 < In-Fold性能 であるべき（過学習していない証拠）
            # ただし、差が大きすぎる場合は問題
            assert oof_auc < in_fold_auc + 0.01, (
                "OOF予測の性能がIn-Fold予測より高いです。"
                "データリークの可能性があります。"
            )

            assert in_fold_auc - oof_auc < 0.5, (
                f"OOFとIn-Foldの性能差が大きすぎます（{in_fold_auc - oof_auc:.4f}）。"
                "モデルが過学習している可能性があります。"
            )

            print("✅ OOF予測が合理的な性能を示しています")

        except Exception as e:
            print(f"AUC計算でエラー: {e}")
            raise e


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


