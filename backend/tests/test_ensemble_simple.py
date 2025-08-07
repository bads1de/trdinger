"""
scikit-learnベースのアンサンブル実装の簡単な動作確認

BaggingClassifierとStackingClassifierの基本動作をテストします。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from app.services.ml.ensemble.bagging import BaggingEnsemble
from app.services.ml.ensemble.stacking import StackingEnsemble


def create_sample_data():
    """テスト用のサンプルデータを生成"""
    print("📊 サンプルデータを生成中...")

    # 3クラス分類問題を作成
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    # DataFrameとSeriesに変換
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    # 訓練・テストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42, stratify=y
    )

    print(f"✅ データ生成完了: 訓練データ{len(X_train)}件, テストデータ{len(X_test)}件")
    print(f"   特徴量数: {X_train.shape[1]}, クラス数: {len(np.unique(y))}")

    return X_train, X_test, y_train, y_test


def test_bagging_ensemble():
    """BaggingEnsembleのテスト"""
    print("\n🎯 BaggingEnsemble テスト開始")

    try:
        X_train, X_test, y_train, y_test = create_sample_data()

        # BaggingEnsemble設定
        config = {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "random_forest",
            "random_state": 42,
            "n_jobs": 1,  # テスト用に1に設定
        }

        print(f"⚙️  設定: {config}")

        # アンサンブル作成
        ensemble = BaggingEnsemble(config)
        print(f"✅ BaggingEnsemble初期化完了")

        # 学習
        print("🔄 学習開始...")
        result = ensemble.fit(X_train, y_train, X_test, y_test)
        print(f"✅ 学習完了")

        # 結果確認
        print(f"📈 学習結果:")
        print(f"   モデルタイプ: {result.get('model_type', 'N/A')}")
        print(
            f"   精度: {result.get('accuracy', 'N/A'):.4f}"
            if "accuracy" in result
            else "   精度: N/A"
        )
        print(f"   sklearn実装: {result.get('sklearn_implementation', 'N/A')}")

        # 予測
        print("🔮 予測実行...")
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)

        print(f"✅ 予測完了")
        print(f"   予測結果形状: {y_pred.shape}")
        print(f"   予測確率形状: {y_pred_proba.shape}")
        print(f"   確率合計チェック: {np.allclose(y_pred_proba.sum(axis=1), 1.0)}")

        # 特徴量重要度
        importance = ensemble.get_feature_importance()
        if importance:
            print(f"📊 特徴量重要度取得成功: {len(importance)}個の特徴量")
        else:
            print("📊 特徴量重要度: 取得できませんでした")

        print("🎉 BaggingEnsemble テスト成功!")
        return True

    except Exception as e:
        print(f"❌ BaggingEnsemble テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_stacking_ensemble():
    """StackingEnsembleのテスト"""
    print("\n🎯 StackingEnsemble テスト開始")

    try:
        X_train, X_test, y_train, y_test = create_sample_data()

        # StackingEnsemble設定
        config = {
            "base_models": ["random_forest", "gradient_boosting"],
            "meta_model": "logistic_regression",
            "cv_folds": 3,
            "stack_method": "predict_proba",
            "random_state": 42,
            "n_jobs": 1,  # テスト用に1に設定
        }

        print(f"⚙️  設定: {config}")

        # アンサンブル作成
        ensemble = StackingEnsemble(config)
        print(f"✅ StackingEnsemble初期化完了")

        # 学習
        print("🔄 学習開始...")
        result = ensemble.fit(X_train, y_train, X_test, y_test)
        print(f"✅ 学習完了")

        # 結果確認
        print(f"📈 学習結果:")
        print(f"   モデルタイプ: {result.get('model_type', 'N/A')}")
        print(
            f"   精度: {result.get('accuracy', 'N/A'):.4f}"
            if "accuracy" in result
            else "   精度: N/A"
        )
        print(f"   sklearn実装: {result.get('sklearn_implementation', 'N/A')}")

        # 予測
        print("🔮 予測実行...")
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)

        print(f"✅ 予測完了")
        print(f"   予測結果形状: {y_pred.shape}")
        print(f"   予測確率形状: {y_pred_proba.shape}")
        print(f"   確率合計チェック: {np.allclose(y_pred_proba.sum(axis=1), 1.0)}")

        # 特徴量重要度
        importance = ensemble.get_feature_importance()
        if importance:
            print(f"📊 特徴量重要度取得成功: {len(importance)}個のベースモデル")
            for model_name, imp in importance.items():
                print(f"     {model_name}: {imp:.4f}")
        else:
            print("📊 特徴量重要度: 取得できませんでした")

        print("🎉 StackingEnsemble テスト成功!")
        return True

    except Exception as e:
        print(f"❌ StackingEnsemble テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_save_load_models():
    """モデル保存・読み込みテスト"""
    print("\n🎯 モデル保存・読み込み テスト開始")

    try:
        X_train, X_test, y_train, y_test = create_sample_data()

        # BaggingEnsemble保存・読み込みテスト
        config = {
            "n_estimators": 2,
            "bootstrap_fraction": 0.8,
            "base_model_type": "random_forest",
            "random_state": 42,
        }

        ensemble = BaggingEnsemble(config)
        ensemble.fit(X_train, y_train)

        # 保存
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, "test_model")
            saved_paths = ensemble.save_models(base_path)
            print(f"✅ モデル保存完了: {len(saved_paths)}個のファイル")

            # 新しいインスタンスで読み込み
            new_ensemble = BaggingEnsemble(config)
            success = new_ensemble.load_models(base_path)

            if success:
                print("✅ モデル読み込み成功")

                # 予測テスト
                y_pred_original = ensemble.predict(X_test[:5])
                y_pred_loaded = new_ensemble.predict(X_test[:5])

                if np.array_equal(y_pred_original, y_pred_loaded):
                    print("✅ 読み込み後の予測結果が一致")
                    return True
                else:
                    print("❌ 読み込み後の予測結果が不一致")
                    return False
            else:
                print("❌ モデル読み込み失敗")
                return False

    except Exception as e:
        print(f"❌ モデル保存・読み込みテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("🚀 scikit-learnベースアンサンブル実装テスト開始")
    print("=" * 60)

    results = []

    # BaggingEnsembleテスト
    results.append(test_bagging_ensemble())

    # StackingEnsembleテスト
    results.append(test_stacking_ensemble())

    # モデル保存・読み込みテスト
    results.append(test_save_load_models())

    # 結果サマリー
    print("\n" + "=" * 60)
    print("📋 テスト結果サマリー")
    print(f"BaggingEnsemble: {'✅ 成功' if results[0] else '❌ 失敗'}")
    print(f"StackingEnsemble: {'✅ 成功' if results[1] else '❌ 失敗'}")
    print(f"モデル保存・読み込み: {'✅ 成功' if results[2] else '❌ 失敗'}")

    if all(results):
        print("\n🎉 全テスト成功! scikit-learnベースの実装が正常に動作しています。")
        print("\n📈 移行のメリット:")
        print("   ✅ 正確なアンサンブル手法の実装")
        print("   ✅ 最適化されたパフォーマンス")
        print("   ✅ 自動並列処理サポート")
        print("   ✅ 大幅なコード削減")
        print("   ✅ 標準ライブラリの安定性")
        print("   ✅ モデル保存・読み込み機能")
    else:
        print("\n❌ 一部のテストが失敗しました。")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
