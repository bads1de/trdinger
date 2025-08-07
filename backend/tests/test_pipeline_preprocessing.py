"""
Pipeline前処理の修正テスト

3.6の問題修正が正しく動作することを確認するテストファイル
"""

import numpy as np
import pandas as pd
import sys
import os

# パスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


def test_pipeline_creation():
    """パイプライン作成テスト"""
    print("=== パイプライン作成テスト ===")

    try:
        from utils.data_processing import DataProcessor

        processor = DataProcessor()
        print("✅ DataProcessor インスタンス作成成功")

        # パイプライン作成
        pipeline = processor.create_preprocessing_pipeline(
            numeric_strategy="median",
            scaling_method="robust",
            remove_outliers=True,
            outlier_method="iqr",
        )

        print("✅ 前処理パイプライン作成成功")
        print(f"   パイプラインステップ数: {len(pipeline.steps)}")

        # パイプラインの構造確認
        for i, (name, step) in enumerate(pipeline.steps):
            print(f"   ステップ{i+1}: {name} - {type(step).__name__}")

        return True

    except Exception as e:
        print(f"❌ パイプライン作成テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pipeline_preprocessing():
    """パイプライン前処理テスト"""
    print("\n=== パイプライン前処理テスト ===")

    try:
        from utils.data_processing import DataProcessor

        processor = DataProcessor()

        # テストデータ作成
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="h")

        test_data = pd.DataFrame(
            {
                "Open": 50000 + np.random.randn(100) * 1000,
                "High": 50000 + np.random.randn(100) * 1000 + 500,
                "Low": 50000 + np.random.randn(100) * 1000 - 500,
                "Close": 50000 + np.random.randn(100) * 1000,
                "Volume": np.random.randint(1000, 10000, 100),
                "Category": np.random.choice(["A", "B", "C"], 100),
                "Text_Feature": np.random.choice(["Type1", "Type2", "Type3"], 100),
            },
            index=dates,
        )

        # 一部にNaNと外れ値を追加
        test_data.loc[test_data.index[10:15], "Open"] = np.nan
        test_data.loc[test_data.index[20:25], "Category"] = np.nan
        test_data.loc[test_data.index[5], "Close"] = 1000000  # 外れ値

        print("✅ テストデータ作成完了")
        print(f"   データサイズ: {len(test_data)}行, {len(test_data.columns)}列")
        print(f"   欠損値数: {test_data.isnull().sum().sum()}")

        # Pipeline前処理実行
        result = processor.preprocess_with_pipeline(
            test_data,
            pipeline_name="test_pipeline",
            numeric_strategy="median",
            scaling_method="robust",
            remove_outliers=True,
        )

        print("✅ Pipeline前処理実行成功")
        print(f"   結果サイズ: {len(result)}行, {len(result.columns)}列")
        print(f"   残り欠損値数: {result.isnull().sum().sum()}")

        # 結果の検証
        assert len(result) == len(test_data), "行数が一致しません"
        assert result.isnull().sum().sum() == 0, "欠損値が残っています"

        print("✅ 結果検証成功")

        return True

    except Exception as e:
        print(f"❌ パイプライン前処理テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ml_pipeline():
    """ML用パイプラインテスト"""
    print("\n=== ML用パイプラインテスト ===")

    try:
        from utils.data_processing import DataProcessor

        processor = DataProcessor()

        # ML用パイプライン作成
        ml_pipeline = processor.create_ml_preprocessing_pipeline(
            target_column="Close", feature_selection=True, n_features=5
        )

        print("✅ ML用パイプライン作成成功")
        print(f"   パイプラインステップ数: {len(ml_pipeline.steps)}")

        # テストデータ作成
        np.random.seed(42)
        test_data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "feature4": np.random.randn(100),
                "feature5": np.random.randn(100),
                "feature6": np.random.randn(100),
                "feature7": np.random.randn(100),
                "Close": np.random.randn(100),
            }
        )

        # パイプラインをfitしてtransform（特徴選択のためにターゲット変数を分離）
        X = test_data.drop("Close", axis=1)
        y = test_data["Close"]

        fitted_pipeline = ml_pipeline.fit(X, y)
        result = fitted_pipeline.transform(X)

        print("✅ ML用パイプライン実行成功")
        print(f"   入力特徴数: {X.shape[1]}")
        print(f"   出力特徴数: {result.shape[1]}")

        # 特徴選択が動作していることを確認
        assert result.shape[1] <= X.shape[1], "特徴選択が動作していません"

        print("✅ ML用パイプライン検証成功")

        return True

    except Exception as e:
        print(f"❌ ML用パイプラインテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pipeline_caching():
    """パイプラインキャッシュテスト"""
    print("\n=== パイプラインキャッシュテスト ===")

    try:
        from utils.data_processing import DataProcessor

        processor = DataProcessor()

        # テストデータ作成
        test_data = pd.DataFrame(
            {
                "A": np.random.randn(50),
                "B": np.random.randn(50),
                "C": np.random.choice(["X", "Y"], 50),
            }
        )

        # 初回実行（fit=True）
        result1 = processor.preprocess_with_pipeline(
            test_data, pipeline_name="cache_test", fit_pipeline=True
        )

        # パイプライン情報取得
        info = processor.get_pipeline_info("cache_test")
        print("✅ パイプライン情報取得成功")
        print(f"   存在: {info['exists']}")
        print(f"   ステップ: {info['steps']}")

        # 2回目実行（fit=False、キャッシュ使用）
        result2 = processor.preprocess_with_pipeline(
            test_data, pipeline_name="cache_test", fit_pipeline=False
        )

        print("✅ キャッシュされたパイプライン使用成功")

        # 結果が同じことを確認
        np.testing.assert_array_almost_equal(result1.values, result2.values)
        print("✅ キャッシュ結果の一致確認成功")

        return True

    except Exception as e:
        print(f"❌ パイプラインキャッシュテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n=== パフォーマンス比較テスト ===")

    try:
        from utils.data_processing import DataProcessor
        import time

        processor = DataProcessor()

        # 大きなテストデータ作成
        np.random.seed(42)
        large_data = pd.DataFrame(
            {
                "num1": np.random.randn(1000),
                "num2": np.random.randn(1000),
                "num3": np.random.randn(1000),
                "cat1": np.random.choice(["A", "B", "C", "D"], 1000),
                "cat2": np.random.choice(["X", "Y", "Z"], 1000),
            }
        )

        # 従来の方法
        start_time = time.time()
        result_old = processor.preprocess_features(
            large_data,
            imputation_strategy="median",
            scale_features=True,
            remove_outliers=True,
        )
        old_time = time.time() - start_time

        # Pipeline方法
        start_time = time.time()
        result_new = processor.preprocess_with_pipeline(
            large_data,
            pipeline_name="performance_test",
            numeric_strategy="median",
            scaling_method="robust",
            remove_outliers=True,
        )
        new_time = time.time() - start_time

        print(f"✅ パフォーマンス比較完了")
        print(f"   従来の方法: {old_time:.3f}秒")
        print(f"   Pipeline方法: {new_time:.3f}秒")
        print(f"   速度比: {old_time/new_time:.2f}倍")

        # 結果の形状確認
        print(f"   従来結果: {result_old.shape}")
        print(f"   Pipeline結果: {result_new.shape}")

        return True

    except Exception as e:
        print(f"❌ パフォーマンス比較テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("Pipeline前処理修正テスト開始\n")

    # テスト実行
    test1_result = test_pipeline_creation()
    test2_result = test_pipeline_preprocessing()
    test3_result = test_ml_pipeline()
    test4_result = test_pipeline_caching()
    test5_result = test_performance_comparison()

    # 結果サマリー
    print("\n" + "=" * 50)
    print("=== 最終テスト結果サマリー ===")
    print(f"パイプライン作成: {'✅ 成功' if test1_result else '❌ 失敗'}")
    print(f"パイプライン前処理: {'✅ 成功' if test2_result else '❌ 失敗'}")
    print(f"ML用パイプライン: {'✅ 成功' if test3_result else '❌ 失敗'}")
    print(f"パイプラインキャッシュ: {'✅ 成功' if test4_result else '❌ 失敗'}")
    print(f"パフォーマンス比較: {'✅ 成功' if test5_result else '❌ 失敗'}")

    all_success = all(
        [test1_result, test2_result, test3_result, test4_result, test5_result]
    )

    if all_success:
        print("\n🎉 すべてのPipeline前処理テストが成功しました！")
        print("3.6の問題修正が正常に動作しています。")
        print("\n改善点:")
        print("- 独立した前処理関数をPipelineで統合")
        print("- 処理順序の明確化")
        print("- カラム管理の簡素化")
        print("- 宣言的で見通しの良い実装")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")

    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
