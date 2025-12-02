"""
CatBoost実装の簡易動作確認スクリプト

feature_evaluator.pyのCatBoostサポートが正しく動作するか確認します。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.feature_evaluation.feature_evaluator import (
    FeatureEvaluator,
    FeatureEvaluationConfig,
)
from scripts.feature_evaluation.common_feature_evaluator import (
    CommonFeatureEvaluator,
)


def test_catboost_integration():
    """CatBoost統合テスト"""
    print("=" * 80)
    print("CatBoost実装の動作確認")
    print("=" * 80)

    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 300
    n_features = 15

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(0, 3, n_samples), name="label")

    print(f"\n✓ サンプルデータ生成完了: {n_samples}サンプル, {n_features}特徴量")

    # 評価器の初期化
    config = FeatureEvaluationConfig(
        model="all",
        mode="importance",
        output_dir="backend/tests/temp/catboost_test",
    )
    common_evaluator = CommonFeatureEvaluator()
    evaluator = FeatureEvaluator(common_evaluator, config)

    print("\n✓ 評価器初期化完了")

    # CatBoost重要度計算
    print("\n--- CatBoost重要度計算 ---")
    cb_importance = evaluator._calculate_catboost_importance(X, y)
    print(f"✓ CatBoost重要度: {len(cb_importance)}個の特徴量")
    print(f"  合計: {sum(cb_importance.values()):.6f}")
    print(f"  Top 5:")
    sorted_features = sorted(cb_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_features[:5], 1):
        print(f"    {i}. {feat}: {score:.6f}")

    # CatBoostトレーニング
    print("\n--- CatBoostトレーニング ---")
    train_size = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = evaluator._train_catboost(X_train, y_train)
    print(f"✓ CatBoostモデル学習完了: {train_size}サンプル")

    # 予測
    predictions = model.predict(X_test, prediction_type="Probability")
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = (pred_labels == y_test.values).mean()
    print(f"✓ 予測完了: テスト精度 {accuracy:.4f}")

    # 全モデルでの重要度分析
    print("\n--- 全モデル重要度分析 ---")
    importance_results = evaluator.analyze_importance(X, y, model_type="all")
    print(f"✓ 重要度分析完了:")
    for model_name in importance_results.keys():
        if model_name != "combined":
            print(f"  - {model_name}: {len(importance_results[model_name])}特徴量")

    # CatBoostのみでの分析
    print("\n--- CatBoostのみでの分析 ---")
    cb_only = evaluator.analyze_importance(X, y, model_type="catboost")
    print(f"✓ CatBoostのみの重要度分析完了")
    print(f"  特徴量数: {len(cb_only.get('catboost', {}))}")

    # CV評価
    print("\n--- CatBoost CV評価 ---")
    cv_results = evaluator._evaluate_with_cv(X, y, list(X.columns), "catboost")
    print(f"✓ CV評価完了:")
    print(f"  Accuracy: {cv_results['accuracy']:.4f}")
    print(f"  Precision: {cv_results['precision']:.4f}")
    print(f"  Recall: {cv_results['recall']:.4f}")
    print(f"  F1 Score: {cv_results['f1_score']:.4f}")
    print(f"  Train Time: {cv_results['train_time']:.3f}秒")

    print("\n" + "=" * 80)
    print("✅ 全てのCatBoost機能が正常に動作しました！")
    print("=" * 80)

    common_evaluator.close()


if __name__ == "__main__":
    test_catboost_integration()
