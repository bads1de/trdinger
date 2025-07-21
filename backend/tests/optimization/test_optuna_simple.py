"""
OptunaOptimizerの簡単なテスト
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.optimization.optuna_optimizer import (
    OptunaOptimizer,
    ParameterSpace,
)


def test_basic_optuna():
    """基本的なOptunaテスト"""
    print("🚀 Optuna基本テストを開始...")

    optimizer = OptunaOptimizer()

    def objective(params):
        # x=0.5で最大値を取る関数
        return -((params["x"] - 0.5) ** 2)

    parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=1.0)}

    result = optimizer.optimize(objective, parameter_space, n_calls=10)

    print(f"✅ 最適化完了!")
    print(f"   ベストパラメータ: {result.best_params}")
    print(f"   ベストスコア: {result.best_score:.4f}")
    print(f"   評価回数: {result.total_evaluations}")
    print(f"   最適化時間: {result.optimization_time:.2f}秒")

    # 結果の検証
    assert abs(result.best_params["x"] - 0.5) < 0.3
    assert result.best_score > -0.2

    print("🎉 テスト成功!")


def test_default_parameter_space():
    """デフォルトパラメータ空間のテスト"""
    print("\n🔧 デフォルトパラメータ空間テストを開始...")

    space = OptunaOptimizer.get_default_parameter_space()

    print(f"✅ デフォルトパラメータ空間:")
    for param_name, param_config in space.items():
        print(
            f"   {param_name}: {param_config.type} [{param_config.low}, {param_config.high}]"
        )

    # 期待されるパラメータが存在することを確認
    expected_params = ["num_leaves", "learning_rate", "feature_fraction"]
    for param in expected_params:
        assert param in space

    print("🎉 テスト成功!")


if __name__ == "__main__":
    test_basic_optuna()
    test_default_parameter_space()
    print("\n🎊 全てのテストが成功しました!")
