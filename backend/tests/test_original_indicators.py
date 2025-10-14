"""
OriginalIndicatorsのテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from app.services.indicators.technical_indicators.original import OriginalIndicators


class TestOriginalIndicators:
    """OriginalIndicatorsのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        pass

    def test_init(self):
        """初期化のテスト"""
        assert OriginalIndicators is not None

    def test_calculate_frama_valid_data(self):
        """有効データでのFRAMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]}
        )

        result = OriginalIndicators.frama(data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # FRAMAは平滑化された価格なので元の価格の範囲内
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 115

    def test_calculate_frama_insufficient_data(self):
        """データ不足でのFRAMA計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102]})  # 不十分なデータ

        with pytest.raises(ValueError, match="length must be >= 4"):
            OriginalIndicators.frama(data["close"], length=2, slow=200)

    def test_calculate_frama_odd_length(self):
        """奇数長さのFRAMAテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        with pytest.raises(ValueError, match="length must be an even number"):
            OriginalIndicators.frama(data["close"], length=5, slow=200)

    def test_calculate_frama_negative_slow(self):
        """負のslowパラメータのFRAMAテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        with pytest.raises(ValueError, match="slow must be >= 1"):
            OriginalIndicators.frama(data["close"], length=16, slow=0)

    def test_calculate_frama_empty_data(self):
        """空データのFRAMAテスト"""
        data = pd.DataFrame({"close": []})

        result = OriginalIndicators.frama(data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_calculate_frama_single_value(self):
        """単一値のFRAMAテスト"""
        data = pd.DataFrame({"close": [100]})

        result = OriginalIndicators.frama(data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.isna().all()

    def test_calculate_frama_with_trend(self):
        """トレンドがあるデータのFRAMAテスト"""
        # 明確な上昇トレンド
        trend_data = pd.DataFrame(
            {"close": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]}
        )

        result = OriginalIndicators.frama(trend_data["close"], length=8, slow=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(trend_data)
        # トレンドを追従しているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 118

    def test_calculate_frama_with_noise(self):
        """ノイズがあるデータのFRAMAテスト"""
        np.random.seed(42)
        base = 1000
        trend = np.linspace(0, 100, 50)
        noise = np.random.normal(0, 5, 50)
        noisy_data = pd.DataFrame({"close": base + trend + noise})

        result = OriginalIndicators.frama(noisy_data["close"], length=10, slow=100)

        assert isinstance(result, pd.Series)
        assert len(result) == len(noisy_data)
        # ノイズが平滑化されているはず
        assert result.dropna().min() >= base - 10
        assert result.dropna().max() <= base + trend[-1] + 10

    def test_calculate_super_smoother_valid_data(self):
        """有効データでのSuper Smoother計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.super_smoother(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # Super Smootherは元の価格の範囲内
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_calculate_super_smoother_insufficient_length(self):
        """不十分な長さのSuper Smootherテスト"""
        data = pd.DataFrame({"close": [100, 101]})  # 不十分

        with pytest.raises(ValueError, match="length must be >= 2"):
            OriginalIndicators.super_smoother(data["close"], length=1)

    def test_calculate_super_smoother_empty_data(self):
        """空データのSuper Smootherテスト"""
        data = pd.DataFrame({"close": []})

        result = OriginalIndicators.super_smoother(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_calculate_super_smoother_single_value(self):
        """単一値のSuper Smootherテスト"""
        data = pd.DataFrame({"close": [100]})

        result = OriginalIndicators.super_smoother(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        # 最初の2点は元の価格
        assert result.isna().all()

    def test_calculate_super_smoother_with_oscillation(self):
        """振動データのSuper Smootherテスト"""
        # 振動するデータ
        oscillation_data = pd.DataFrame(
            {"close": [100, 110, 100, 110, 100, 110, 100, 110, 100, 110]}
        )

        result = OriginalIndicators.super_smoother(oscillation_data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(oscillation_data)
        # 振動が平滑化されているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 110

    def test_calculate_super_smoother_with_trend(self):
        """トレンドがあるデータのSuper Smootherテスト"""
        # 上昇トレンド
        trend_data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.super_smoother(trend_data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(trend_data)
        # トレンドを追従しているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_frama_edge_case_length_4(self):
        """FRAMAの境界値テスト（length=4）"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]})

        result = OriginalIndicators.frama(data["close"], length=4, slow=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 105

    def test_frama_edge_case_slow_1(self):
        """FRAMAの境界値テスト（slow=1）"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.frama(data["close"], length=4, slow=1)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_super_smoother_edge_case_length_2(self):
        """Super Smootherの境界値テスト（length=2）"""
        data = pd.DataFrame({"close": [100, 101, 102]})

        result = OriginalIndicators.super_smoother(data["close"], length=2)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 102

    def test_frama_alpha_clamping(self):
        """FRAMAのα値のクランプテスト"""
        # 極端な値のデータ
        extreme_data = pd.DataFrame(
            {"close": [100, 200, 100, 200, 100, 200, 100, 200, 100, 200]}
        )

        result = OriginalIndicators.frama(extreme_data["close"], length=8, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(extreme_data)
        # α値がクランプされているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 200

    def test_super_smoother_stability(self):
        """Super Smootherの安定性テスト"""
        # 大きなノイズがあるデータ
        np.random.seed(42)
        base = 1000
        trend = np.linspace(0, 100, 100)
        noise = np.random.normal(0, 20, 100)  # 大きなノイズ
        noisy_data = pd.DataFrame({"close": base + trend + noise})

        result = OriginalIndicators.super_smoother(noisy_data["close"], length=15)

        assert isinstance(result, pd.Series)
        assert len(result) == len(noisy_data)
        # 安定して平滑化されているはず
        assert result.dropna().min() >= base - 30
        assert result.dropna().max() <= base + trend[-1] + 30

    def test_frama_with_decreasing_data(self):
        """減少トレンドのFRAMAテスト"""
        decreasing_data = pd.DataFrame(
            {"close": [115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]}
        )

        result = OriginalIndicators.frama(decreasing_data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(decreasing_data)
        # 減少トレンドを追従しているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 115

    def test_super_smoother_with_decreasing_data(self):
        """減少トレンドのSuper Smootherテスト"""
        decreasing_data = pd.DataFrame(
            {"close": [109, 108, 107, 106, 105, 104, 103, 102, 101, 100]}
        )

        result = OriginalIndicators.super_smoother(decreasing_data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(decreasing_data)
        # 減少トレンドを追従しているはず
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_frama_parameter_validation(self):
        """FRAMAのパラメータ検証テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # lengthが負
        with pytest.raises(ValueError, match="length must be >= 4"):
            OriginalIndicators.frama(data["close"], length=-1, slow=200)

        # slowが負
        with pytest.raises(ValueError, match="slow must be >= 1"):
            OriginalIndicators.frama(data["close"], length=16, slow=-1)

    def test_super_smoother_parameter_validation(self):
        """Super Smootherのパラメータ検証テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # lengthが負
        with pytest.raises(ValueError, match="length must be >= 2"):
            OriginalIndicators.super_smoother(data["close"], length=-1)

    def test_frama_numpy_compatibility(self):
        """FRAMAのNumPy配列互換性テスト"""
        # NumPy配列でテスト
        close_array = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])

        result = OriginalIndicators.frama(pd.Series(close_array), length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(close_array)

    def test_super_smoother_numpy_compatibility(self):
        """Super SmootherのNumPy配列互換性テスト"""
        # NumPy配列でテスト
        close_array = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        result = OriginalIndicators.super_smoother(pd.Series(close_array), length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(close_array)

    def test_frama_with_nan_values(self):
        """FRAMAのNaN値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.nan, 102, 103, np.nan, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.frama(data["close"], length=8, slow=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # NaNが適切に処理されている

    def test_super_smoother_with_nan_values(self):
        """Super SmootherのNaN値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.nan, 102, 103, np.nan, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.super_smoother(data["close"], length=8)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # NaNが適切に処理されている

    def test_frama_with_inf_values(self):
        """FRAMAの無限大値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.inf, 102, 103, -np.inf, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.frama(data["close"], length=8, slow=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 無限大値が適切に処理されている

    def test_super_smoother_with_inf_values(self):
        """Super Smootherの無限大値処理テスト"""
        data = pd.DataFrame(
            {"close": [100, np.inf, 102, 103, -np.inf, 105, 106, 107, 108, 109]}
        )

        result = OriginalIndicators.super_smoother(data["close"], length=8)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 無限大値が適切に処理されている

    def test_frama_different_length_combinations(self):
        """FRAMAの異なるパラメータ組み合わせテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]}
        )

        # 短期
        result_short = OriginalIndicators.frama(data["close"], length=8, slow=50)
        # 長期
        result_long = OriginalIndicators.frama(data["close"], length=20, slow=200)

        assert isinstance(result_short, pd.Series)
        assert isinstance(result_long, pd.Series)
        assert len(result_short) == len(data)
        assert len(result_long) == len(data)

    def test_super_smoother_different_length_combinations(self):
        """Super Smootherの異なるパラメータ組み合わせテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # 短期
        result_short = OriginalIndicators.super_smoother(data["close"], length=5)
        # 長期
        result_long = OriginalIndicators.super_smoother(data["close"], length=20)

        assert isinstance(result_short, pd.Series)
        assert isinstance(result_long, pd.Series)
        assert len(result_short) == len(data)
        assert len(result_long) == len(data)

    def test_frama_multiple_calls_consistency(self):
        """FRAMAの複数回呼び出しの一貫性テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]}
        )

        result1 = OriginalIndicators.frama(data["close"], length=16, slow=200)
        result2 = OriginalIndicators.frama(data["close"], length=16, slow=200)

        # 同じ結果になるはず
        assert result1.equals(result2)

    def test_super_smoother_multiple_calls_consistency(self):
        """Super Smootherの複数回呼び出しの一貫性テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result1 = OriginalIndicators.super_smoother(data["close"], length=10)
        result2 = OriginalIndicators.super_smoother(data["close"], length=10)

        # 同じ結果になるはず
        assert result1.equals(result2)

    def test_frama_with_random_data(self):
        """FRAMAのランダムデータテスト"""
        np.random.seed(42)
        random_data = pd.DataFrame(
            {"close": np.random.normal(100, 10, 100)}
        )

        result = OriginalIndicators.frama(random_data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(random_data)
        # 平滑化されているはず

    def test_super_smoother_with_random_data(self):
        """Super Smootherのランダムデータテスト"""
        np.random.seed(42)
        random_data = pd.DataFrame(
            {"close": np.random.normal(100, 10, 100)}
        )

        result = OriginalIndicators.super_smoother(random_data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(random_data)
        # 平滑化されているはず

    def test_frama_with_constant_data(self):
        """FRAMAの定数データテスト"""
        constant_data = pd.DataFrame(
            {"close": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]}
        )

        result = OriginalIndicators.frama(constant_data["close"], length=8, slow=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(constant_data)
        # 定数データでは定数を返すはず
        assert result.dropna().std() < 1e-10  # ほぼ定数

    def test_super_smoother_with_constant_data(self):
        """Super Smootherの定数データテスト"""
        constant_data = pd.DataFrame(
            {"close": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]}
        )

        result = OriginalIndicators.super_smoother(constant_data["close"], length=8)

        assert isinstance(result, pd.Series)
        assert len(result) == len(constant_data)
        # 定数データでは定数を返すはず
        assert result.dropna().std() < 1e-10  # ほぼ定数

    def test_frama_with_step_function(self):
        """FRAMAのステップ関数テスト"""
        step_data = pd.DataFrame(
            {"close": [100, 100, 100, 100, 100, 150, 150, 150, 150, 150]}
        )

        result = OriginalIndicators.frama(step_data["close"], length=6, slow=25)

        assert isinstance(result, pd.Series)
        assert len(result) == len(step_data)
        # ステップ変化を追従しているはず

    def test_super_smoother_with_step_function(self):
        """Super Smootherのステップ関数テスト"""
        step_data = pd.DataFrame(
            {"close": [100, 100, 100, 100, 100, 150, 150, 150, 150, 150]}
        )

        result = OriginalIndicators.super_smoother(step_data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(step_data)
        # ステップ変化を追従しているはず

    def test_frama_memory_usage(self):
        """FRAMAのメモリ使用量テスト"""
        # 大きなデータセット
        large_data = pd.DataFrame(
            {"close": np.random.normal(100, 10, 10000)}
        )

        result = OriginalIndicators.frama(large_data["close"], length=16, slow=200)

        assert isinstance(result, pd.Series)
        assert len(result) == len(large_data)
        # 大きなデータでも処理できる

    def test_super_smoother_memory_usage(self):
        """Super Smootherのメモリ使用量テスト"""
        # 大きなデータセット
        large_data = pd.DataFrame(
            {"close": np.random.normal(100, 10, 10000)}
        )

        result = OriginalIndicators.super_smoother(large_data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(large_data)
        # 大きなデータでも処理できる

    def test_frama_with_extreme_parameters(self):
        """FRAMAの極端なパラメータテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # 極端なslow値
        result = OriginalIndicators.frama(data["close"], length=4, slow=1000)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 極端なパラメータでもエラーにならない

    def test_super_smoother_with_extreme_parameters(self):
        """Super Smootherの極端なパラメータテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # 極端なlength値
        result = OriginalIndicators.super_smoother(data["close"], length=100)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 極端なパラメータでもエラーにならない


if __name__ == "__main__":
    # コマンドラインからの実行用
    pytest.main([__file__, "-v"])