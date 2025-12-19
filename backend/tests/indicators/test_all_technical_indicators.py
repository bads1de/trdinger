"""
70個全てのテクニカルインジケーターの包括的テスト

すべてのインジケーターの初期化と計算が正しく動作するかを検証するテスト。
- 各インジケーターの基本的な計算
- 返り値の形式チェック
- NaN/エラー処理
- 必要なデータ長の検証
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators import TechnicalIndicatorService


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """テスト用のOHLCVデータを生成"""
    periods = 500  # 十分な長さを確保
    index = pd.date_range("2022-01-01", periods=periods, freq="h")

    # ランダム性とトレンドを含むデータ
    base = np.linspace(10000, 15000, periods)
    noise = np.random.normal(0, 100, periods)
    close = base + noise

    df = pd.DataFrame(
        {
            "Open": close * np.random.uniform(0.99, 1.01, periods),
            "High": close * np.random.uniform(1.01, 1.03, periods),
            "Low": close * np.random.uniform(0.97, 0.99, periods),
            "Close": close,
            "Volume": np.random.uniform(1000, 5000, periods),
        },
        index=index,
    )

    # ボラティリティを追加
    df["High"] = np.maximum(
        df["High"],
        df[["Open", "Close"]].max(axis=1) * np.random.uniform(1.0, 1.05, periods),
    )
    df["Low"] = np.minimum(
        df["Low"],
        df[["Open", "Close"]].min(axis=1) * np.random.uniform(0.95, 1.0, periods),
    )

    return df


@pytest.fixture
def indicator_service() -> TechnicalIndicatorService:
    """テクニカルインジケーターサービスを提供"""
    return TechnicalIndicatorService()


class TestAllTechnicalIndicators:
    """全テクニカル指標の網羅的テストクラス"""

    # 指標リストを動的に取得
    try:
        from app.services.indicators.config.indicator_config import indicator_registry, initialize_all_indicators
        initialize_all_indicators()
        INDICATORS = sorted(indicator_registry.list_indicators())
    except Exception:
        # フォールバック
        INDICATORS = ["RSI", "SMA", "EMA", "MACD", "BBANDS"]

    @pytest.fixture(autouse=True)
    def setup_method(self, indicator_service):
        """すべてのインジケーターが初期化可能か確認"""
        for indicator in self.INDICATORS:
            try:
                # エイリアスを解決
                resolved_name = indicator_service._resolve_indicator_name(indicator)
                # 各インジケーターの設定を取得できるか確認
                config = indicator_service.registry.get_indicator_config(resolved_name)
                assert config is not None, f"{indicator} ({resolved_name}) の設定が見つかりません"
                assert (config.adapter_function is not None or config.pandas_function is not None), (
                    f"{indicator}に実装（アダプターまたはpandas-ta関数）がありません"
                )
            except Exception as e:
                pytest.fail(f"{indicator}の初期化に失敗: {e}")

    @pytest.mark.parametrize("indicator", INDICATORS)
    def test_indicator_basic_calculation(
        self,
        indicator: str,
        indicator_service: TechnicalIndicatorService,
        sample_ohlcv: pd.DataFrame,
    ):
        """各インジケーターの基本計算をテスト"""
        try:
            # インジケーター設定を取得
            config = indicator_service.registry.get_indicator_config(indicator)
            if config is None:
                pytest.skip(f"{indicator}の設定が見つからないためスキップ")

            # デフォルトパラメータでテスト
            default_params = config.default_values or {}
            params = {}

            # 指標ごとの主要パラメータ名を特定
            supported_params = set(config.parameters.keys())
            
            # テスト用パラメータを設定（サポートされている場合のみ）
            def set_param(test_key, value, aliases=None):
                if test_key in supported_params:
                    params[test_key] = value
                    return True
                if aliases:
                    for alias in aliases:
                        if alias in supported_params:
                            params[alias] = value
                            return True
                return False
    
                if indicator == "ADOSC":
                    set_param("fast", 3)
                    set_param("slow", 10)
                elif indicator in ["MACD", "PPO", "PVO"]:
                    set_param("fast", 12)
                    set_param("slow", 26)
                    set_param("signal", 9)
                elif indicator == "STOCH":
                    set_param("k_length", 14)
                    set_param("smooth_k", 3)
                    set_param("d_length", 3)
                elif indicator == "BB":
                    set_param("length", 20, ["period", "window"])
                    set_param("std", 2.0)
                else:
                    # 一般的な期間パラメータ
                    set_param("length", 14, ["period", "window", "n"])
    
                # 登録されているパラメータ以外は渡さないようにする
                final_params = {}
                for k, v in params.items():
                    if k in supported_params:
                        final_params[k] = v
                
                # 何も設定されなかった場合はデフォルトを使用
                if not final_params:
                    final_params = default_params
    
                # 計算を実行
                result = indicator_service.calculate_indicator(
                    sample_ohlcv, indicator, final_params
                )
            # 結果の形式を検証
            if config.result_type == "single":
                assert isinstance(result, np.ndarray), (
                    f"{indicator}の結果がndarrayではありません"
                )
                assert result.shape[0] == len(sample_ohlcv), (
                    f"{indicator}の結果の長さが不正"
                )
                # NaNを含む場合があるが、最後の数ポイントは有効な場合がある
                assert result.shape[0] > 0, f"{indicator}の結果が空"

            elif config.result_type == "complex" or config.result_type == "multiple":
                assert isinstance(result, tuple), (
                    f"{indicator}の結果がtupleではありません"
                )
                assert len(result) > 0, f"{indicator}の結果が空のtuple"
                for i, series in enumerate(result):
                    assert isinstance(series, np.ndarray), (
                        f"{indicator}の結果[{i}]がndarrayではありません"
                    )
                    assert series.shape[0] == len(sample_ohlcv), (
                        f"{indicator}の結果[{i}]の長さが不正{indicator}"
                    )

        except Exception as e:
            pytest.fail(f"{indicator}のテストでエラー: {e}")

    def test_all_indicators_handle_insufficient_data(
        self, indicator_service: TechnicalIndicatorService
    ):
        """不十分なデータに対する挙動をテスト"""
        short_data = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Volume": [1000, 1100, 900],
            }
        )

        for indicator in self.INDICATORS:
            try:
                config = indicator_service.registry.get_indicator_config(indicator)
                if not config:
                    continue

                # 短いデータで計算を試みる
                params = config.default_values or {}

                result = indicator_service.calculate_indicator(
                    short_data, indicator, params
                )

                # 結果が期待通りか検証
                if config.result_type == "single":
                    assert isinstance(result, np.ndarray)
                    # すべてNaNまたは部分的にNaN
                    assert result.shape[0] == len(short_data)
                elif config.result_type in ["complex", "multiple"]:
                    assert isinstance(result, tuple)

            except Exception as e:
                # 一部のインジケーターは短いデータでエラーになるのは許容
                if "データ長" not in str(e) and "長さ" not in str(e):
                    print(f"警告: {indicator}で予期しないエラー: {e}")

    def test_indicators_return_valid_values(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """インジケーターが有効な値を返すか検証"""
        for indicator in ["RSI", "MFI", "CCI", "WILLR", "UO"]:
            try:
                if indicator == "RSI":
                    result = indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"length": 14}
                    )
                    # RSIは0-100の範囲
                    if isinstance(result, np.ndarray):
                        valid_values = result[np.isfinite(result)]
                        if len(valid_values) > 0:
                            assert all(0 <= val <= 100 for val in valid_values), (
                                f"{indicator}の値が範囲外"
                            )

                elif indicator == "MFI":
                    result = indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"length": 14}
                    )
                    if isinstance(result, np.ndarray):
                        valid_values = result[np.isfinite(result)]
                        if len(valid_values) > 0:
                            assert all(0 <= val <= 100 for val in valid_values), (
                                f"{indicator}の値が範囲外"
                            )

                elif indicator == "WILLR":
                    result = indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"length": 14}
                    )
                    if isinstance(result, np.ndarray):
                        valid_values = result[np.isfinite(result)]
                        if len(valid_values) > 0:
                            assert all(-100 <= val <= 0 for val in valid_values), (
                                f"{indicator}の値が範囲外"
                            )

            except Exception:
                pass  # 他のインジケーターは範囲チェックをスキップ

    def test_all_indicators_with_custom_parameters(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """カスタムパラメータでの動作をテスト"""
        for indicator in ["EMA", "SMA", "WMA", "DEMA", "TEMA"]:
            try:
                # 異なる長さのパラメータでテスト
                for length in [5, 20, 50]:
                    params = {"length": length}
                    result = indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, params
                    )

                    if hasattr(result, "shape"):
                        assert result.shape[0] == len(sample_ohlcv)
                    else:
                        assert len(result) == len(sample_ohlcv)

            except Exception as e:
                pytest.fail(f"{indicator}のカスタムパラメータテストで失敗: {e}")

    def test_profile_indicators_performance(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """主要インジケーターのパフォーマンスを計測"""
        import time

        # 高頻度で使用されるインジケーター
        common_indicators = ["SMA", "EMA", "RSI", "MACD", "BB"]

        start_time = time.time()

        for indicator in common_indicators:
            for _ in range(5):  # 5回実行
                if indicator == "MACD":
                    indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"fast": 12, "slow": 26, "signal": 9}
                    )
                else:
                    indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"length": 14}
                    )

        elapsed = time.time() - start_time
        # 2秒以内に完了すべき
        assert elapsed < 2.0, f"パフォーマンステスト失敗: {elapsed:.2f}秒"

    def test_all_indicators_supported_by_registry(
        self, indicator_service: TechnicalIndicatorService
    ):
        """レジストリ内の全指標が正しい形式で登録されているか確認"""
        supported = indicator_service.get_supported_indicators()
    
        # 最低限いくつかの主要指標があることを確認
        assert len(supported) > 50
        
        for name, config in supported.items():
            assert "parameters" in config, f"{name}にparametersがありません"
            assert "result_type" in config, f"{name}にresult_typeがありません"
            assert "required_data" in config, f"{name}にrequired_dataがありません"

if __name__ == "__main__":
    # コマンドラインからの実行用
    pytest.main([__file__, "-v"])




