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
from app.services.indicators.config import IndicatorResultType


class TestAllTechnicalIndicators:
    """全テクニカル指標の網羅的テストクラス"""

    _SPARSE_STANDARD_INDICATORS = {"TD_SEQ", "DECAY", "EBSW", "HA", "HWC", "JMA", "PMAX", "STC", "VIDYA"}

    # 指標リストを動的に取得
    try:
        from app.services.indicators.config.indicator_config import (
            indicator_registry,
            initialize_all_indicators,
        )

        initialize_all_indicators()
        INDICATORS = sorted(indicator_registry.list_indicators())
    except Exception:
        # フォールバック
        INDICATORS = ["RSI", "SMA", "EMA", "MACD", "BBANDS"]

    @staticmethod
    def _count_finite_values(result) -> int:
        arrays = result if isinstance(result, tuple) else (result,)
        finite_count = 0
        for array in arrays:
            finite_count += int(np.isfinite(np.asarray(array)).sum())
        return finite_count

    @staticmethod
    def _has_required_data(
        indicator_service: TechnicalIndicatorService,
        df: pd.DataFrame,
        required_data: list[str],
    ) -> bool:
        return all(
            indicator_service._resolve_column_name(df, data_key) is not None
            for data_key in required_data
        )

    def test_all_indicators_initializable(self, indicator_service):
        """すべてのインジケーターが初期化可能か確認"""
        for indicator in self.INDICATORS:
            try:
                # エイリアスを解決
                resolved_name = indicator_service._resolve_indicator_name(indicator)
                # 各インジケーターの設定を取得できるか確認
                config = indicator_service.registry.get_indicator_config(resolved_name)
                assert (
                    config is not None
                ), f"{indicator} ({resolved_name}) の設定が見つかりません"
                assert (
                    config.adapter_function is not None
                    or config.pandas_function is not None
                ), f"{indicator}に実装（アダプターまたはpandas-ta関数）がありません"
            except Exception as e:
                pytest.fail(f"{indicator}の初期化に失敗: {e}")

    def test_registry_does_not_include_calculate_wrappers(self):
        """discovery が calculate_* ラッパーを実指標として登録しないこと"""
        assert not any(name.startswith("CALCULATE_") for name in self.INDICATORS)

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

            if not self._has_required_data(
                indicator_service, sample_ohlcv, config.required_data
            ):
                pytest.skip(
                    f"{indicator} は sample_ohlcv にない列を要求するためスキップ"
                )

            # デフォルトパラメータでテスト
            default_params = config.default_values or {}
            final_params = dict(default_params)

            # 指標ごとの主要パラメータ名を特定
            supported_params = set(config.parameters.keys())

            # テスト用パラメータを設定（サポートされている場合のみ）
            def set_param(test_key, value, aliases=None):
                actual_key = None
                if test_key in supported_params:
                    actual_key = test_key
                elif aliases:
                    for alias in aliases:
                        if alias in supported_params:
                            actual_key = alias
                            break

                if actual_key is None:
                    return False

                if actual_key in {"length", "period", "window", "n"}:
                    param_config = config.parameters.get(actual_key)
                    current_value = final_params.get(actual_key)
                    safe_value = value

                    if isinstance(current_value, (int, float)):
                        safe_value = max(safe_value, current_value)
                    if param_config is not None and isinstance(
                        param_config.min_value, (int, float)
                    ):
                        safe_value = max(safe_value, param_config.min_value)

                    if isinstance(value, int) or isinstance(current_value, int):
                        safe_value = int(safe_value)

                    final_params[actual_key] = safe_value
                    return True

                final_params[actual_key] = value
                return True

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

            # 計算を実行
            result = indicator_service.calculate_indicator(
                sample_ohlcv, indicator, final_params
            )
            support_info = indicator_service.get_supported_indicators().get(
                indicator, {}
            )

            # 結果の形式を検証
            if config.result_type == IndicatorResultType.SINGLE:
                assert isinstance(
                    result, np.ndarray
                ), f"{indicator}の結果がndarrayではありません"
                assert result.shape[0] == len(
                    sample_ohlcv
                ), f"{indicator}の結果の長さが不正"
                # NaNを含む場合があるが、最後の数ポイントは有効な場合がある
                assert result.shape[0] > 0, f"{indicator}の結果が空"

            elif config.result_type == IndicatorResultType.COMPLEX:
                assert isinstance(
                    result, tuple
                ), f"{indicator}の結果がtupleではありません"
                assert len(result) > 0, f"{indicator}の結果が空のtuple"
                for i, series in enumerate(result):
                    assert isinstance(
                        series, np.ndarray
                    ), f"{indicator}の結果[{i}]がndarrayではありません"
                    assert series.shape[0] == len(
                        sample_ohlcv
                    ), f"{indicator}の結果[{i}]の長さが不正{indicator}"

            if (
                support_info.get("support_tier") == "standard"
                and indicator not in self._SPARSE_STANDARD_INDICATORS
            ):
                assert (
                    self._count_finite_values(result) > 0
                ), f"{indicator} は標準 OHLCV 指標なのに有限値を返していません"

        except Exception as e:
            pytest.fail(f"{indicator}のテストでエラー: {e}")

    def test_td_seq_produces_sparse_events_on_trending_data(
        self,
        indicator_service: TechnicalIndicatorService,
    ):
        """TD_SEQ はトレンド系列で疎なイベント値を返す。"""
        index = pd.date_range("2022-01-01", periods=500, freq="h")
        close = pd.Series(np.linspace(100.0, 150.0, 500), index=index)
        trending_ohlcv = pd.DataFrame(
            {
                "Open": close.shift(1).fillna(close.iloc[0]),
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": np.linspace(1000.0, 2000.0, 500),
            },
            index=index,
        )

        result = indicator_service.calculate_indicator(trending_ohlcv, "TD_SEQ", {})

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(series, np.ndarray) for series in result)
        assert all(series.shape[0] == len(trending_ohlcv) for series in result)
        assert self._count_finite_values(result) > 0

    @pytest.mark.parametrize("indicator", INDICATORS)
    def test_indicator_handle_insufficient_data(
        self, indicator: str, indicator_service: TechnicalIndicatorService
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

        try:
            config = indicator_service.registry.get_indicator_config(indicator)
            if not config:
                pytest.skip(f"{indicator}の設定が見つからないためスキップ")

            if not self._has_required_data(
                indicator_service, short_data, config.required_data
            ):
                pytest.skip(f"{indicator} は short_data にない列を要求するためスキップ")

            # 標準OHLCV以外のデータを必要とするインジケーターはスキップ
            standard_ohlcv = {"close", "high", "low", "open", "volume"}
            required_data_lower = {d.lower() for d in config.required_data}
            if not required_data_lower.issubset(standard_ohlcv):
                pytest.skip(f"{indicator} は標準OHLCV以外のデータを必要とするためスキップ")

            # 短いデータで計算を試みる
            params = config.default_values or {}

            result = indicator_service.calculate_indicator(
                short_data, indicator, params
            )

            # 結果が期待通りか検証
            if config.result_type == IndicatorResultType.SINGLE:
                assert isinstance(result, np.ndarray)
                # データ不足の場合は空の配列が返ってくる可能性がある
                if result.shape[0] > 0:
                    assert result.shape[0] == len(short_data)
            elif config.result_type == IndicatorResultType.COMPLEX:
                if isinstance(result, tuple):
                    assert len(result) > 0
                    for series in result:
                        assert isinstance(series, np.ndarray)
                        if series.shape[0] > 0:
                            assert series.shape[0] == len(short_data)
                else:
                    assert isinstance(result, np.ndarray)
                    if result.shape[0] > 0:
                        assert result.shape[0] == len(short_data)

        except Exception as e:
            # データ不足に関連するエラーは許容
            error_msg = str(e).lower()
            # データ長、長さ、インデックス、サイズに関連するエラーを許容
            allowed_errors = [
                "データ長",
                "長さ",
                "length",
                "insufficient",
                "index",
                "size",
                "required",
                "need",
                "out of range",
                "position",
                "loc",
                "iloc",
            ]
            if not any(err in error_msg for err in allowed_errors):
                print(f"Error message for {indicator}: {error_msg}")
                pytest.fail(f"{indicator}で予期しないエラー: {e}")

    @pytest.mark.parametrize("indicator", ["RSI", "MFI", "CCI", "WILLR", "UO"])
    def test_indicator_returns_valid_values(
        self,
        indicator: str,
        indicator_service: TechnicalIndicatorService,
        sample_ohlcv: pd.DataFrame,
    ):
        """特定のインジケーターが有効な値を返すか検証"""
        try:
            params = {"length": 14}
            # UOはパラメータが異なる
            if indicator == "UO":
                params = {}

            result = indicator_service.calculate_indicator(
                sample_ohlcv, indicator, params
            )

            if not isinstance(result, np.ndarray):
                pytest.skip(f"{indicator}の結果がndarrayでないためスキップ")

            valid_values = result[np.isfinite(result)]
            if len(valid_values) == 0:
                pytest.skip(f"{indicator}の有効な値がないためスキップ")

            if indicator == "RSI":
                assert all(
                    0 <= val <= 100 for val in valid_values
                ), f"{indicator}の値が範囲外"
            elif indicator == "MFI":
                assert all(
                    0 <= val <= 100 for val in valid_values
                ), f"{indicator}の値が範囲外"
            elif indicator == "WILLR":
                assert all(
                    -100 <= val <= 0 for val in valid_values
                ), f"{indicator}の値が範囲外"

        except Exception as e:
            pytest.fail(f"{indicator}の範囲チェックでエラー: {e}")

    @pytest.mark.parametrize("indicator", ["EMA", "SMA", "WMA", "DEMA", "TEMA"])
    def test_indicator_with_custom_parameters(
        self,
        indicator: str,
        indicator_service: TechnicalIndicatorService,
        sample_ohlcv: pd.DataFrame,
    ):
        """カスタムパラメータでの動作をテスト"""
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
