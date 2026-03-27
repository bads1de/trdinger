"""
IndicatorServiceのテストモジュール

TechnicalIndicatorServiceの機能をテストする。
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.indicator_orchestrator import (
    TechnicalIndicatorService,
)


class TestTechnicalIndicatorService:
    """TechnicalIndicatorServiceクラスのテスト"""

    def test_initialization(self, indicator_service):
        """初期化テスト"""
        assert indicator_service.registry is not None

    def test_get_indicator_config(self, indicator_service):
        """指標設定取得テスト"""
        config = indicator_service._get_indicator_config("SMA")
        assert config is not None
        assert config.indicator_name == "SMA"

    def test_get_indicator_config_invalid(self, indicator_service):
        """無効な指標設定取得テスト"""
        with pytest.raises(ValueError, match="サポートされていない指標タイプ"):
            indicator_service._get_indicator_config(
                "INVALID_INDICATOR_THAT_DOES_NOT_EXIST"
            )

    @patch.object(TechnicalIndicatorService, "_get_pandas_ta_config")
    @patch.object(TechnicalIndicatorService, "_normalize_params")
    @patch.object(TechnicalIndicatorService, "_basic_validation")
    @patch.object(TechnicalIndicatorService, "_call_pandas_ta")
    @patch.object(TechnicalIndicatorService, "_post_process")
    def test_calculate_indicator_pandas_ta_success(
        self,
        mock_post_process,
        mock_call_pandas_ta,
        mock_validation,
        mock_normalize,
        mock_get_config,
        indicator_service,
        sample_df,
    ):
        """pandas-taを使用した指標計算成功テスト"""
        mock_config = {"function": "sma", "returns": "single"}
        mock_get_config.return_value = mock_config
        mock_normalize.return_value = {"length": 10}
        mock_validation.return_value = True
        mock_call_pandas_ta.return_value = pd.Series([1, 2, 3])
        mock_post_process.return_value = np.array([1, 2, 3])

        result = indicator_service.calculate_indicator(sample_df, "SMA", {"length": 10})

        assert isinstance(result, np.ndarray)
        mock_get_config.assert_called_with("SMA")
        mock_normalize.assert_called()
        mock_validation.assert_called()
        mock_call_pandas_ta.assert_called()
        mock_post_process.assert_called()

    @patch.object(TechnicalIndicatorService, "_get_pandas_ta_config")
    @patch.object(TechnicalIndicatorService, "_get_indicator_config")
    @patch.object(TechnicalIndicatorService, "_calculate_with_adapter")
    def test_calculate_indicator_adapter_fallback(
        self,
        mock_calculate_adapter,
        mock_get_indicator_config,
        mock_get_config,
        indicator_service,
        sample_df,
    ):
        """アダプターフォールバックテスト"""
        indicator_service.clear_cache()  # キャッシュをクリア
        mock_config = Mock()
        mock_config.adapter_function = Mock()
        mock_get_indicator_config.return_value = mock_config
        mock_get_config.return_value = None  # pandas-ta設定なし
        mock_calculate_adapter.return_value = np.array([1, 2, 3])

        result = indicator_service.calculate_indicator(sample_df, "SMA", {"length": 10})

        assert isinstance(result, np.ndarray)
        mock_calculate_adapter.assert_called_once()

    def test_calculate_indicator_unsupported(self, indicator_service, sample_df):
        """サポートされていない指標テスト"""
        with patch.object(
            indicator_service, "_get_pandas_ta_config", return_value=None
        ):
            with patch.object(
                indicator_service, "_get_indicator_config", side_effect=ValueError
            ):
                with pytest.raises(ValueError, match="実装が見つかりません"):
                    indicator_service.calculate_indicator(sample_df, "UNSUPPORTED", {})

    @patch(
        "app.services.indicators.indicator_orchestrator.validate_data_length_with_fallback"
    )
    def test_basic_validation_success(
        self, mock_validate_length, indicator_service, sample_df
    ):
        """基本検証成功テスト"""
        mock_validate_length.return_value = (True, 10)
        config = {"function": "sma", "data_column": "close", "multi_column": False}

        result = indicator_service._basic_validation(sample_df, config, {"length": 10})
        assert result is True

    @patch(
        "app.services.indicators.indicator_orchestrator.validate_data_length_with_fallback"
    )
    def test_basic_validation_data_too_short(
        self, mock_validate_length, indicator_service, sample_df
    ):
        """データ長不足の検証テスト"""
        mock_validate_length.return_value = (False, 10)
        config = {"function": "sma"}

        result = indicator_service._basic_validation(sample_df, config, {"length": 100})
        assert result is False

    def test_resolve_column_name(self, indicator_service, sample_df):
        """カラム名解決テスト"""
        # 直接一致
        assert indicator_service._resolve_column_name(sample_df, "close") == "close"

        # 大文字
        assert indicator_service._resolve_column_name(sample_df, "Close") == "close"

        # 小文字
        assert indicator_service._resolve_column_name(sample_df, "CLOSE") == "close"

        # 存在しないカラム
        assert indicator_service._resolve_column_name(sample_df, "nonexistent") is None

    def test_normalize_params(self, indicator_service):
        """パラメータ正規化テスト"""
        config = {
            "params": {"length": ["period"], "multiplier": ["factor"]},
            "default_values": {"length": 14},
        }
        params = {"period": 20, "other": 1.5}

        result = indicator_service._normalize_params(params, config)
        assert result["length"] == 20
        assert "multiplier" not in result  # デフォルト値なしで入力なし

    def test_normalize_params_min_length_guard(self, indicator_service):
        """最小長ガードのテスト"""
        config = {
            "params": {"length": ["period"]},
            "default_values": {},
            "min_length": 5,
        }
        params = {"period": 3}  # 最小値未満

        result = indicator_service._normalize_params(params, config)
        assert result["length"] == 5  # 調整される

    @patch("app.services.indicators.indicator_orchestrator.ta")
    def test_call_pandas_ta_single_column(self, mock_ta, indicator_service, sample_df):
        """pandas-ta単一カラム呼び出しテスト"""
        expected_series = pd.Series([1, 2, 3])
        mock_func = Mock(return_value=expected_series)
        mock_ta.sma = mock_func

        config = {"function": "sma", "data_column": "close", "multi_column": False}
        params = {"length": 10}

        result = indicator_service._call_pandas_ta(sample_df, config, params)

        assert result.equals(expected_series)
        mock_func.assert_called_once()

    @patch("app.services.indicators.indicator_orchestrator.create_nan_result")
    def test_create_nan_result(self, mock_create_nan, indicator_service, sample_df):
        """NaN結果作成テスト"""
        mock_create_nan.return_value = np.array([np.nan] * len(sample_df))
        config = {"function": "sma"}

        result = indicator_service._create_nan_result(sample_df, config)
        assert isinstance(result, np.ndarray)
        mock_create_nan.assert_called_once()

    def test_post_process_single_return(self, indicator_service):
        """単一戻り値の後処理テスト"""
        config = {"returns": "single"}
        result = pd.Series([1, 2, 3])

        processed = indicator_service._post_process(result, config)
        assert isinstance(processed, np.ndarray)
        assert np.array_equal(processed, [1, 2, 3])

    def test_post_process_multiple_return(self, indicator_service):
        """複数戻り値の後処理テスト"""
        config = {"returns": "multiple", "return_cols": ["upper", "lower"]}
        result = pd.DataFrame({"upper": [1, 2, 3], "lower": [0.5, 1.5, 2.5]})

        processed = indicator_service._post_process(result, config)
        assert isinstance(processed, tuple)
        assert len(processed) == 2

    def test_get_supported_indicators(self, indicator_service):
        """サポート指標取得テスト"""
        result = indicator_service.get_supported_indicators()

        # SMAが含まれていることを確認
        assert "SMA" in result

        # SMA設定が正しい形式であることを確認
        sma_config = result["SMA"]
        assert "parameters" in sma_config
        assert "result_type" in sma_config
        assert "required_data" in sma_config

        # lengthパラメータが存在することを確認
        assert "length" in sma_config["parameters"]
        assert "default" in sma_config["parameters"]["length"]
        assert "min" in sma_config["parameters"]["length"]
        assert "max" in sma_config["parameters"]["length"]

    def test_clear_cache(self, indicator_service, sample_df):
        """キャッシュクリアのテスト"""
        # 一度計算してキャッシュさせる
        indicator_service.calculate_indicator(sample_df, "SMA", {"length": 10})
        assert len(indicator_service._calculation_cache) > 0

        # クリア
        indicator_service.clear_cache()
        assert len(indicator_service._calculation_cache) == 0

    def test_calculate_unknown_indicator(self, indicator_service, sample_df):
        """未登録の指標に対するテスト"""
        with pytest.raises(ValueError, match="実装が見つかりません"):
            indicator_service.calculate_indicator(sample_df, "UNKNOWN_INDICATOR", {})

    def test_calculate_indicator_error_handling(self, indicator_service, sample_df):
        """指標計算時のエラーハンドリングテスト"""
        unique_df = sample_df.copy()
        with patch.object(
            indicator_service,
            "_get_pandas_ta_config",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(Exception):
                indicator_service.calculate_indicator(unique_df, "SMA", {"length": 10})

    def test_make_cache_key(self, indicator_service, sample_df):
        """キャッシュキー生成テスト"""
        cache_key = indicator_service._make_cache_key("SMA", {"length": 10}, sample_df)
        assert cache_key is not None
        assert len(cache_key) == 3
        assert cache_key[0] == "SMA"

    def test_cache_invalidation_bug(self, indicator_service, sample_df):
        """キャッシュ無効化が in-place 更新でも効くことを確認する"""
        params = {"length": 5}
        indicator = "SMA"
        data = sample_df.copy()

        # 1回目の計算
        result1 = indicator_service.calculate_indicator(data, indicator, params)

        # 同じ DataFrame を in-place 更新する
        data["close"] = data["close"] * 2  # 値を2倍にする

        # 2回目の計算
        result2 = indicator_service.calculate_indicator(data, indicator, params)

        # 本来は異なる結果になるべき。NaN を同位置に含んでも判定できるようにする。
        assert not np.allclose(
            result1,
            result2,
            equal_nan=True,
        ), "データが変更されたのにキャッシュされた古い結果が返されています"

    def test_cache_key_reflects_dataframe_mutation(self, indicator_service, sample_df):
        """キャッシュキーが DataFrame の内容変化を反映することを確認する"""
        params = {"length": 5}
        indicator = "SMA"
        data = sample_df.copy()

        # 1. 生成時に DataFrame へ属性を載せない
        key1 = indicator_service._make_cache_key(indicator, params, data)
        assert key1 is not None
        assert not hasattr(data, "_cached_hash")

        # 2. 同じ DataFrame を in-place 更新したらキーが変わる
        data.loc[data.index[0], "close"] = data.loc[data.index[0], "close"] * 10
        key2 = indicator_service._make_cache_key(indicator, params, data)
        assert key2 is not None
        assert key1 != key2

    def test_registry_has_indicators(self, indicator_service):
        """レジストリに指標が登録されているか確認"""
        assert indicator_service.registry is not None
        # レジストリから設定を取得できることを確認
        sma_config = indicator_service.registry.get_indicator_config("SMA")
        assert sma_config is not None

    def test_calculate_single_indicator_invalid_data(self, indicator_service):
        """無効なデータでの指標計算テスト"""
        data = pd.DataFrame()  # 空のデータ

        # 空データはNaN結果を返す（例外は投げない）
        result = indicator_service.calculate_indicator(data, "SMA", {"length": 5})
        assert result is not None
        # 空の結果またはNaNで埋められた結果が返される
        assert isinstance(result, (np.ndarray, pd.Series, tuple))

    def test_calculate_rsi(self, indicator_service):
        """RSI指標計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 2,
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
                * 2,
            }
        )

        result = indicator_service.calculate_indicator(data, "RSI", {"length": 14})

        assert result is not None
        assert isinstance(result, (np.ndarray, pd.Series))

    def test_calculate_macd(self, indicator_service):
        """MACD指標計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 3,
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
                * 3,
            }
        )

        result = indicator_service.calculate_indicator(
            data, "MACD", {"fast": 12, "slow": 26, "signal": 9}
        )

        # MACDは複数の値を返す（tuple）
        assert result is not None
        assert isinstance(result, tuple)
