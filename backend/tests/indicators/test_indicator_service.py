"""
IndicatorServiceのテストモジュール

TechnicalIndicatorServiceの機能をテストする。
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.services.indicators.indicator_orchestrator import (
    TechnicalIndicatorService,
)


class TestTechnicalIndicatorService:
    """TechnicalIndicatorServiceクラスのテスト"""

    @pytest.fixture
    def indicator_service(self):
        """TechnicalIndicatorServiceインスタンス"""
        return TechnicalIndicatorService()

    @pytest.fixture
    def sample_df(self):
        """サンプルOHLCVデータ"""
        return pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104] * 20,
                "high": [105, 106, 107, 108, 109] * 20,
                "low": [95, 96, 97, 98, 99] * 20,
                "close": [102, 103, 104, 105, 106] * 20,
                "volume": [1000, 1100, 1200, 1300, 1400] * 20,
            }
        )

    def test_initialization(self, indicator_service):
        """初期化テスト"""
        assert indicator_service.registry is not None

    def test_get_indicator_config(self, indicator_service):
        """指標設定取得テスト"""
        # 実際のregistryを使用してSMA設定を取得
        config = indicator_service._get_indicator_config("SMA")
        assert config is not None
        assert config.indicator_name == "SMA"
        assert config.adapter_function is not None

    def test_get_indicator_config_invalid(self, indicator_service):
        """無効な指標設定取得テスト"""
        # 実際のregistryは存在しない指標に対してNoneを返すか、例外を発生させる
        with pytest.raises(ValueError, match="サポートされていない指標タイプ"):
            indicator_service._get_indicator_config(
                "INVALID_INDICATOR_THAT_DOES_NOT_EXIST"
            )

    @patch.object(TechnicalIndicatorService, "_get_config")
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
        # モックの設定
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

    @patch.object(TechnicalIndicatorService, "_get_config")
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
        with patch.object(indicator_service, "_get_config", return_value=None):
            with patch.object(
                indicator_service, "_get_indicator_config", side_effect=ValueError
            ):
                with pytest.raises(ValueError, match="実装が見つかりません"):
                    indicator_service.calculate_indicator(sample_df, "UNSUPPORTED", {})

    @patch.object(TechnicalIndicatorService, "validate_data_length_with_fallback")
    def test_basic_validation_success(
        self, mock_validate_length, indicator_service, sample_df
    ):
        """基本検証成功テスト"""
        mock_validate_length.return_value = (True, 10)
        config = {"function": "sma", "data_column": "close", "multi_column": False}

        result = indicator_service._basic_validation(sample_df, config, {"length": 10})
        assert result is True

    @patch.object(TechnicalIndicatorService, "validate_data_length_with_fallback")
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

    @patch("backend.app.services.indicators.indicator_orchestrator.ta")
    def test_call_pandas_ta_single_column(self, mock_ta, indicator_service, sample_df):
        """pandas-ta単一カラム呼び出しテスト"""
        expected_series = pd.Series([1, 2, 3])
        mock_func = Mock(return_value=expected_series)
        mock_ta.sma = mock_func

        config = {"function": "sma", "data_column": "close", "multi_column": False}
        params = {"length": 10}

        result = indicator_service._call_pandas_ta(sample_df, config, params)

        # pandas Seriesの比較は.equals()を使用
        assert result.equals(expected_series)
        mock_func.assert_called_once()

    @patch("backend.app.services.indicators.indicator_orchestrator.create_nan_result")
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
        # 実際のregistryを使用してサポート指標リストを取得
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

    def test_validate_data_length_with_fallback(self, indicator_service, sample_df):
        """データ長検証テスト"""
        result = indicator_service.validate_data_length_with_fallback(
            sample_df, "SMA", {"length": 10}
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], int)

    def test_calculate_indicator_error_handling(self, indicator_service, sample_df):
        """指標計算時のエラーハンドリングテスト"""
        with patch.object(
            indicator_service, "_get_config", side_effect=Exception("Test error")
        ):
            with pytest.raises(Exception):
                indicator_service.calculate_indicator(sample_df, "SMA", {"length": 10})
