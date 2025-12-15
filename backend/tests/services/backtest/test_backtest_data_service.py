"""
バックテストデータサービステスト

BacktestDataServiceの機能をテストします。
"""

import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.data.data_integration_service import DataIntegrationError
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_ohlcv_repo():
    """モックOHLCVリポジトリ"""
    return MagicMock(spec=OHLCVRepository)


@pytest.fixture
def mock_oi_repo():
    """モックOpen Interestリポジトリ"""
    return MagicMock(spec=OpenInterestRepository)


@pytest.fixture
def mock_fr_repo():
    """モックFunding Rateリポジトリ"""
    return MagicMock(spec=FundingRateRepository)


@pytest.fixture
def backtest_data_service(mock_ohlcv_repo, mock_oi_repo, mock_fr_repo):
    """BacktestDataServiceインスタンス"""
    return BacktestDataService(
        ohlcv_repo=mock_ohlcv_repo,
        oi_repo=mock_oi_repo,
        fr_repo=mock_fr_repo,
    )


@pytest.fixture
def sample_ohlcv_data():
    """テスト用OHLCVデータ"""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [105.0 + i * 0.1 for i in range(100)],
            "low": [95.0 + i * 0.1 for i in range(100)],
            "close": [102.0 + i * 0.1 for i in range(100)],
            "volume": [1000.0 + i * 10 for i in range(100)],
        }
    ).set_index("timestamp")


@pytest.fixture
def sample_oi_data():
    """テスト用Open Interestデータ"""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open_interest": [5000.0 + i * 50 for i in range(100)],
        }
    ).set_index("timestamp")


@pytest.fixture
def sample_fr_data():
    """テスト用Funding Rateデータ"""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "funding_rate": [0.0001 + i * 0.00001 for i in range(100)],
        }
    ).set_index("timestamp")


class TestServiceInitialization:
    """サービス初期化テスト"""

    def test_initialize_with_repositories(
        self, mock_ohlcv_repo, mock_oi_repo, mock_fr_repo
    ):
        """リポジトリ付きで初期化できること"""
        service = BacktestDataService(
            ohlcv_repo=mock_ohlcv_repo,
            oi_repo=mock_oi_repo,
            fr_repo=mock_fr_repo,
        )

        assert service.ohlcv_repo == mock_ohlcv_repo
        assert service.oi_repo == mock_oi_repo
        assert service.fr_repo == mock_fr_repo
        assert service._retrieval_service is not None
        assert service._conversion_service is not None
        assert service._integration_service is not None

    def test_initialize_without_repositories(self):
        """リポジトリなしで初期化できること"""
        service = BacktestDataService()

        assert service.ohlcv_repo is None
        assert service.oi_repo is None
        assert service.fr_repo is None


class TestDataRetrieval:
    """データ取得テスト"""

    def test_get_data_for_backtest_success(
        self, backtest_data_service, sample_ohlcv_data
    ):
        """バックテスト用データを正常に取得できること"""
        with patch.object(
            backtest_data_service._integration_service,
            "create_backtest_dataframe",
            return_value=sample_ohlcv_data,
        ):
            result = backtest_data_service.get_data_for_backtest(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert not result.empty
            assert len(result) == 100
            assert "open" in result.columns
            assert "close" in result.columns

    def test_get_data_for_backtest_with_additional_data(
        self, backtest_data_service, sample_ohlcv_data, sample_oi_data, sample_fr_data
    ):
        """OI・FRデータを含むバックテスト用データを取得できること"""
        integrated_data = sample_ohlcv_data.copy()
        integrated_data["open_interest"] = sample_oi_data["open_interest"]
        integrated_data["funding_rate"] = sample_fr_data["funding_rate"]

        with patch.object(
            backtest_data_service._integration_service,
            "create_backtest_dataframe",
            return_value=integrated_data,
        ):
            result = backtest_data_service.get_data_for_backtest(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert "open_interest" in result.columns
            assert "funding_rate" in result.columns
            assert not result["open_interest"].isna().all()
            assert not result["funding_rate"].isna().all()

    def test_get_data_for_backtest_empty_data(self, backtest_data_service):
        """データが空の場合の処理"""
        with patch.object(
            backtest_data_service._integration_service,
            "create_backtest_dataframe",
            return_value=pd.DataFrame(),
        ):
            result = backtest_data_service.get_data_for_backtest(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert result.empty

    def test_get_data_for_backtest_integration_error(self, backtest_data_service):
        """データ統合エラー時の処理"""
        with patch.object(
            backtest_data_service._integration_service,
            "create_backtest_dataframe",
            side_effect=DataIntegrationError("統合エラー"),
        ):
            with pytest.raises(
                ValueError, match="バックテスト用データの作成に失敗しました"
            ):
                backtest_data_service.get_data_for_backtest(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5),
                )


class TestOHLCVDataRetrieval:
    """OHLCVデータ取得テスト"""

    def test_get_ohlcv_data_success(self, backtest_data_service, sample_ohlcv_data):
        """OHLCVデータを正常に取得できること"""
        with (
            patch.object(
                backtest_data_service._retrieval_service,
                "get_ohlcv_data",
                return_value=[],
            ),
            patch.object(
                backtest_data_service._conversion_service,
                "convert_ohlcv_to_dataframe",
                return_value=sample_ohlcv_data,
            ),
        ):
            result = backtest_data_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert not result.empty
            assert "open" in result.columns
            assert "high" in result.columns
            assert "low" in result.columns
            assert "close" in result.columns
            assert "volume" in result.columns

    def test_get_ohlcv_data_numeric_conversion(
        self, backtest_data_service, sample_ohlcv_data
    ):
        """OHLCVデータが数値型に変換されること"""
        with (
            patch.object(
                backtest_data_service._retrieval_service,
                "get_ohlcv_data",
                return_value=[],
            ),
            patch.object(
                backtest_data_service._conversion_service,
                "convert_ohlcv_to_dataframe",
                return_value=sample_ohlcv_data,
            ),
        ):
            result = backtest_data_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert pd.api.types.is_numeric_dtype(result["open"])
            assert pd.api.types.is_numeric_dtype(result["high"])
            assert pd.api.types.is_numeric_dtype(result["low"])
            assert pd.api.types.is_numeric_dtype(result["close"])
            assert pd.api.types.is_numeric_dtype(result["volume"])

    def test_get_ohlcv_data_empty_result(self, backtest_data_service):
        """OHLCVデータが空の場合の処理"""
        with (
            patch.object(
                backtest_data_service._retrieval_service,
                "get_ohlcv_data",
                return_value=[],
            ),
            patch.object(
                backtest_data_service._conversion_service,
                "convert_ohlcv_to_dataframe",
                return_value=pd.DataFrame(),
            ),
        ):
            result = backtest_data_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert result.empty


class TestMLTrainingData:
    """MLトレーニングデータ取得テスト"""

    def test_get_ml_training_data_success(
        self, backtest_data_service, sample_ohlcv_data
    ):
        """MLトレーニング用データを正常に取得できること"""
        with patch.object(
            backtest_data_service._integration_service,
            "create_ml_training_dataframe",
            return_value=sample_ohlcv_data,
        ):
            result = backtest_data_service.get_ml_training_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert not result.empty
            assert len(result) == 100

    def test_get_ml_training_data_with_all_features(
        self, backtest_data_service, sample_ohlcv_data, sample_oi_data, sample_fr_data
    ):
        """全ての特徴量を含むMLトレーニングデータを取得できること"""
        integrated_data = sample_ohlcv_data.copy()
        integrated_data["open_interest"] = sample_oi_data["open_interest"]
        integrated_data["funding_rate"] = sample_fr_data["funding_rate"]

        with patch.object(
            backtest_data_service._integration_service,
            "create_ml_training_dataframe",
            return_value=integrated_data,
        ):
            result = backtest_data_service.get_ml_training_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert "open_interest" in result.columns
            assert "funding_rate" in result.columns

    def test_get_ml_training_data_integration_error(self, backtest_data_service):
        """MLトレーニングデータ統合エラー時の処理"""
        with patch.object(
            backtest_data_service._integration_service,
            "create_ml_training_dataframe",
            side_effect=DataIntegrationError("統合エラー"),
        ):
            with pytest.raises(
                ValueError, match="MLトレーニング用データの作成に失敗しました"
            ):
                backtest_data_service.get_ml_training_data(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5),
                )


class TestEventLabeledData:
    """イベントラベル付きデータテスト"""

    def test_get_event_labeled_training_data_success(
        self, backtest_data_service, sample_ohlcv_data
    ):
        """イベントラベル付きデータを正常に取得できること"""
        labels_df = pd.DataFrame(
            {
                "label": ["HRHP", "LRLP"] * 50,
            },
            index=sample_ohlcv_data.index,
        )
        profile_info = {
            "regime_profiles": {"regime_1": {"count": 50}},
            "label_distribution": {"HRHP": 50, "LRLP": 50},
        }

        with (
            patch.object(
                backtest_data_service._integration_service,
                "create_ml_training_dataframe",
                return_value=sample_ohlcv_data,
            ),
            patch.object(
                backtest_data_service._event_label_generator,
                "generate_hrhp_lrlp_labels",
                return_value=(labels_df, profile_info),
            ),
        ):
            labeled_df, info = backtest_data_service.get_event_labeled_training_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert not labeled_df.empty
            assert "label" in labeled_df.columns
            assert "regime_profiles" in info
            assert "label_distribution" in info

    def test_get_event_labeled_training_data_empty_data(self, backtest_data_service):
        """空のデータでイベントラベリングをスキップすること"""
        with patch.object(
            backtest_data_service._integration_service,
            "create_ml_training_dataframe",
            return_value=pd.DataFrame(),
        ):
            labeled_df, info = backtest_data_service.get_event_labeled_training_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert labeled_df.empty
            assert info["regime_profiles"] == {}
            assert info["label_distribution"] == {}


class TestDataSummary:
    """データ概要取得テスト"""

    def test_get_data_summary_success(self, backtest_data_service, sample_ohlcv_data):
        """データ概要を正常に取得できること"""
        summary = {
            "total_records": 100,
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-05T03:00:00",
            "columns": ["open", "high", "low", "close", "volume"],
        }

        with patch.object(
            backtest_data_service._integration_service,
            "get_data_summary",
            return_value=summary,
        ):
            result = backtest_data_service.get_data_summary(sample_ohlcv_data)

            assert result["total_records"] == 100
            assert "start_date" in result
            assert "end_date" in result
            assert "columns" in result


class TestDataValidation:
    """データ検証テスト"""

    def test_validate_ohlcv_data_structure(
        self, backtest_data_service, sample_ohlcv_data
    ):
        """OHLCVデータの構造が正しいこと"""
        with (
            patch.object(
                backtest_data_service._retrieval_service,
                "get_ohlcv_data",
                return_value=[],
            ),
            patch.object(
                backtest_data_service._conversion_service,
                "convert_ohlcv_to_dataframe",
                return_value=sample_ohlcv_data,
            ),
        ):
            result = backtest_data_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            # OHLCVの基本的なプロパティを検証
            assert result["high"].min() >= result["low"].max() - 100  # 概ね正しい範囲
            assert (result["volume"] >= 0).all()

    def test_validate_integrated_data_completeness(
        self, backtest_data_service, sample_ohlcv_data
    ):
        """統合データの完全性を検証できること"""
        integrated_data = sample_ohlcv_data.copy()
        integrated_data["open_interest"] = 5000.0
        integrated_data["funding_rate"] = 0.0001

        with patch.object(
            backtest_data_service._integration_service,
            "create_backtest_dataframe",
            return_value=integrated_data,
        ):
            result = backtest_data_service.get_data_for_backtest(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            # 全てのカラムが存在することを確認
            expected_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
                "funding_rate",
            ]
            for col in expected_columns:
                assert col in result.columns


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_handle_repository_none(self):
        """リポジトリがNoneの場合でも動作すること"""
        service = BacktestDataService(
            ohlcv_repo=None,
            oi_repo=None,
            fr_repo=None,
        )

        assert service.ohlcv_repo is None
        assert service.oi_repo is None
        assert service.fr_repo is None

    def test_handle_data_retrieval_error(self, backtest_data_service):
        """データ取得エラーを適切に処理すること"""
        with patch.object(
            backtest_data_service._retrieval_service,
            "get_ohlcv_data",
            side_effect=Exception("データベースエラー"),
        ):
            with pytest.raises(Exception):
                backtest_data_service.get_ohlcv_data(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5),
                )

    def test_handle_date_range_error(self, backtest_data_service):
        """不正な日付範囲を処理すること"""
        with patch.object(
            backtest_data_service._integration_service,
            "create_backtest_dataframe",
            side_effect=DataIntegrationError("開始日が終了日より後です"),
        ):
            with pytest.raises(
                ValueError, match="バックテスト用データの作成に失敗しました"
            ):
                backtest_data_service.get_data_for_backtest(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    start_date=datetime(2024, 1, 5),
                    end_date=datetime(2024, 1, 1),
                )


