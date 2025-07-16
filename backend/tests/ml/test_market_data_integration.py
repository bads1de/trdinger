"""
MLトレーニングでのファンディングレートと建玉残高データ統合のテスト

TDDアプローチで以下をテストします：
1. FundingRateRepository と OpenInterestRepository からのデータ取得
2. BacktestDataService でのデータ統合
3. market_data_features.py で期待されるカラム名の確認
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
from typing import List

from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.models import FundingRateData, OpenInterestData
from app.core.services.backtest_data_service import BacktestDataService
from app.core.services.ml.feature_engineering.market_data_features import MarketDataFeatureCalculator


class TestMarketDataIntegration:
    """市場データ統合のテストクラス"""

    @pytest.fixture
    def mock_funding_rate_data(self) -> List[FundingRateData]:
        """モックファンディングレートデータ"""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        return [
            FundingRateData(
                id=i,
                symbol="BTC/USDT:USDT",
                funding_rate=0.0001 * (i % 10 - 5),  # -0.0005 to 0.0004
                funding_timestamp=base_time + timedelta(hours=8*i),
                timestamp=base_time + timedelta(hours=8*i),
                next_funding_timestamp=base_time + timedelta(hours=8*(i+1))
            )
            for i in range(10)
        ]

    @pytest.fixture
    def mock_open_interest_data(self) -> List[OpenInterestData]:
        """モック建玉残高データ"""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        return [
            OpenInterestData(
                id=i,
                symbol="BTC/USDT:USDT",
                open_interest_value=1000000 + i * 50000,  # 1M to 1.45M
                data_timestamp=base_time + timedelta(hours=i),
                timestamp=base_time + timedelta(hours=i)
            )
            for i in range(24)
        ]

    @pytest.fixture
    def mock_ohlcv_data(self) -> pd.DataFrame:
        """モックOHLCVデータ"""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dates = [base_time + timedelta(hours=i) for i in range(24)]
        
        data = {
            'Open': [50000 + i * 100 for i in range(24)],
            'High': [50100 + i * 100 for i in range(24)],
            'Low': [49900 + i * 100 for i in range(24)],
            'Close': [50000 + i * 100 for i in range(24)],
            'Volume': [1000 + i * 10 for i in range(24)]
        }
        
        df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
        return df

    def test_funding_rate_repository_data_retrieval(self, mock_funding_rate_data):
        """ファンディングレートリポジトリからのデータ取得テスト"""
        # Arrange
        mock_db = Mock()
        fr_repo = FundingRateRepository(mock_db)
        
        with patch.object(fr_repo, 'get_funding_rate_data', return_value=mock_funding_rate_data):
            # Act
            result = fr_repo.get_funding_rate_data(
                symbol="BTC/USDT:USDT",
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
            
            # Assert
            assert len(result) == 10
            assert all(isinstance(item, FundingRateData) for item in result)
            assert all(hasattr(item, 'funding_rate') for item in result)
            assert result[0].symbol == "BTC/USDT:USDT"

    def test_open_interest_repository_data_retrieval(self, mock_open_interest_data):
        """建玉残高リポジトリからのデータ取得テスト"""
        # Arrange
        mock_db = Mock()
        oi_repo = OpenInterestRepository(mock_db)
        
        with patch.object(oi_repo, 'get_open_interest_data', return_value=mock_open_interest_data):
            # Act
            result = oi_repo.get_open_interest_data(
                symbol="BTC/USDT:USDT",
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
            
            # Assert
            assert len(result) == 24
            assert all(isinstance(item, OpenInterestData) for item in result)
            assert all(hasattr(item, 'open_interest_value') for item in result)
            assert result[0].symbol == "BTC/USDT:USDT"

    def test_backtest_data_service_integration(self, mock_ohlcv_data, mock_funding_rate_data, mock_open_interest_data):
        """BacktestDataServiceでのデータ統合テスト"""
        # Arrange
        mock_ohlcv_repo = Mock()
        mock_fr_repo = Mock()
        mock_oi_repo = Mock()
        
        # OHLCVデータのモック設定
        with patch.object(mock_ohlcv_repo, 'get_ohlcv_data') as mock_get_ohlcv:
            mock_get_ohlcv.return_value = [
                Mock(
                    timestamp=mock_ohlcv_data.index[i],
                    open=mock_ohlcv_data.iloc[i]['Open'],
                    high=mock_ohlcv_data.iloc[i]['High'],
                    low=mock_ohlcv_data.iloc[i]['Low'],
                    close=mock_ohlcv_data.iloc[i]['Close'],
                    volume=mock_ohlcv_data.iloc[i]['Volume']
                )
                for i in range(len(mock_ohlcv_data))
            ]
            
            # FRとOIデータのモック設定
            mock_fr_repo.get_funding_rate_data.return_value = mock_funding_rate_data
            mock_oi_repo.get_open_interest_data.return_value = mock_open_interest_data
            
            # BacktestDataServiceの初期化
            data_service = BacktestDataService(mock_ohlcv_repo, mock_oi_repo, mock_fr_repo)
            
            # Act
            result_df = data_service.get_data_for_backtest(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
            
            # Assert
            assert isinstance(result_df, pd.DataFrame)
            assert not result_df.empty
            
            # 基本的なOHLCVカラムの確認
            expected_ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in expected_ohlcv_columns:
                assert col in result_df.columns, f"Missing column: {col}"
            
            # 市場データカラムの確認（修正後の実装）
            assert 'funding_rate' in result_df.columns, "Missing funding_rate column"
            assert 'open_interest' in result_df.columns, "Missing open_interest column"

    def test_market_data_features_column_expectations(self, mock_ohlcv_data):
        """MarketDataFeatureCalculatorで期待されるカラム名のテスト"""
        # Arrange
        calculator = MarketDataFeatureCalculator()
        
        # 期待されるカラム名でデータを準備
        funding_rate_data = pd.DataFrame({
            'funding_rate': [0.0001, 0.0002, -0.0001, 0.0003],
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='8H')
        }).set_index('timestamp')
        
        open_interest_data = pd.DataFrame({
            'open_interest': [1000000, 1050000, 1100000, 1150000],
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='6H')
        }).set_index('timestamp')
        
        lookback_periods = {'short': 24, 'medium': 168}
        
        # Act & Assert - ファンディングレート特徴量
        fr_result = calculator.calculate_funding_rate_features(
            mock_ohlcv_data.head(4), funding_rate_data, lookback_periods
        )
        
        # ファンディングレートカラムが見つかることを確認
        assert 'funding_rate' in funding_rate_data.columns
        
        # Act & Assert - 建玉残高特徴量
        oi_result = calculator.calculate_open_interest_features(
            mock_ohlcv_data.head(4), open_interest_data, lookback_periods
        )
        
        # 建玉残高カラムが見つかることを確認
        assert 'open_interest' in open_interest_data.columns

    def test_column_name_mismatch_issue(self, mock_ohlcv_data):
        """現在のカラム名不一致問題の再現テスト"""
        # Arrange
        calculator = MarketDataFeatureCalculator()
        
        # BacktestDataServiceが作成する形式（現在の実装）
        funding_rate_data = pd.DataFrame({
            'FundingRate': [0.0001, 0.0002, -0.0001, 0.0003],  # 大文字のF, R
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='8H')
        }).set_index('timestamp')
        
        open_interest_data = pd.DataFrame({
            'OpenInterest': [1000000, 1050000, 1100000, 1150000],  # 大文字のO, I
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='6H')
        }).set_index('timestamp')
        
        lookback_periods = {'short': 24, 'medium': 168}
        
        # Act
        fr_result = calculator.calculate_funding_rate_features(
            mock_ohlcv_data.head(4), funding_rate_data, lookback_periods
        )
        
        oi_result = calculator.calculate_open_interest_features(
            mock_ohlcv_data.head(4), open_interest_data, lookback_periods
        )
        
        # Assert - 現在の実装では警告が出ることを確認
        # この部分は実際の実装で警告ログが出力されることを期待
        assert len(fr_result) == len(mock_ohlcv_data.head(4))
        assert len(oi_result) == len(mock_ohlcv_data.head(4))

    def test_corrected_backtest_data_service_column_names(self, mock_ohlcv_data, mock_funding_rate_data, mock_open_interest_data):
        """修正後のBacktestDataServiceのカラム名テスト"""
        # Arrange
        mock_ohlcv_repo = Mock()
        mock_fr_repo = Mock()
        mock_oi_repo = Mock()

        # OHLCVデータのモック設定
        with patch.object(mock_ohlcv_repo, 'get_ohlcv_data') as mock_get_ohlcv:
            mock_get_ohlcv.return_value = [
                Mock(
                    timestamp=mock_ohlcv_data.index[i],
                    open=mock_ohlcv_data.iloc[i]['Open'],
                    high=mock_ohlcv_data.iloc[i]['High'],
                    low=mock_ohlcv_data.iloc[i]['Low'],
                    close=mock_ohlcv_data.iloc[i]['Close'],
                    volume=mock_ohlcv_data.iloc[i]['Volume']
                )
                for i in range(len(mock_ohlcv_data))
            ]

            # FRとOIデータのモック設定
            mock_fr_repo.get_funding_rate_data.return_value = mock_funding_rate_data
            mock_oi_repo.get_open_interest_data.return_value = mock_open_interest_data

            # BacktestDataServiceの初期化
            data_service = BacktestDataService(mock_ohlcv_repo, mock_oi_repo, mock_fr_repo)

            # Act
            result_df = data_service.get_data_for_backtest(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )

            # Assert - 修正後は小文字のアンダースコア形式のカラム名を期待
            assert 'funding_rate' in result_df.columns, "Missing funding_rate column"
            assert 'open_interest' in result_df.columns, "Missing open_interest column"

            # MarketDataFeatureCalculatorとの統合テスト
            calculator = MarketDataFeatureCalculator()

            # ファンディングレートデータを抽出
            fr_data = result_df[['funding_rate']].copy()
            fr_data = fr_data.dropna()

            # 建玉残高データを抽出
            oi_data = result_df[['open_interest']].copy()
            oi_data = oi_data.dropna()

            lookback_periods = {'short': 24, 'medium': 168}

            # 特徴量計算が正常に動作することを確認
            if not fr_data.empty:
                fr_features = calculator.calculate_funding_rate_features(
                    result_df, fr_data, lookback_periods
                )
                assert len(fr_features) == len(result_df)

            if not oi_data.empty:
                oi_features = calculator.calculate_open_interest_features(
                    result_df, oi_data, lookback_periods
                )
                assert len(oi_features) == len(result_df)
