"""
エンドツーエンドのカラム名修正テスト

実際のMLトレーニングフローでカラム名の修正が正しく動作することを確認します。
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
import logging

from app.core.services.backtest_data_service import BacktestDataService
from app.core.services.ml.feature_engineering.market_data_features import MarketDataFeatureCalculator
from database.models import FundingRateData, OpenInterestData, OHLCVData


class TestEndToEndColumnFix:
    """エンドツーエンドのカラム名修正テストクラス"""

    @pytest.fixture
    def sample_data(self):
        """サンプルデータの準備"""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # OHLCVデータ
        ohlcv_data = [
            OHLCVData(
                id=i,
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                timestamp=base_time + timedelta(hours=i),
                open=50000 + i * 100,
                high=50100 + i * 100,
                low=49900 + i * 100,
                close=50000 + i * 100,
                volume=1000 + i * 10
            )
            for i in range(24)
        ]
        
        # ファンディングレートデータ（8時間間隔）
        funding_rate_data = [
            FundingRateData(
                id=i,
                symbol="BTC/USDT:USDT",
                funding_rate=0.0001 * (i % 10 - 5),
                funding_timestamp=base_time + timedelta(hours=8*i),
                timestamp=base_time + timedelta(hours=8*i)
            )
            for i in range(3)
        ]
        
        # 建玉残高データ（1時間間隔）
        open_interest_data = [
            OpenInterestData(
                id=i,
                symbol="BTC/USDT:USDT",
                open_interest_value=1000000 + i * 50000,
                data_timestamp=base_time + timedelta(hours=i),
                timestamp=base_time + timedelta(hours=i)
            )
            for i in range(24)
        ]
        
        return {
            'ohlcv': ohlcv_data,
            'funding_rate': funding_rate_data,
            'open_interest': open_interest_data
        }

    def test_backtest_data_service_integration(self, sample_data):
        """BacktestDataServiceの統合テスト"""
        # モックリポジトリの設定
        mock_ohlcv_repo = Mock()
        mock_fr_repo = Mock()
        mock_oi_repo = Mock()
        
        mock_ohlcv_repo.get_ohlcv_data.return_value = sample_data['ohlcv']
        mock_fr_repo.get_funding_rate_data.return_value = sample_data['funding_rate']
        mock_oi_repo.get_open_interest_data.return_value = sample_data['open_interest']
        
        # BacktestDataServiceの初期化
        service = BacktestDataService(mock_ohlcv_repo, mock_oi_repo, mock_fr_repo)
        
        # データ取得
        result_df = service.get_data_for_backtest(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        # 正しいカラム名が使用されていることを確認
        assert 'funding_rate' in result_df.columns, "funding_rate column should be present"
        assert 'open_interest' in result_df.columns, "open_interest column should be present"
        
        # 古いカラム名が使用されていないことを確認
        assert 'FundingRate' not in result_df.columns, "Old FundingRate column should not be present"
        assert 'OpenInterest' not in result_df.columns, "Old OpenInterest column should not be present"
        
        # データが正しく統合されていることを確認
        assert len(result_df) == 24
        assert result_df['funding_rate'].notna().any()
        assert result_df['open_interest'].notna().any()

    def test_market_data_features_integration(self, sample_data):
        """MarketDataFeatureCalculatorの統合テスト"""
        # BacktestDataServiceでデータを準備
        mock_ohlcv_repo = Mock()
        mock_fr_repo = Mock()
        mock_oi_repo = Mock()
        
        mock_ohlcv_repo.get_ohlcv_data.return_value = sample_data['ohlcv']
        mock_fr_repo.get_funding_rate_data.return_value = sample_data['funding_rate']
        mock_oi_repo.get_open_interest_data.return_value = sample_data['open_interest']
        
        service = BacktestDataService(mock_ohlcv_repo, mock_oi_repo, mock_fr_repo)
        
        integrated_data = service.get_data_for_backtest(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        # MarketDataFeatureCalculatorで特徴量計算
        calculator = MarketDataFeatureCalculator()
        
        # OHLCVデータを抽出
        ohlcv_data = integrated_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # ファンディングレートと建玉残高データを抽出
        funding_rate_data = integrated_data[['funding_rate']].copy()
        open_interest_data = integrated_data[['open_interest']].copy()
        
        lookback_periods = {'short': 6, 'medium': 12}
        
        # 特徴量計算（警告が出ないことを確認）
        with patch('app.core.services.ml.feature_engineering.market_data_features.logger') as mock_logger:
            fr_features = calculator.calculate_funding_rate_features(
                ohlcv_data, funding_rate_data, lookback_periods
            )
            
            oi_features = calculator.calculate_open_interest_features(
                ohlcv_data, open_interest_data, lookback_periods
            )
            
            composite_features = calculator.calculate_composite_features(
                ohlcv_data, funding_rate_data, open_interest_data, lookback_periods
            )
            
            # 警告ログが呼ばれていないことを確認
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if 'カラムが見つかりません' in str(call)]
            assert len(warning_calls) == 0, f"Unexpected warning calls: {warning_calls}"
        
        # 特徴量が正常に計算されていることを確認
        assert len(fr_features) == len(ohlcv_data)
        assert len(oi_features) == len(ohlcv_data)
        assert len(composite_features) == len(ohlcv_data)

    def test_ml_management_column_detection(self, sample_data, caplog):
        """ml_management.pyのカラム検出テスト"""
        caplog.set_level(logging.INFO)
        
        # BacktestDataServiceでデータを準備
        mock_ohlcv_repo = Mock()
        mock_fr_repo = Mock()
        mock_oi_repo = Mock()
        
        mock_ohlcv_repo.get_ohlcv_data.return_value = sample_data['ohlcv']
        mock_fr_repo.get_funding_rate_data.return_value = sample_data['funding_rate']
        mock_oi_repo.get_open_interest_data.return_value = sample_data['open_interest']
        
        service = BacktestDataService(mock_ohlcv_repo, mock_oi_repo, mock_fr_repo)
        
        training_data = service.get_data_for_backtest(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        # ml_management.pyのカラム検出ロジックをシミュレート（修正後）
        funding_rate_data = None
        open_interest_data = None
        
        # ファンディングレートデータの確認
        if 'funding_rate' in training_data.columns:
            valid_fr_count = training_data['funding_rate'].notna().sum()
            if valid_fr_count > 0:
                funding_rate_data = training_data[['funding_rate']].copy()
        
        # オープンインタレストデータの確認
        if 'open_interest' in training_data.columns:
            valid_oi_count = training_data['open_interest'].notna().sum()
            if valid_oi_count > 0:
                open_interest_data = training_data[['open_interest']].copy()
        
        # データが正しく検出されることを確認
        assert funding_rate_data is not None, "Funding rate data should be detected"
        assert open_interest_data is not None, "Open interest data should be detected"
        
        # 「カラムが存在しません」というメッセージが出ていないことを確認
        log_messages = [record.message for record in caplog.records]
        missing_column_messages = [msg for msg in log_messages 
                                 if 'カラムが存在しません' in msg]
        
        assert len(missing_column_messages) == 0, f"Unexpected missing column messages: {missing_column_messages}"

    def test_complete_workflow_no_warnings(self, sample_data, caplog):
        """完全なワークフローで警告が出ないことを確認"""
        caplog.set_level(logging.WARNING)
        
        # 1. BacktestDataServiceでデータ統合
        mock_ohlcv_repo = Mock()
        mock_fr_repo = Mock()
        mock_oi_repo = Mock()
        
        mock_ohlcv_repo.get_ohlcv_data.return_value = sample_data['ohlcv']
        mock_fr_repo.get_funding_rate_data.return_value = sample_data['funding_rate']
        mock_oi_repo.get_open_interest_data.return_value = sample_data['open_interest']
        
        service = BacktestDataService(mock_ohlcv_repo, mock_oi_repo, mock_fr_repo)
        
        training_data = service.get_data_for_backtest(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        # 2. ml_management.pyのデータ検出
        funding_rate_data = None
        open_interest_data = None
        
        if 'funding_rate' in training_data.columns:
            valid_fr_count = training_data['funding_rate'].notna().sum()
            if valid_fr_count > 0:
                funding_rate_data = training_data[['funding_rate']].copy()
        
        if 'open_interest' in training_data.columns:
            valid_oi_count = training_data['open_interest'].notna().sum()
            if valid_oi_count > 0:
                open_interest_data = training_data[['open_interest']].copy()
        
        # 3. MarketDataFeatureCalculatorで特徴量計算
        calculator = MarketDataFeatureCalculator()
        ohlcv_data = training_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        lookback_periods = {'short': 6, 'medium': 12}
        
        fr_features = calculator.calculate_funding_rate_features(
            ohlcv_data, funding_rate_data, lookback_periods
        )
        
        oi_features = calculator.calculate_open_interest_features(
            ohlcv_data, open_interest_data, lookback_periods
        )
        
        # 警告メッセージが出ていないことを確認
        warning_messages = [record.message for record in caplog.records 
                          if record.levelno >= logging.WARNING]
        
        column_related_warnings = [msg for msg in warning_messages 
                                 if 'カラムが見つかりません' in msg or 'カラムが存在しません' in msg]
        
        assert len(column_related_warnings) == 0, f"Unexpected column-related warnings: {column_related_warnings}"
        
        # 結果が正常に生成されていることを確認
        assert len(fr_features) == len(ohlcv_data)
        assert len(oi_features) == len(ohlcv_data)
        assert funding_rate_data is not None
        assert open_interest_data is not None
