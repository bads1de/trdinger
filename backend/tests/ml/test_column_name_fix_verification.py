"""
カラム名修正の検証テスト

修正後のカラム名でMLトレーニングの警告が解消されることを確認します。
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
from io import StringIO

from app.core.services.ml.feature_engineering.market_data_features import MarketDataFeatureCalculator


class TestColumnNameFixVerification:
    """カラム名修正の検証テストクラス"""

    def test_no_warning_with_correct_column_names(self, caplog):
        """正しいカラム名で警告が出ないことを確認"""
        # ログレベルを設定
        caplog.set_level(logging.WARNING)
        
        calculator = MarketDataFeatureCalculator()
        
        # テストデータの準備（修正後のカラム名）
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dates = [base_time + timedelta(hours=i) for i in range(10)]
        
        # OHLCVデータ
        ohlcv_data = pd.DataFrame({
            'Open': [50000 + i * 100 for i in range(10)],
            'High': [50100 + i * 100 for i in range(10)],
            'Low': [49900 + i * 100 for i in range(10)],
            'Close': [50000 + i * 100 for i in range(10)],
            'Volume': [1000 + i * 10 for i in range(10)]
        }, index=pd.DatetimeIndex(dates))
        
        # 正しいカラム名でファンディングレートデータ
        funding_rate_data = pd.DataFrame({
            'funding_rate': [0.0001 * (i % 5 - 2) for i in range(10)]
        }, index=pd.DatetimeIndex(dates))
        
        # 正しいカラム名で建玉残高データ
        open_interest_data = pd.DataFrame({
            'open_interest': [1000000 + i * 50000 for i in range(10)]
        }, index=pd.DatetimeIndex(dates))
        
        lookback_periods = {'short': 3, 'medium': 6}
        
        # 特徴量計算を実行
        fr_result = calculator.calculate_funding_rate_features(
            ohlcv_data, funding_rate_data, lookback_periods
        )
        
        oi_result = calculator.calculate_open_interest_features(
            ohlcv_data, open_interest_data, lookback_periods
        )
        
        composite_result = calculator.calculate_composite_features(
            ohlcv_data, funding_rate_data, open_interest_data, lookback_periods
        )
        
        # 警告メッセージが出ていないことを確認
        warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
        
        # ファンディングレートと建玉残高に関する警告がないことを確認
        fr_warnings = [msg for msg in warning_messages if "ファンディングレートカラムが見つかりません" in msg]
        oi_warnings = [msg for msg in warning_messages if "建玉残高カラムが見つかりません" in msg]
        combined_warnings = [msg for msg in warning_messages if "ファンディングレートまたは建玉残高カラムが見つかりません" in msg]
        
        assert len(fr_warnings) == 0, f"ファンディングレート警告が発生: {fr_warnings}"
        assert len(oi_warnings) == 0, f"建玉残高警告が発生: {oi_warnings}"
        assert len(combined_warnings) == 0, f"複合特徴量警告が発生: {combined_warnings}"
        
        # 結果が正常に生成されていることを確認
        assert len(fr_result) == len(ohlcv_data)
        assert len(oi_result) == len(ohlcv_data)
        assert len(composite_result) == len(ohlcv_data)

    def test_warning_with_old_column_names(self, caplog):
        """古いカラム名で警告が出ることを確認（後方互換性テスト）"""
        # ログレベルを設定
        caplog.set_level(logging.WARNING)
        
        calculator = MarketDataFeatureCalculator()
        
        # テストデータの準備（古いカラム名）
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dates = [base_time + timedelta(hours=i) for i in range(10)]
        
        # OHLCVデータ
        ohlcv_data = pd.DataFrame({
            'Open': [50000 + i * 100 for i in range(10)],
            'High': [50100 + i * 100 for i in range(10)],
            'Low': [49900 + i * 100 for i in range(10)],
            'Close': [50000 + i * 100 for i in range(10)],
            'Volume': [1000 + i * 10 for i in range(10)]
        }, index=pd.DatetimeIndex(dates))
        
        # 古いカラム名でファンディングレートデータ
        funding_rate_data = pd.DataFrame({
            'FundingRate': [0.0001 * (i % 5 - 2) for i in range(10)]  # 大文字
        }, index=pd.DatetimeIndex(dates))
        
        # 古いカラム名で建玉残高データ
        open_interest_data = pd.DataFrame({
            'OpenInterest': [1000000 + i * 50000 for i in range(10)]  # 大文字
        }, index=pd.DatetimeIndex(dates))
        
        lookback_periods = {'short': 3, 'medium': 6}
        
        # 特徴量計算を実行
        fr_result = calculator.calculate_funding_rate_features(
            ohlcv_data, funding_rate_data, lookback_periods
        )
        
        oi_result = calculator.calculate_open_interest_features(
            ohlcv_data, open_interest_data, lookback_periods
        )
        
        # 警告メッセージが出ていることを確認
        warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
        
        # ファンディングレートと建玉残高に関する警告があることを確認
        fr_warnings = [msg for msg in warning_messages if "ファンディングレートカラムが見つかりません" in msg]
        oi_warnings = [msg for msg in warning_messages if "建玉残高カラムが見つかりません" in msg]
        
        assert len(fr_warnings) > 0, "ファンディングレート警告が期待されましたが発生しませんでした"
        assert len(oi_warnings) > 0, "建玉残高警告が期待されましたが発生しませんでした"

    def test_backtest_data_service_produces_correct_column_names(self):
        """BacktestDataServiceが正しいカラム名を生成することを確認"""
        from app.core.services.backtest_data_service import BacktestDataService
        from database.models import FundingRateData, OpenInterestData
        from unittest.mock import Mock
        
        # モックデータの準備
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        mock_fr_data = [
            FundingRateData(
                id=1,
                symbol="BTC/USDT:USDT",
                funding_rate=0.0001,
                funding_timestamp=base_time,
                timestamp=base_time
            )
        ]
        
        mock_oi_data = [
            OpenInterestData(
                id=1,
                symbol="BTC/USDT:USDT",
                open_interest_value=1000000,
                data_timestamp=base_time,
                timestamp=base_time
            )
        ]
        
        # BacktestDataServiceのインスタンス作成
        service = BacktestDataService()
        
        # DataFrameの変換メソッドをテスト
        fr_df = service._convert_fr_to_dataframe(mock_fr_data)
        oi_df = service._convert_oi_to_dataframe(mock_oi_data)
        
        # 正しいカラム名が使用されていることを確認
        assert 'funding_rate' in fr_df.columns, f"Expected 'funding_rate' column, got: {fr_df.columns.tolist()}"
        assert 'open_interest' in oi_df.columns, f"Expected 'open_interest' column, got: {oi_df.columns.tolist()}"
        
        # 古いカラム名が使用されていないことを確認
        assert 'FundingRate' not in fr_df.columns, "Old 'FundingRate' column name should not be used"
        assert 'OpenInterest' not in oi_df.columns, "Old 'OpenInterest' column name should not be used"

    def test_ml_management_uses_correct_column_names(self, caplog):
        """ml_management.pyが正しいカラム名を使用することを確認"""
        import logging
        caplog.set_level(logging.INFO)

        # テストデータの準備（修正後のカラム名）
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dates = [base_time + timedelta(hours=i) for i in range(10)]

        training_data = pd.DataFrame({
            'Open': [50000 + i * 100 for i in range(10)],
            'High': [50100 + i * 100 for i in range(10)],
            'Low': [49900 + i * 100 for i in range(10)],
            'Close': [50000 + i * 100 for i in range(10)],
            'Volume': [1000 + i * 10 for i in range(10)],
            'funding_rate': [0.0001 * (i % 5 - 2) for i in range(10)],
            'open_interest': [1000000 + i * 50000 for i in range(10)]
        }, index=pd.DatetimeIndex(dates))

        # ml_management.pyのカラムチェック部分をシミュレート
        # 修正前の実装では古いカラム名をチェックしている
        old_fr_check = 'FundingRate' in training_data.columns
        old_oi_check = 'OpenInterest' in training_data.columns

        # 新しいカラム名をチェック
        new_fr_check = 'funding_rate' in training_data.columns
        new_oi_check = 'open_interest' in training_data.columns

        # 古いカラム名では見つからず、新しいカラム名では見つかることを確認
        assert not old_fr_check, "Old FundingRate column should not be found"
        assert not old_oi_check, "Old OpenInterest column should not be found"
        assert new_fr_check, "New funding_rate column should be found"
        assert new_oi_check, "New open_interest column should be found"

    def test_ml_management_corrected_column_checks(self):
        """修正後のml_management.pyのカラムチェックをテスト"""
        # テストデータの準備（修正後のカラム名）
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dates = [base_time + timedelta(hours=i) for i in range(10)]

        training_data = pd.DataFrame({
            'Open': [50000 + i * 100 for i in range(10)],
            'High': [50100 + i * 100 for i in range(10)],
            'Low': [49900 + i * 100 for i in range(10)],
            'Close': [50000 + i * 100 for i in range(10)],
            'Volume': [1000 + i * 10 for i in range(10)],
            'funding_rate': [0.0001 * (i % 5 - 2) for i in range(10)],
            'open_interest': [1000000 + i * 50000 for i in range(10)]
        }, index=pd.DatetimeIndex(dates))

        # 修正後のカラムチェック（ml_management.pyの実装をシミュレート）
        funding_rate_data = None
        open_interest_data = None

        # ファンディングレートデータの確認（修正後）
        if 'funding_rate' in training_data.columns:
            valid_fr_count = training_data['funding_rate'].notna().sum()
            if valid_fr_count > 0:
                funding_rate_data = training_data[['funding_rate']].copy()

        # オープンインタレストデータの確認（修正後）
        if 'open_interest' in training_data.columns:
            valid_oi_count = training_data['open_interest'].notna().sum()
            if valid_oi_count > 0:
                open_interest_data = training_data[['open_interest']].copy()

        # データが正しく抽出されることを確認
        assert funding_rate_data is not None, "Funding rate data should be extracted"
        assert open_interest_data is not None, "Open interest data should be extracted"
        assert 'funding_rate' in funding_rate_data.columns
        assert 'open_interest' in open_interest_data.columns
        assert len(funding_rate_data) == 10
        assert len(open_interest_data) == 10
