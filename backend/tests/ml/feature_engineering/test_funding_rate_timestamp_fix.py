import pandas as pd
import pytest
from app.services.ml.feature_engineering.funding_rate_features import FundingRateFeatureCalculator

class TestFundingRateFeatureCalculator:
    def test_timestamp_ambiguity_fix(self):
        """
        timestampがインデックスとカラムの両方に存在する場合の重複エラー修正をテスト
        """
        # テストデータ作成 (OHLCV)
        dates = pd.date_range(start="2023-01-01", periods=24, freq="1h")
        ohlcv_df = pd.DataFrame({
            "open": [100.0] * 24,
            "high": [105.0] * 24,
            "low": [95.0] * 24,
            "close": [102.0] * 24,
            "volume": [1000] * 24,
            "timestamp": dates  # カラムとして存在
        }, index=dates)  # インデックスとしても存在
        
        # テストデータ作成 (Funding Rate)
        fr_dates = pd.date_range(start="2023-01-01", periods=3, freq="8h")
        funding_df = pd.DataFrame({
            "timestamp": fr_dates,
            "funding_rate": [0.0001, 0.0002, 0.0001]
        })
        
        calculator = FundingRateFeatureCalculator()
        
        # エラーが発生せずに実行できることを確認
        try:
            result_df = calculator.calculate_features(ohlcv_df, funding_df)
        except Exception as e:
            pytest.fail(f"calculate_features raised Exception: {e}")
            
        # timestampカラムが1つだけ存在することを確認 (重複していないこと)
        # DataFrame.columnsに重複がある場合は len(df.columns) != len(set(df.columns)) になるわけではないが、
        # ここではマージ後の挙動としてエラーが出ないことが最重要
        # 実装が変更されてfr_lag_3pのみが生成されるようになった
        assert "fr_lag_3p" in result_df.columns
        assert not result_df.columns.duplicated().any()

    def test_missing_timestamp_column(self):
        """
        timestampカラムがなく、インデックスのみの場合の自動生成をテスト
        """
        dates = pd.date_range(start="2023-01-01", periods=24, freq="1h")
        ohlcv_df = pd.DataFrame({
            "open": [100.0] * 24,
            "close": [102.0] * 24,
            "volume": [1000] * 24,
        }, index=dates)
        
        fr_dates = pd.date_range(start="2023-01-01", periods=3, freq="8h")
        funding_df = pd.DataFrame({
            "timestamp": fr_dates,
            "funding_rate": [0.0001, 0.0002, 0.0001]
        })
        
        calculator = FundingRateFeatureCalculator()
        result_df = calculator.calculate_features(ohlcv_df, funding_df)
        
        # timestampカラムはインデックスに設定される仕様のため、インデックスの型を確認
        assert isinstance(result_df.index, pd.DatetimeIndex)
        # 実装が変更されてfr_lag_3pのみが生成されるようになった
        assert "fr_lag_3p" in result_df.columns




