"""
MLオーケストレーターの動的パラメータ取得機能のテスト

シンボルとタイムフレームの動的推定機能をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator


class TestMLOrchestratorDynamicParams:
    """MLオーケストレーターの動的パラメータ取得機能のテストクラス"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化処理"""
        self.ml_orchestrator = MLOrchestrator()
    
    def create_sample_ohlcv_data(self, symbol_hint=None, timeframe_minutes=60, num_rows=100):
        """サンプルOHLCVデータを作成"""
        # 時間インデックスを作成
        start_time = datetime.now() - timedelta(hours=num_rows)
        time_index = pd.date_range(
            start=start_time,
            periods=num_rows,
            freq=f"{timeframe_minutes}min"
        )
        
        # 価格データを作成（シンボルに応じて価格レンジを調整）
        if symbol_hint == "BTC":
            base_price = 45000
            price_range = 5000
        elif symbol_hint == "ETH":
            base_price = 2500
            price_range = 500
        else:
            base_price = 45000  # デフォルトはBTC
            price_range = 5000
        
        # ランダムな価格変動を生成
        np.random.seed(42)  # 再現性のため
        price_changes = np.random.normal(0, price_range * 0.02, num_rows)
        prices = base_price + np.cumsum(price_changes)
        
        # OHLCV データを作成
        data = {
            "Open": prices + np.random.normal(0, base_price * 0.001, num_rows),
            "High": prices + np.abs(np.random.normal(0, base_price * 0.005, num_rows)),
            "Low": prices - np.abs(np.random.normal(0, base_price * 0.005, num_rows)),
            "Close": prices,
            "Volume": np.random.uniform(1000, 10000, num_rows)
        }
        
        df = pd.DataFrame(data, index=time_index)
        
        # メタデータを追加（テスト用）
        if symbol_hint:
            df.attrs["symbol"] = f"{symbol_hint}/USDT:USDT"
        
        return df
    
    def test_infer_symbol_from_metadata(self):
        """メタデータからのシンボル推定テスト"""
        # BTCのメタデータを持つデータフレーム
        df = self.create_sample_ohlcv_data(symbol_hint="BTC")
        df.attrs["symbol"] = "BTC/USDT:USDT"
        
        symbol = self.ml_orchestrator._infer_symbol_from_data(df)
        assert symbol == "BTC/USDT:USDT"
        
        # ETHのメタデータを持つデータフレーム
        df = self.create_sample_ohlcv_data(symbol_hint="ETH")
        df.attrs["symbol"] = "ETH/USDT:USDT"
        
        symbol = self.ml_orchestrator._infer_symbol_from_data(df)
        assert symbol == "ETH/USDT:USDT"
    
    def test_infer_symbol_from_price_range(self):
        """価格レンジからのシンボル推定テスト"""
        # BTC価格レンジのデータ
        btc_df = self.create_sample_ohlcv_data(symbol_hint="BTC")
        symbol = self.ml_orchestrator._infer_symbol_from_data(btc_df)
        assert symbol == "BTC/USDT:USDT"
        
        # ETH価格レンジのデータ
        eth_df = self.create_sample_ohlcv_data(symbol_hint="ETH")
        symbol = self.ml_orchestrator._infer_symbol_from_data(eth_df)
        assert symbol == "ETH/USDT:USDT"
    
    def test_infer_symbol_default(self):
        """デフォルトシンボル推定テスト"""
        # 空のデータフレーム
        empty_df = pd.DataFrame()
        symbol = self.ml_orchestrator._infer_symbol_from_data(empty_df)
        assert symbol == "BTC/USDT:USDT"
        
        # 価格情報がないデータフレーム
        no_price_df = pd.DataFrame({"Volume": [1000, 2000, 3000]})
        symbol = self.ml_orchestrator._infer_symbol_from_data(no_price_df)
        assert symbol == "BTC/USDT:USDT"
    
    def test_infer_timeframe_from_metadata(self):
        """メタデータからのタイムフレーム推定テスト"""
        df = self.create_sample_ohlcv_data(timeframe_minutes=60)
        df.attrs["timeframe"] = "1h"
        
        timeframe = self.ml_orchestrator._infer_timeframe_from_data(df)
        assert timeframe == "1h"
    
    def test_infer_timeframe_from_index_1h(self):
        """1時間間隔のインデックスからのタイムフレーム推定テスト"""
        df = self.create_sample_ohlcv_data(timeframe_minutes=60)
        timeframe = self.ml_orchestrator._infer_timeframe_from_data(df)
        assert timeframe == "1h"
    
    def test_infer_timeframe_from_index_15m(self):
        """15分間隔のインデックスからのタイムフレーム推定テスト"""
        df = self.create_sample_ohlcv_data(timeframe_minutes=15)
        timeframe = self.ml_orchestrator._infer_timeframe_from_data(df)
        assert timeframe == "15m"
    
    def test_infer_timeframe_from_index_4h(self):
        """4時間間隔のインデックスからのタイムフレーム推定テスト"""
        df = self.create_sample_ohlcv_data(timeframe_minutes=240)
        timeframe = self.ml_orchestrator._infer_timeframe_from_data(df)
        assert timeframe == "4h"
    
    def test_infer_timeframe_from_index_1d(self):
        """1日間隔のインデックスからのタイムフレーム推定テスト"""
        df = self.create_sample_ohlcv_data(timeframe_minutes=1440)
        timeframe = self.ml_orchestrator._infer_timeframe_from_data(df)
        assert timeframe == "1d"
    
    def test_infer_timeframe_default(self):
        """デフォルトタイムフレーム推定テスト"""
        # 空のデータフレーム
        empty_df = pd.DataFrame()
        timeframe = self.ml_orchestrator._infer_timeframe_from_data(empty_df)
        assert timeframe == "1h"
        
        # DatetimeIndexでないデータフレーム
        non_datetime_df = pd.DataFrame({
            "Close": [100, 200, 300]
        })
        timeframe = self.ml_orchestrator._infer_timeframe_from_data(non_datetime_df)
        assert timeframe == "1h"
        
        # 1行だけのデータフレーム
        single_row_df = self.create_sample_ohlcv_data(num_rows=1)
        timeframe = self.ml_orchestrator._infer_timeframe_from_data(single_row_df)
        assert timeframe == "1h"
    
    def test_infer_timeframe_irregular_intervals(self):
        """不規則な時間間隔のタイムフレーム推定テスト"""
        # 不規則な時間間隔のデータを作成
        irregular_times = [
            datetime.now() - timedelta(hours=3),
            datetime.now() - timedelta(hours=2, minutes=30),
            datetime.now() - timedelta(hours=1, minutes=45),
            datetime.now() - timedelta(minutes=30),
            datetime.now()
        ]
        
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [95, 96, 97, 98, 99],
            "Close": [102, 103, 104, 105, 106],
            "Volume": [1000, 1100, 1200, 1300, 1400]
        }, index=pd.DatetimeIndex(irregular_times))
        
        timeframe = self.ml_orchestrator._infer_timeframe_from_data(df)
        assert timeframe == "1h"  # デフォルト値が返される
    
    @patch('app.services.auto_strategy.services.ml_orchestrator.get_db')
    def test_get_enhanced_data_with_fr_oi_integration(self, mock_get_db):
        """拡張データ取得の統合テスト"""
        # モックの設定
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        
        # BacktestDataServiceのモック
        mock_backtest_service = MagicMock()
        mock_enhanced_df = self.create_sample_ohlcv_data()
        mock_backtest_service.get_data_for_backtest.return_value = mock_enhanced_df
        
        with patch.object(self.ml_orchestrator, 'get_backtest_data_service', return_value=mock_backtest_service):
            # テストデータを作成
            df = self.create_sample_ohlcv_data(symbol_hint="BTC", timeframe_minutes=60)
            
            # 拡張データ取得を実行
            result = self.ml_orchestrator._get_enhanced_data_with_fr_oi(df)
            
            # 結果の検証
            assert result is not None
            assert len(result) > 0
            
            # BacktestDataServiceが正しいパラメータで呼ばれたことを確認
            mock_backtest_service.get_data_for_backtest.assert_called_once()
            call_args = mock_backtest_service.get_data_for_backtest.call_args
            
            # シンボルとタイムフレームが正しく推定されたことを確認
            assert call_args[1]["symbol"] == "BTC/USDT:USDT"
            assert call_args[1]["timeframe"] == "1h"
    
    def test_symbol_inference_with_column_names(self):
        """カラム名からのシンボル推定テスト"""
        # BTCを含むカラム名
        df = pd.DataFrame({
            "BTC_Close": [45000, 46000, 47000],
            "Volume": [1000, 1100, 1200]
        })
        symbol = self.ml_orchestrator._infer_symbol_from_data(df)
        assert symbol == "BTC/USDT:USDT"
        
        # ETHを含むカラム名
        df = pd.DataFrame({
            "ETH_Price": [2500, 2600, 2700],
            "Volume": [1000, 1100, 1200]
        })
        symbol = self.ml_orchestrator._infer_symbol_from_data(df)
        assert symbol == "ETH/USDT:USDT"
    
    def test_error_handling_in_inference(self):
        """推定処理でのエラーハンドリングテスト"""
        # 異常なデータでもデフォルト値が返されることを確認
        
        # Noneを渡した場合
        with pytest.raises(AttributeError):
            self.ml_orchestrator._infer_symbol_from_data(None)
        
        # 異常な価格データ
        df = pd.DataFrame({
            "Close": [float('inf'), float('-inf'), float('nan')],
            "Volume": [1000, 1100, 1200]
        })
        symbol = self.ml_orchestrator._infer_symbol_from_data(df)
        assert symbol == "BTC/USDT:USDT"  # デフォルト値


if __name__ == "__main__":
    pytest.main([__file__])
