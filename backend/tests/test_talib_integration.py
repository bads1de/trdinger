"""
TA-lib統合テスト

technical_features.pyとenhanced_crypto_features.pyの
TA-lib統合修正が正常に動作することを確認するテストファイル
"""

import logging
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.ml.feature_engineering.technical_features import TechnicalFeatureCalculator
from app.services.ml.feature_engineering.enhanced_crypto_features import EnhancedCryptoFeatures

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTALibIntegration(unittest.TestCase):
    """TA-lib統合テストクラス"""

    def setUp(self):
        """テストデータの準備"""
        # サンプルOHLCVデータを生成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        
        # 現実的な価格データを生成
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # 最低価格を設定
        
        # OHLCV データフレーム作成
        self.test_data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        
        # High >= Low >= 0 を保証
        self.test_data['High'] = np.maximum(self.test_data['High'], self.test_data['Close'])
        self.test_data['Low'] = np.minimum(self.test_data['Low'], self.test_data['Close'])
        
        logger.info(f"テストデータ作成完了: {len(self.test_data)} rows")
        logger.info(f"価格範囲: {self.test_data['Close'].min():.2f} - {self.test_data['Close'].max():.2f}")

    def test_technical_features_calculator(self):
        """TechnicalFeatureCalculatorのテスト"""
        logger.info("TechnicalFeatureCalculatorのテスト開始")
        
        calculator = TechnicalFeatureCalculator()
        config = {
            "lookback_periods": {
                "short_ma": 10,
                "long_ma": 20,
                "volatility": 14
            }
        }
        
        try:
            result = calculator.calculate_features(self.test_data, config)
            
            # 基本的な検証
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(self.test_data))
            
            # TA-lib指標が計算されていることを確認
            expected_columns = [
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'Stochastic_K', 'Stochastic_D', 'Williams_R',
                'CCI', 'ROC', 'Momentum'
            ]
            
            for col in expected_columns:
                self.assertIn(col, result.columns, f"{col} が結果に含まれていません")
                self.assertFalse(result[col].isna().all(), f"{col} がすべてNaNです")
            
            # 値の範囲チェック
            self.assertTrue((result['RSI'] >= 0).all() and (result['RSI'] <= 100).all(), 
                          "RSIの値が範囲外です")
            self.assertTrue((result['Williams_R'] >= -100).all() and (result['Williams_R'] <= 0).all(), 
                          "Williams %Rの値が範囲外です")
            
            logger.info("TechnicalFeatureCalculatorテスト成功")
            
        except Exception as e:
            logger.error(f"TechnicalFeatureCalculatorテストエラー: {e}")
            raise

    def test_enhanced_crypto_features(self):
        """EnhancedCryptoFeaturesのテスト"""
        logger.info("EnhancedCryptoFeaturesのテスト開始")
        
        features = EnhancedCryptoFeatures()
        
        try:
            # テクニカル特徴量の計算をテスト
            result = features._create_technical_features(self.test_data, {"technical": 20})
            
            # 基本的な検証
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(self.test_data))
            
            # TA-lib指標が計算されていることを確認
            expected_patterns = ['rsi_14', 'rsi_24', 'bb_upper_20', 'bb_lower_20', 
                               'bb_upper_48', 'bb_lower_48', 'macd', 'macd_signal', 'macd_histogram']
            
            for pattern in expected_patterns:
                matching_cols = [col for col in result.columns if pattern in col]
                self.assertTrue(len(matching_cols) > 0, f"{pattern} に一致する列が見つかりません")
            
            # RSI値の範囲チェック
            rsi_cols = [col for col in result.columns if col.startswith('rsi_')]
            for col in rsi_cols:
                self.assertTrue((result[col] >= 0).all() and (result[col] <= 100).all(), 
                              f"{col}の値が範囲外です")
            
            # ボリンジャーバンドの関係チェック
            if 'bb_upper_20' in result.columns and 'bb_lower_20' in result.columns:
                self.assertTrue((result['bb_upper_20'] >= result['bb_lower_20']).all(), 
                              "ボリンジャーバンドの上限が下限より小さい箇所があります")
            
            logger.info("EnhancedCryptoFeaturesテスト成功")
            
        except Exception as e:
            logger.error(f"EnhancedCryptoFeaturesテストエラー: {e}")
            raise

    def test_talib_import(self):
        """TA-libライブラリのインポートテスト"""
        logger.info("TA-libインポートテスト開始")
        
        try:
            import talib
            
            # 基本的な関数が利用可能かテスト
            test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20, dtype=np.float64)
            
            rsi = talib.RSI(test_data, timeperiod=14)
            self.assertIsInstance(rsi, np.ndarray)
            
            macd, signal, hist = talib.MACD(test_data)
            self.assertIsInstance(macd, np.ndarray)
            self.assertIsInstance(signal, np.ndarray)
            self.assertIsInstance(hist, np.ndarray)
            
            logger.info("TA-libインポートテスト成功")
            
        except ImportError as e:
            logger.error(f"TA-libインポートエラー: {e}")
            self.fail("TA-libライブラリがインポートできません")
        except Exception as e:
            logger.error(f"TA-lib基本機能テストエラー: {e}")
            raise


def run_tests():
    """テスト実行関数"""
    logger.info("TA-lib統合テスト開始")
    
    # テストスイート作成
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTALibIntegration)
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果サマリー
    if result.wasSuccessful():
        logger.info("すべてのテストが成功しました！")
        return True
    else:
        logger.error(f"テスト失敗: {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
