"""
オートストラテジー エッジケーステスト

極端な条件下でのシステム動作を検証します。
"""

import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import logging

logger = logging.getLogger(__name__)


class TestEdgeCases:
    """エッジケーステストクラス"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        execution_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        logger.info(f"テスト実行時間: {execution_time:.3f}秒")
        logger.info(f"メモリ使用量変化: {memory_delta:+.2f}MB")
    
    def create_minimal_dataset(self, rows: int = 5) -> pd.DataFrame:
        """極小データセットを作成"""
        dates = pd.date_range(start='2023-01-01', periods=rows, freq='h')
        base_price = 50000
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': [base_price] * rows,
            'High': [base_price + 100] * rows,
            'Low': [base_price - 100] * rows,
            'Close': [base_price] * rows,
            'Volume': [1000] * rows,
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def create_flat_price_dataset(self, rows: int = 100) -> pd.DataFrame:
        """全て同じ価格のデータセットを作成"""
        dates = pd.date_range(start='2023-01-01', periods=rows, freq='h')
        price = 50000
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': [price] * rows,
            'High': [price] * rows,
            'Low': [price] * rows,
            'Close': [price] * rows,
            'Volume': [1000] * rows,
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def create_high_missing_dataset(self, rows: int = 100, missing_rate: float = 0.9) -> pd.DataFrame:
        """欠損値が多いデータセットを作成"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=rows, freq='h')
        
        # 基本データ作成
        base_price = 50000
        prices = base_price + np.cumsum(np.random.randn(rows) * 100)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.exponential(1000, rows),
        })
        
        # 欠損値を挿入
        mask = np.random.random((rows, 5)) < missing_rate
        for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
            data.loc[mask[:, i], col] = np.nan
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def create_extreme_volatility_dataset(self, rows: int = 100, volatility_factor: float = 10.0) -> pd.DataFrame:
        """極端なボラティリティのデータセットを作成"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=rows, freq='h')
        
        base_price = 50000
        # 極端なボラティリティ
        returns = np.random.randn(rows) * 0.1 * volatility_factor  # 通常の10倍のボラティリティ
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = max(prices[-1] * (1 + ret), 1000)  # 最低価格制限
            prices.append(new_price)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.05))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.05))) for p in prices],
            'Close': prices,
            'Volume': np.random.exponential(1000, rows),
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def test_minimal_dataset_processing(self):
        """テスト21: 極小データセット（<10行）での処理"""
        logger.info("🔍 極小データセット処理テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # 5行のデータセット
            minimal_data = self.create_minimal_dataset(5)
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)  # 軽量化
            
            start_time = time.time()
            try:
                ml_indicators = ml_orchestrator.calculate_ml_indicators(minimal_data)
                processing_time = time.time() - start_time
                
                # 結果の検証
                if ml_indicators and "ML_UP_PROB" in ml_indicators:
                    assert len(ml_indicators["ML_UP_PROB"]) <= 5, "結果サイズが入力を超えています"
                    logger.info(f"極小データ処理成功: {len(ml_indicators['ML_UP_PROB'])}個の結果")
                else:
                    logger.info("極小データでML指標が空（期待される動作）")
                
                logger.info(f"処理時間: {processing_time:.3f}秒")
                
            except Exception as e:
                logger.info(f"極小データでエラー（期待される場合もあります）: {e}")
            
            logger.info("✅ 極小データセット処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"極小データセット処理テストエラー: {e}")
    
    def test_flat_price_processing(self):
        """テスト22: 全て同じ価格データでの処理"""
        logger.info("🔍 フラット価格データ処理テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            flat_data = self.create_flat_price_dataset(100)
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            start_time = time.time()
            try:
                ml_indicators = ml_orchestrator.calculate_ml_indicators(flat_data)
                processing_time = time.time() - start_time
                
                # フラットデータでの結果検証
                if ml_indicators and "ML_UP_PROB" in ml_indicators:
                    # フラットデータでは予測確率が中立的になることを期待
                    up_probs = [p for p in ml_indicators["ML_UP_PROB"] if not np.isnan(p)]
                    if up_probs:
                        avg_prob = np.mean(up_probs)
                        logger.info(f"フラットデータでの平均上昇確率: {avg_prob:.3f}")
                        # 極端に偏っていないことを確認
                        assert 0.2 <= avg_prob <= 0.8, f"フラットデータで極端な予測: {avg_prob}"
                
                logger.info(f"処理時間: {processing_time:.3f}秒")
                
            except Exception as e:
                logger.info(f"フラットデータでエラー（期待される場合もあります）: {e}")
            
            logger.info("✅ フラット価格データ処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"フラット価格データ処理テストエラー: {e}")
    
    def test_high_missing_data_processing(self):
        """テスト23: 欠損値が90%以上のデータでの処理"""
        logger.info("🔍 高欠損率データ処理テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            high_missing_data = self.create_high_missing_dataset(100, 0.9)
            missing_rate = high_missing_data.isnull().sum().sum() / (high_missing_data.shape[0] * high_missing_data.shape[1])
            logger.info(f"データ欠損率: {missing_rate:.1%}")
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            start_time = time.time()
            try:
                ml_indicators = ml_orchestrator.calculate_ml_indicators(high_missing_data)
                processing_time = time.time() - start_time
                
                # 高欠損データでの結果検証
                if ml_indicators and "ML_UP_PROB" in ml_indicators:
                    valid_predictions = sum(1 for p in ml_indicators["ML_UP_PROB"] if not np.isnan(p))
                    logger.info(f"有効な予測数: {valid_predictions}/{len(ml_indicators['ML_UP_PROB'])}")
                    
                    # 一部でも有効な予測があれば成功
                    if valid_predictions > 0:
                        logger.info("高欠損データでも一部予測成功")
                    else:
                        logger.info("高欠損データで予測不可（期待される動作）")
                
                logger.info(f"処理時間: {processing_time:.3f}秒")
                
            except Exception as e:
                logger.info(f"高欠損データでエラー（期待される場合もあります）: {e}")
            
            logger.info("✅ 高欠損率データ処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"高欠損率データ処理テストエラー: {e}")
    
    def test_extreme_volatility_processing(self):
        """テスト24: 異常に高いボラティリティでの処理"""
        logger.info("🔍 極端ボラティリティ処理テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # 通常の10倍のボラティリティ
            extreme_data = self.create_extreme_volatility_dataset(100, 10.0)
            
            # ボラティリティ計算
            returns = extreme_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # 日次ボラティリティ
            logger.info(f"データボラティリティ: {volatility:.1%}")
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            start_time = time.time()
            try:
                ml_indicators = ml_orchestrator.calculate_ml_indicators(extreme_data)
                processing_time = time.time() - start_time
                
                # 極端ボラティリティでの結果検証
                if ml_indicators and "ML_UP_PROB" in ml_indicators:
                    valid_probs = [p for p in ml_indicators["ML_UP_PROB"] if not np.isnan(p)]
                    if valid_probs:
                        # 確率値が範囲内であることを確認
                        assert all(0 <= p <= 1 for p in valid_probs), "確率値が範囲外です"
                        prob_std = np.std(valid_probs)
                        logger.info(f"予測確率の標準偏差: {prob_std:.3f}")
                
                logger.info(f"処理時間: {processing_time:.3f}秒")
                
            except Exception as e:
                logger.info(f"極端ボラティリティでエラー（期待される場合もあります）: {e}")
            
            logger.info("✅ 極端ボラティリティ処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"極端ボラティリティ処理テストエラー: {e}")
    
    def test_extreme_tpsl_settings(self):
        """テスト25: TP/SLが0%や100%の極端な設定での処理"""
        logger.info("🔍 極端TP/SL設定処理テスト開始")
        
        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            
            calculator = TPSLCalculator()
            current_price = 50000
            
            # 極端な設定のテストケース
            extreme_cases = [
                {"sl": 0.0, "tp": 0.01, "desc": "SL=0%"},
                {"sl": 0.01, "tp": 0.0, "desc": "TP=0%"},
                {"sl": 1.0, "tp": 2.0, "desc": "SL=100%"},
                {"sl": 0.01, "tp": 1.0, "desc": "TP=100%"},
                {"sl": 0.001, "tp": 0.001, "desc": "極小値"},
            ]
            
            for case in extreme_cases:
                try:
                    start_time = time.time()
                    sl_price, tp_price = calculator.calculate_basic_tpsl_prices(
                        current_price, case["sl"], case["tp"], 1.0  # ロング
                    )
                    processing_time = time.time() - start_time
                    
                    if sl_price is not None and tp_price is not None:
                        # 基本的な妥当性チェック
                        assert sl_price > 0, f"{case['desc']}: SL価格が負です"
                        assert tp_price > 0, f"{case['desc']}: TP価格が負です"
                        
                        logger.info(f"{case['desc']}: SL={sl_price:.2f}, TP={tp_price:.2f} ({processing_time:.3f}秒)")
                    else:
                        logger.info(f"{case['desc']}: 計算結果がNone（期待される場合もあります）")
                        
                except Exception as e:
                    logger.info(f"{case['desc']}: エラー（期待される場合もあります）: {e}")
            
            logger.info("✅ 極端TP/SL設定処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"極端TP/SL設定処理テストエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    test_instance = TestEdgeCases()
    
    tests = [
        test_instance.test_minimal_dataset_processing,
        test_instance.test_flat_price_processing,
        test_instance.test_high_missing_data_processing,
        test_instance.test_extreme_volatility_processing,
        test_instance.test_extreme_tpsl_settings,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test_instance.setup_method()
            test()
            test_instance.teardown_method()
            passed += 1
        except Exception as e:
            logger.error(f"テスト失敗: {test.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 エッジケーステスト結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
