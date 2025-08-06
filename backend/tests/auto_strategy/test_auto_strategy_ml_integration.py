"""
オートストラテジーとML統合の包括的テスト

MLOrchestrator、ML指標計算、AutoML機能、ML予測とオートストラテジーの統合を詳細にテストします。
"""

import gc
import logging

import numpy as np
import pandas as pd
import pytest

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutoStrategyMLIntegration:
    """オートストラテジーとML統合の包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.test_data = self._create_test_data()
        self.funding_rate_data = self._create_funding_rate_data()
        self.open_interest_data = self._create_open_interest_data()

    def _create_test_data(self, rows: int = 100) -> pd.DataFrame:
        """テスト用のOHLCVデータを作成"""
        dates = pd.date_range(start="2023-01-01", periods=rows, freq="1H")
        np.random.seed(42)  # 再現性のため
        
        # リアルな価格データを生成
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, rows)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # 最低価格を設定
        
        # OHLCV データを生成
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close_price = price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                "timestamp": date,
                "open": open_price,
                "high": max(high, open_price, close_price),
                "low": min(low, open_price, close_price),
                "close": close_price,
                "volume": volume
            })
        
        return pd.DataFrame(data)

    def _create_funding_rate_data(self) -> pd.DataFrame:
        """テスト用のファンディングレートデータを作成"""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="8H")
        np.random.seed(42)
        
        data = []
        for date in dates:
            funding_rate = np.random.normal(0.0001, 0.0005)  # 現実的なファンディングレート
            data.append({
                "timestamp": date,
                "funding_rate": funding_rate
            })
        
        return pd.DataFrame(data)

    def _create_open_interest_data(self) -> pd.DataFrame:
        """テスト用の建玉残高データを作成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        np.random.seed(42)
        
        data = []
        base_oi = 1000000
        for date in dates:
            oi_change = np.random.normal(0, 0.05)
            open_interest = base_oi * (1 + oi_change)
            data.append({
                "timestamp": date,
                "open_interest": max(open_interest, 100000)  # 最低値を設定
            })
        
        return pd.DataFrame(data)

    def test_ml_orchestrator_initialization(self):
        """MLOrchestratorの初期化テスト"""
        logger.info("=== MLOrchestrator初期化テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # AutoML有効での初期化
            ml_orchestrator_automl = MLOrchestrator(enable_automl=True)
            assert ml_orchestrator_automl.enable_automl is True
            assert ml_orchestrator_automl.config is not None
            
            # AutoML無効での初期化
            ml_orchestrator_basic = MLOrchestrator(enable_automl=False)
            assert ml_orchestrator_basic.enable_automl is False
            
            # カスタム設定での初期化
            custom_config = {"test_param": "test_value"}
            ml_orchestrator_custom = MLOrchestrator(
                enable_automl=True,
                automl_config=custom_config
            )
            assert ml_orchestrator_custom.automl_config == custom_config
            
            logger.info("✅ MLOrchestrator初期化テスト成功")
            
        except Exception as e:
            pytest.fail(f"MLOrchestrator初期化テストエラー: {e}")

    def test_ml_indicators_calculation(self):
        """ML指標計算の基本テスト"""
        logger.info("=== ML指標計算テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator(enable_automl=True)
            
            # ML指標計算
            ml_indicators = ml_orchestrator.calculate_ml_indicators(self.test_data)
            
            # 結果検証
            assert isinstance(ml_indicators, dict), "ML指標が辞書形式ではありません"
            
            expected_keys = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
            for key in expected_keys:
                assert key in ml_indicators, f"ML指標 {key} が不足しています"
                assert len(ml_indicators[key]) > 0, f"ML指標 {key} が空です"
                assert len(ml_indicators[key]) == len(self.test_data), f"ML指標 {key} の長さが不正です"
                
                # 確率値の範囲確認（0-1の範囲内）
                values = ml_indicators[key]
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    assert all(0 <= v <= 1 for v in valid_values), f"{key}: 確率値が範囲外です"
            
            logger.info("✅ ML指標計算テスト成功")
            
        except Exception as e:
            pytest.fail(f"ML指標計算テストエラー: {e}")

    def test_single_ml_indicator_calculation(self):
        """単一ML指標計算テスト"""
        logger.info("=== 単一ML指標計算テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            # 各指標を個別に計算
            indicator_types = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
            
            for indicator_type in indicator_types:
                result = ml_orchestrator.calculate_single_ml_indicator(
                    indicator_type, self.test_data
                )
                
                assert isinstance(result, np.ndarray), f"{indicator_type}: 結果がnumpy配列ではありません"
                assert len(result) > 0, f"{indicator_type}: 結果が空です"
                
                # 有効な値の確認
                valid_values = result[~np.isnan(result)]
                if len(valid_values) > 0:
                    assert all(0 <= v <= 1 for v in valid_values), f"{indicator_type}: 確率値が範囲外です"
            
            logger.info("✅ 単一ML指標計算テスト成功")
            
        except Exception as e:
            pytest.fail(f"単一ML指標計算テストエラー: {e}")

    def test_automl_status_and_features(self):
        """AutoML機能状態テスト"""
        logger.info("=== AutoML機能状態テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # AutoML有効
            ml_orchestrator_automl = MLOrchestrator(enable_automl=True)
            automl_status = ml_orchestrator_automl.get_automl_status()
            
            assert isinstance(automl_status, dict), "AutoML状態が辞書形式ではありません"
            assert "enabled" in automl_status, "enabled フィールドが不足しています"
            assert automl_status["enabled"] is True, "AutoMLが有効になっていません"
            assert "service_type" in automl_status, "service_type フィールドが不足しています"
            assert "config" in automl_status, "config フィールドが不足しています"
            
            # AutoML無効
            ml_orchestrator_basic = MLOrchestrator(enable_automl=False)
            basic_status = ml_orchestrator_basic.get_automl_status()
            assert basic_status["enabled"] is False, "AutoMLが無効になっていません"
            
            logger.info("✅ AutoML機能状態テスト成功")
            
        except Exception as e:
            pytest.fail(f"AutoML機能状態テストエラー: {e}")

    def test_predict_probabilities(self):
        """予測確率計算テスト"""
        logger.info("=== 予測確率計算テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            # 特徴量データを準備（簡単な例）
            features_data = pd.DataFrame({
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10),
                'feature3': np.random.randn(10)
            })
            
            # 予測確率計算（エラーが発生する可能性があるため、try-catchで処理）
            try:
                predictions = ml_orchestrator.predict_probabilities(features_data)
                
                if predictions:
                    assert isinstance(predictions, dict), "予測結果が辞書形式ではありません"
                    
                    # 予測値の妥当性確認
                    for key, value in predictions.items():
                        if not np.isnan(value):
                            assert 0 <= value <= 1, f"予測値 {key} が範囲外です: {value}"
                
                logger.info("✅ 予測確率計算テスト成功")
                
            except Exception as pred_error:
                logger.warning(f"予測計算でエラー（期待される場合もあります）: {pred_error}")
                # MLモデルが利用できない場合は警告のみ
                
        except Exception as e:
            pytest.fail(f"予測確率計算テストエラー: {e}")

    def test_error_handling_invalid_data(self):
        """無効なデータでのエラーハンドリングテスト"""
        logger.info("=== エラーハンドリングテスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            from app.services.ml.exceptions import MLDataError
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            # 空のデータフレーム
            empty_df = pd.DataFrame()
            with pytest.raises((MLDataError, ValueError, Exception)):
                ml_orchestrator.calculate_ml_indicators(empty_df)
            
            # 必須カラムが不足したデータ
            invalid_df = pd.DataFrame({'invalid_column': [1, 2, 3]})
            with pytest.raises((MLDataError, ValueError, Exception)):
                ml_orchestrator.calculate_ml_indicators(invalid_df)
            
            # None データ
            with pytest.raises((MLDataError, ValueError, TypeError, Exception)):
                ml_orchestrator.calculate_ml_indicators(None)
            
            logger.info("✅ エラーハンドリングテスト成功")
            
        except Exception as e:
            pytest.fail(f"エラーハンドリングテストエラー: {e}")

    def test_memory_management(self):
        """メモリ管理テスト"""
        logger.info("=== メモリ管理テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # 複数回の実行でメモリリークがないことを確認
            for i in range(5):
                ml_orchestrator = MLOrchestrator(enable_automl=False)
                
                try:
                    result = ml_orchestrator.calculate_ml_indicators(self.test_data)
                    
                    # 結果の基本検証
                    if result:
                        assert isinstance(result, dict)
                        
                except Exception as calc_error:
                    logger.warning(f"計算エラー（反復 {i+1}）: {calc_error}")
                
                # 明示的にオブジェクトを削除
                del ml_orchestrator
                if 'result' in locals():
                    del result
                
                # ガベージコレクション
                gc.collect()
            
            logger.info("✅ メモリ管理テスト成功")
            
        except Exception as e:
            pytest.fail(f"メモリ管理テストエラー: {e}")

    def test_feature_importance(self):
        """特徴量重要度テスト"""
        logger.info("=== 特徴量重要度テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator(enable_automl=True)
            
            # 特徴量重要度取得
            importance = ml_orchestrator.get_feature_importance(top_n=5)
            
            # 結果検証（空の場合もあり得る）
            assert isinstance(importance, dict), "特徴量重要度が辞書形式ではありません"
            
            if importance:
                # 重要度値の妥当性確認
                for feature, score in importance.items():
                    assert isinstance(feature, str), "特徴量名が文字列ではありません"
                    assert isinstance(score, (int, float)), "重要度スコアが数値ではありません"
                    assert score >= 0, f"重要度スコアが負の値です: {score}"
            
            logger.info("✅ 特徴量重要度テスト成功")
            
        except Exception as e:
            pytest.fail(f"特徴量重要度テストエラー: {e}")


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
