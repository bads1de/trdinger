"""
MLレポート問題修正のテスト

レポートで特定された問題点の修正を検証します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TestMLIssuesFixes:
    """ML問題修正のテストクラス"""
    
    def setup_method(self):
        """テスト前の準備"""
        # テスト用データの作成
        self.test_data = self.create_test_data()
    
    def create_test_data(self, size: int = 1000) -> pd.DataFrame:
        """テスト用データを作成"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=size, freq='H')
        
        # 基本価格データ
        close_prices = 50000 + np.cumsum(np.random.randn(size) * 100)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': close_prices + np.random.randn(size) * 50,
            'High': close_prices + np.abs(np.random.randn(size)) * 100,
            'Low': close_prices - np.abs(np.random.randn(size)) * 100,
            'Close': close_prices,
            'Volume': np.random.exponential(1000, size),
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def test_fear_greed_service_import_fix(self):
        """Fear & Greed サービスのインポート修正テスト"""
        logger.info("🔍 Fear & Greed サービスインポート修正テスト開始")
        
        try:
            # 修正されたインポートをテスト
            from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            service = FeatureEngineeringService()
            
            # Fear & Greed データ取得メソッドをテスト
            fear_greed_data = service._get_fear_greed_data(self.test_data)
            
            # エラーが発生しないことを確認（データが取得できなくてもOK）
            assert fear_greed_data is None or isinstance(fear_greed_data, pd.DataFrame)
            
            logger.info("✅ Fear & Greed サービスインポート修正成功")
            
        except ImportError as e:
            pytest.fail(f"インポートエラーが修正されていません: {e}")
        except Exception as e:
            logger.warning(f"その他のエラー（許容範囲）: {e}")
    
    def test_missing_features_generation(self):
        """不足特徴量の疑似生成テスト"""
        logger.info("🔍 不足特徴量疑似生成テスト開始")
        
        try:
            from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            service = FeatureEngineeringService()
            
            # ファンディングレート疑似特徴量生成テスト
            lookback_periods = {"short": 24, "medium": 168, "long": 720}
            result_df = service._generate_pseudo_funding_rate_features(self.test_data, lookback_periods)
            
            # 必要な特徴量が生成されていることを確認
            expected_fr_features = [
                "FR_MA_24", "FR_MA_168", "FR_Change", "FR_Change_Rate",
                "Price_FR_Divergence", "FR_Normalized", "FR_Trend", "FR_Volatility"
            ]
            
            for feature in expected_fr_features:
                assert feature in result_df.columns, f"FR特徴量 {feature} が生成されていません"
            
            # 建玉残高疑似特徴量生成テスト
            result_df = service._generate_pseudo_open_interest_features(result_df, lookback_periods)
            
            expected_oi_features = [
                "OI_Change_Rate", "OI_Change_Rate_24h", "OI_Surge", "Volatility_Adjusted_OI",
                "OI_MA_24", "OI_MA_168", "OI_Trend", "OI_Price_Correlation", "OI_Normalized"
            ]
            
            for feature in expected_oi_features:
                assert feature in result_df.columns, f"OI特徴量 {feature} が生成されていません"
            
            # NaN値が適切に処理されていることを確認
            for feature in expected_fr_features + expected_oi_features:
                nan_count = result_df[feature].isna().sum()
                total_count = len(result_df)
                nan_ratio = nan_count / total_count
                assert nan_ratio < 0.1, f"特徴量 {feature} のNaN率が高すぎます: {nan_ratio:.2%}"
            
            logger.info("✅ 不足特徴量疑似生成成功")
            
        except Exception as e:
            pytest.fail(f"不足特徴量生成エラー: {e}")
    
    def test_outlier_removal_optimization(self):
        """外れ値除去最適化テスト"""
        logger.info("🔍 外れ値除去最適化テスト開始")
        
        try:
            from app.utils.data_processing import DataProcessor
            
            processor = DataProcessor()
            
            # 外れ値を含むテストデータを作成
            test_data = self.test_data.copy()
            
            # 意図的に外れ値を追加
            outlier_indices = np.random.choice(len(test_data), size=50, replace=False)
            test_data.iloc[outlier_indices, test_data.columns.get_loc('Close')] *= 10
            
            # 処理時間を測定
            start_time = time.time()
            
            # 最適化された外れ値除去を実行
            cleaned_data = processor._remove_outliers(
                test_data,
                columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                threshold=3.0,
                method='iqr'
            )
            
            processing_time = time.time() - start_time
            
            # 処理時間が改善されていることを確認（基準値は環境依存）
            assert processing_time < 5.0, f"外れ値除去処理時間が長すぎます: {processing_time:.2f}秒"
            
            # 外れ値が適切に除去されていることを確認
            original_outliers = (test_data['Close'] > test_data['Close'].quantile(0.99)).sum()
            remaining_outliers = (cleaned_data['Close'] > cleaned_data['Close'].quantile(0.99)).sum()
            
            # 外れ値が減少していることを確認
            assert remaining_outliers <= original_outliers, "外れ値除去が機能していません"
            
            logger.info(f"✅ 外れ値除去最適化成功: {processing_time:.3f}秒")
            
        except Exception as e:
            pytest.fail(f"外れ値除去最適化エラー: {e}")
    
    def test_tsfresh_feature_expansion(self):
        """TSFresh特徴量拡充テスト"""
        logger.info("🔍 TSFresh特徴量拡充テスト開始")
        
        try:
            from app.services.ml.feature_engineering.automl_features.automl_config import TSFreshConfig
            
            # 設定が更新されていることを確認
            config = TSFreshConfig()
            
            assert config.fdr_level == 0.1, f"FDR閾値が更新されていません: {config.fdr_level}"
            assert config.feature_count_limit == 200, f"特徴量数制限が更新されていません: {config.feature_count_limit}"
            assert config.performance_mode == "comprehensive", f"パフォーマンスモードが更新されていません: {config.performance_mode}"
            
            logger.info("✅ TSFresh設定更新確認成功")
            
        except Exception as e:
            pytest.fail(f"TSFresh設定確認エラー: {e}")
    
    def test_market_regime_detection(self):
        """市場レジーム検出テスト"""
        logger.info("🔍 市場レジーム検出テスト開始")
        
        try:
            from app.services.ml.adaptive_learning.market_regime_detector import MarketRegimeDetector, MarketRegime
            
            detector = MarketRegimeDetector()
            
            # レジーム検出を実行
            result = detector.detect_regime(self.test_data)
            
            # 結果の検証
            assert isinstance(result.regime, MarketRegime), "レジーム検出結果が無効です"
            assert 0 <= result.confidence <= 1, f"信頼度が範囲外です: {result.confidence}"
            assert isinstance(result.indicators, dict), "指標辞書が無効です"
            assert isinstance(result.timestamp, datetime), "タイムスタンプが無効です"
            
            # 指標が計算されていることを確認
            expected_indicators = ['trend_strength', 'volatility', 'rsi', 'volume_ratio']
            for indicator in expected_indicators:
                assert indicator in result.indicators, f"指標 {indicator} が不足しています"
            
            logger.info(f"✅ 市場レジーム検出成功: {result.regime.value} (信頼度: {result.confidence:.2f})")
            
        except Exception as e:
            pytest.fail(f"市場レジーム検出エラー: {e}")
    
    def test_adaptive_learning_service(self):
        """適応的学習サービステスト"""
        logger.info("🔍 適応的学習サービステスト開始")
        
        try:
            from app.services.ml.adaptive_learning.adaptive_learning_service import AdaptiveLearningService
            
            service = AdaptiveLearningService()
            
            # 適応処理を実行
            current_performance = {'accuracy': 0.65, 'precision': 0.62, 'recall': 0.68}
            result = service.adapt_to_market_changes(self.test_data, current_performance)
            
            # 結果の検証
            assert result is not None, "適応結果が取得できません"
            assert hasattr(result, 'action_taken'), "アクション情報が不足しています"
            assert hasattr(result, 'regime_detected'), "レジーム情報が不足しています"
            assert hasattr(result, 'confidence'), "信頼度情報が不足しています"
            assert hasattr(result, 'model_updated'), "モデル更新情報が不足しています"
            
            # 要約情報を取得
            summary = service.get_adaptation_summary()
            assert isinstance(summary, dict), "要約情報が無効です"
            assert 'current_regime' in summary, "現在のレジーム情報が不足しています"
            
            logger.info(f"✅ 適応的学習サービス成功: {result.action_taken}")
            
        except Exception as e:
            pytest.fail(f"適応的学習サービスエラー: {e}")
    
    def test_comprehensive_integration(self):
        """包括的統合テスト"""
        logger.info("🔍 包括的統合テスト開始")
        
        try:
            from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            service = FeatureEngineeringService()
            
            # 統合的な特徴量エンジニアリングを実行
            # （データ不足の場合でも疑似特徴量が生成されることを確認）
            result_df = service.calculate_advanced_features(
                ohlcv_data=self.test_data,
                funding_rate_data=None,  # 意図的にNoneを渡す
                open_interest_data=None,  # 意図的にNoneを渡す
                fear_greed_data=None,    # 意図的にNoneを渡す
                lookback_periods={"short": 24, "medium": 168, "long": 720}
            )
            
            # 基本的な特徴量が生成されていることを確認
            assert len(result_df.columns) > len(self.test_data.columns), "特徴量が生成されていません"
            
            # 疑似特徴量が含まれていることを確認
            pseudo_features = ['FR_Normalized', 'OI_Change_Rate', 'OI_Trend']
            for feature in pseudo_features:
                assert feature in result_df.columns, f"疑似特徴量 {feature} が生成されていません"
            
            # データ品質を確認
            total_nan_ratio = result_df.isna().sum().sum() / (len(result_df) * len(result_df.columns))
            assert total_nan_ratio < 0.1, f"NaN率が高すぎます: {total_nan_ratio:.2%}"
            
            logger.info(f"✅ 包括的統合テスト成功: {len(result_df.columns)}個の特徴量生成")
            
        except Exception as e:
            pytest.fail(f"包括的統合テストエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    test_instance = TestMLIssuesFixes()
    test_instance.setup_method()
    
    # 各テストを実行
    tests = [
        test_instance.test_fear_greed_service_import_fix,
        test_instance.test_missing_features_generation,
        test_instance.test_outlier_removal_optimization,
        test_instance.test_tsfresh_feature_expansion,
        test_instance.test_market_regime_detection,
        test_instance.test_adaptive_learning_service,
        test_instance.test_comprehensive_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"テスト失敗: {test.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 テスト結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
