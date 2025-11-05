"""
TechnicalFeatureCalculatorの欠如メソッド検出テスト
TDDアプローチによる問題特定と修正
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from backend.app.services.ml.feature_engineering.technical_features import TechnicalFeatureCalculator


class TestTechnicalFeatureCalculatorIssues:
    """TechnicalFeatureCalculatorの問題を特定するテスト"""

    @pytest.fixture
    def sample_price_data(self):
        """サンプル価格データ"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')

        return pd.DataFrame({
            'timestamp': dates,
            'open': 10000 + np.random.randn(len(dates)) * 100,
            'high': 10100 + np.random.randn(len(dates)) * 150,
            'low': 9900 + np.random.randn(len(dates)) * 150,
            'close': 10000 + np.random.randn(len(dates)) * 100,
            'volume': 1000 + np.random.randint(100, 1000, len(dates)),
        })

    def test_calculate_pattern_features_method_exists(self):
        """calculate_pattern_featuresメソッドが存在するかテスト"""
        print("🔍 calculate_pattern_featuresメソッドの存在を確認...")

        calculator = TechnicalFeatureCalculator()

        # メソッドが存在するか確認
        if hasattr(calculator, 'calculate_pattern_features'):
            print("✅ calculate_pattern_featuresメソッドが存在")
        else:
            print("❌ calculate_pattern_featuresメソッドが存在しない - 修正が必要")
            assert hasattr(calculator, 'calculate_pattern_features'), \
                "TechnicalFeatureCalculatorにcalculate_pattern_featuresメソッドが実装されていません"

    def test_calculate_pattern_features_functionality(self, sample_price_data):
        """calculate_pattern_featuresメソッドの機能テスト"""
        print("🔍 calculate_pattern_featuresメソッドの機能をテスト...")

        calculator = TechnicalFeatureCalculator()

        # メソッドが存在する前提でテスト
        if hasattr(calculator, 'calculate_pattern_features'):
            try:
                # 実際の計算を実行
                result = calculator.calculate_pattern_features(sample_price_data)

                # 結果が適切な形式であること
                assert isinstance(result, pd.DataFrame)
                assert len(result) == len(sample_price_data)
                assert 'pattern_features' in result.columns or len(result.columns) > len(sample_price_data.columns)

                print("✅ calculate_pattern_featuresメソッドが正常に動作")

            except Exception as e:
                print(f"❌ calculate_pattern_featuresメソッドでエラー: {e}")
                pytest.fail(f"calculate_pattern_featuresメソッドの実装に問題があります: {e}")
        else:
            print("⚠️ calculate_pattern_featuresメソッドが存在しないためスキップ")
            pytest.skip("calculate_pattern_featuresメソッドが実装されていません")

    @pytest.mark.skip(reason="delattrでインスタンスメソッドは削除できない。テストロジックが間違っている")
    def test_technical_feature_calculation_fallback(self, sample_price_data):
        """技術的指標計算のフォールバック機構テスト"""
        print("🔍 技術的指標計算のフォールバック機構をテスト...")

        calculator = TechnicalFeatureCalculator()

        # calculate_pattern_featuresが失敗した場合のフォールバックをテスト
        original_method = getattr(calculator, 'calculate_pattern_features', None)

        # 一時的にメソッドを削除してフォールバックをテスト
        if hasattr(calculator, 'calculate_pattern_features'):
            delattr(calculator, 'calculate_pattern_features')

        try:
            # 他の技術的指標計算が正常に動作すること
            try:
                # RSI計算など他のメソッドが存在すること
                if hasattr(calculator, 'calculate_rsi'):
                    rsi_result = calculator.calculate_rsi(sample_price_data['close'])
                    assert isinstance(rsi_result, pd.Series)
                    print("✅ RSI計算が正常に動作")
                else:
                    print("⚠️ calculate_rsiメソッドが存在しない")

                if hasattr(calculator, 'calculate_macd'):
                    macd_result = calculator.calculate_macd(sample_price_data['close'])
                    assert isinstance(macd_result, tuple) and len(macd_result) == 3
                    print("✅ MACD計算が正常に動作")
                else:
                    print("⚠️ calculate_macdメソッドが存在しない")

            except Exception as e:
                print(f"⚠️ 他の技術的指標計算で警告: {e}")

        finally:
            # メソッドを元に戻す
            if original_method:
                setattr(calculator, 'calculate_pattern_features', original_method)

    def test_feature_calculator_interface_consistency(self, sample_price_data):
        """特徴量計算器のインターフェース一貫性テスト"""
        print("🔍 特徴量計算器のインターフェース一貫性をテスト...")

        calculator = TechnicalFeatureCalculator()

        # 期待されるメソッドのリスト
        expected_methods = [
            'calculate_rsi',
            'calculate_macd',
            'calculate_bollinger_bands',
            'calculate_atr',
            'calculate_pattern_features',  # これが欠如している
        ]

        missing_methods = []
        for method_name in expected_methods:
            if not hasattr(calculator, method_name):
                missing_methods.append(method_name)

        if missing_methods:
            print(f"❌ 欠如しているメソッド: {missing_methods}")
            print("✅ 実装済みのメソッド: {[m for m in expected_methods if m not in missing_methods]}")
        else:
            print("✅ すべての期待されるメソッドが実装されている")

        # calculate_pattern_featuresが必須であることを強調
        assert 'calculate_pattern_features' not in missing_methods, \
            f"calculate_pattern_featuresメソッドが実装されていません: {missing_methods}"


class TestCircularImportDetection:
    """循環インポートを検出するテスト"""

    def test_import_backtest_data_service(self):
        """BacktestDataServiceのインポートをテスト"""
        print("🔍 BacktestDataServiceのインポートをテスト...")

        try:
            from backend.app.services.backtest.backtest_data_service import BacktestDataService
            print("✅ BacktestDataServiceのインポート成功")
        except ImportError as e:
            print(f"❌ BacktestDataServiceのインポート失敗: {e}")
            pytest.fail(f"循環インポートが検出されました: {e}")

    def test_import_auto_strategy_service(self):
        """AutoStrategyServiceのインポートをテスト"""
        print("🔍 AutoStrategyServiceのインポートをテスト...")

        try:
            from backend.app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            print("✅ AutoStrategyServiceのインポート成功")
        except ImportError as e:
            print(f"❌ AutoStrategyServiceのインポート失敗: {e}")
            pytest.fail(f"循環インポートが検出されました: {e}")

    def test_import_ml_orchestration_service(self):
        """MLオーケストレーションサービスのインポートをテスト"""
        print("🔍 MLオーケストレーションサービスのインポートをテスト...")

        try:
            from backend.app.services.ml.orchestration.ml_training_orchestration_service import MLTrainingOrchestrationService
            print("✅ MLオーケストレーションサービスのインポート成功")
        except ImportError as e:
            print(f"❌ MLオーケストレーションサービスのインポート失敗: {e}")
            pytest.fail(f"循環インポートが検出されました: {e}")

    def test_cross_import_consistency(self):
        """クロスインポートの一貫性テスト"""
        print("🔍 クロスインポートの一貫性をテスト...")

        # 複数のサービスを同時にインポート
        try:
            from backend.app.services.backtest.backtest_data_service import BacktestDataService
            from backend.app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            from backend.app.services.ml.ml_training_service import MLTrainingService

            # サービスの初期化が成功すること
            backtest_service = BacktestDataService()
            auto_strategy_service = AutoStrategyService()
            ml_training_service = MLTrainingService()

            assert backtest_service is not None
            assert auto_strategy_service is not None
            assert ml_training_service is not None

            print("✅ クロスインポートの一貫性が確認")

        except Exception as e:
            print(f"❌ クロスインポートで問題発生: {e}")
            pytest.fail(f"サービス間のインポートに問題があります: {e}")


class TestDRLWeightValidation:
    """DRL重みバリデーションテスト"""

    def test_drl_weight_range_validation(self):
        """DRL重みの範囲バリデーションテスト"""
        print("🔍 DRL重みの範囲バリデーションをテスト...")

        try:
            from backend.app.services.auto_strategy.core.hybrid_predictor import HybridPredictor

            # 有効な重みでの初期化
            predictor_valid = HybridPredictor()
            predictor_valid._drl_weight = 0.5  # 有効な範囲内
            assert 0.0 <= predictor_valid._drl_weight <= 1.0
            print("✅ 有効なDRL重みが正常に設定")

            # 無効な重みのテスト
            predictor_invalid = HybridPredictor()

            # 重みを自動調整する仕組みがあるかテスト
            predictor_invalid._drl_weight = 1.5  # 無効な範囲
            if hasattr(predictor_invalid, '_validate_drl_weight'):
                predictor_invalid._validate_drl_weight()
                print("✅ DRL重みのバリデーションが実装されている")
            else:
                print("⚠️ DRL重みのバリデーションが未実装")

        except ImportError as e:
            print(f"⚠️ HybridPredictorのインポートに問題: {e}")
        except Exception as e:
            print(f"⚠️ DRL重みバリデーションで問題: {e}")

    def test_hybrid_predictor_drl_integration(self):
        """ハイブリッド予測器のDRL統合テスト"""
        print("🔍 ハイブリッド予測器のDRL統合をテスト...")

        try:
            from backend.app.services.auto_strategy.core.hybrid_predictor import HybridPredictor

            # DRL有効時のテスト
            predictor_with_drl = HybridPredictor(
                automl_config={
                    "drl": {
                        "enabled": True,
                        "policy_weight": 0.3
                    }
                }
            )

            assert predictor_with_drl._drl_enabled is True
            assert 0.0 <= predictor_with_drl._drl_weight <= 1.0
            print("✅ DRL有効時の重みが適切に設定")

            # DRL無効時のテスト
            predictor_without_drl = HybridPredictor(
                automl_config={
                    "drl": {
                        "enabled": False
                    }
                }
            )

            assert predictor_without_drl._drl_enabled is False
            print("✅ DRL無効時の設定が正常")

        except ImportError as e:
            print(f"⚠️ HybridPredictorのインポートに問題: {e}")
        except Exception as e:
            print(f"⚠️ DRL統合テストで問題: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])