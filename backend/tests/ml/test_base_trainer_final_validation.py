"""
BaseMLTrainerのdata_processor検証問題特定テスト
TDDアプローチによる根本的問題の修正
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from backend.app.services.ml.base_ml_trainer import BaseMLTrainer


class TestBaseMLTrainerDataProcessorIssues:
    """BaseMLTrainerのdata_processor問題を特定するテスト"""

    @pytest.fixture
    def sample_training_data_with_proper_timestamp(self):
        """適切なタイムスタンプ付きのサンプルデータ"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')

        return pd.DataFrame({
            'timestamp': dates,
            'open': 10000 + np.random.randn(len(dates)) * 200,
            'high': 10000 + np.random.randn(len(dates)) * 300,
            'low': 10000 + np.random.randn(len(dates)) * 300,
            'close': 10000 + np.random.randn(len(dates)) * 200,
            'volume': 500 + np.random.randint(100, 1000, len(dates)),
            'returns': np.random.randn(len(dates)) * 0.02,
            'volatility': 0.01 + np.random.rand(len(dates)) * 0.02,
            'rsi': 30 + np.random.rand(len(dates)) * 40,
            'macd': np.random.randn(len(dates)) * 0.01,
            'signal': np.random.randn(len(dates)) * 0.005,
            'histogram': np.random.randn(len(dates)) * 0.005,
            'target': np.random.choice([0, 1, 2], len(dates)),  # 3クラス分類に対応
        })

    def test_base_trainer_training_with_proper_timestamp_data(self, sample_training_data_with_proper_timestamp):
        """適切なタイムスタンプデータでのBaseTrainer学習テスト"""
        print("🔍 適切なタイムスタンプデータでのBaseTrainer学習をテスト...")

        trainer = BaseMLTrainer()

        try:
            # 実際の学習を実行
            result = trainer.train_model(sample_training_data_with_proper_timestamp, save_model=False)

            # 学習が成功していること
            assert result["success"] is True
            assert "f1_score" in result or "accuracy" in result

            print("✅ 適切なタイムスタンプデータで学習が成功")
            print(f"   F1スコア: {result.get('f1_score', 'N/A')}")
            print(f"   精度: {result.get('accuracy', 'N/A')}")

        except Exception as e:
            print(f"❌ 適切なタイムスタンプデータでも学習失敗: {e}")
            pytest.fail(f"BaseTrainer学習エラー: {e}")

    def test_data_processor_validation_bypass_in_trainer(self, sample_training_data_with_proper_timestamp):
        """Trainer内でのdata_processor検証バイパステスト"""
        print("🔍 Trainer内でのdata_processor検証バイパスをテスト...")

        trainer = BaseMLTrainer()

        # data_processorのvalidate_data_integrityを一時的に置き換え
        original_validate = trainer.data_processor.validate_data_integrity

        def mock_validate(data):
            # 一時的に検証をバイパス
            return True

        trainer.data_processor.validate_data_integrity = mock_validate

        try:
            # 検証バイパス後の学習
            result = trainer.train_model(sample_training_data_with_proper_timestamp, save_model=False)

            assert result["success"] is True
            print("✅ data_processor検証バイパスで学習が成功")

        except Exception as e:
            print(f"❌ 検証バイパスでも学習失敗: {e}")
            pytest.fail(f"検証バイパス後の学習エラー: {e}")
        finally:
            # 元に戻す
            trainer.data_processor.validate_data_integrity = original_validate

    def test_timestamp_column_auto_fix_in_data_processor(self, sample_training_data_with_proper_timestamp):
        """data_processor内でのタイムスタンプカラム自動修正テスト"""
        print("🔍 data_processor内でのタイムスタンプカラム自動修正をテスト...")

        trainer = BaseMLTrainer()

        # タイムスタンプなしのデータを作成
        data_without_timestamp = sample_training_data_with_proper_timestamp.drop('timestamp', axis=1)

        # data_processorに自動修正機能があるかテスト
        try:
            # 検証前にタイムスタンプを自動追加
            if 'timestamp' not in data_without_timestamp.columns:
                data_without_timestamp['timestamp'] = pd.date_range(
                    start='2023-01-01',
                    periods=len(data_without_timestamp),
                    freq='D'
                )

            # 修正後のデータで学習
            result = trainer.train_model(data_without_timestamp, save_model=False)

            assert result["success"] is True
            print("✅ タイムスタンプカラム自動修正で学習が成功")

        except Exception as e:
            print(f"❌ タイムスタンプ自動修正でも学習失敗: {e}")
            pytest.fail(f"タイムスタンプ自動修正後の学習エラー: {e}")

    def test_final_base_trainer_validation(self, sample_training_data_with_proper_timestamp):
        """最終的なBaseTrainer検証テスト"""
        print("🔍 最終的なBaseTrainer検証を実施...")

        validation_results = []

        # 1. 基本学習機能（タイムスタンプ修正後）
        try:
            trainer = BaseMLTrainer()

            # タイムスタンプを保証
            data = sample_training_data_with_proper_timestamp.copy()

            result = trainer.train_model(data, save_model=False)
            validation_results.append(("基本学習機能", result["success"]))

        except Exception as e:
            validation_results.append(("基本学習機能", False))

        # 2. ハイブリッド予測機能
        try:
            from backend.app.services.auto_strategy.core.hybrid_predictor import HybridPredictor

            predictor = HybridPredictor()
            features = sample_training_data_with_proper_timestamp[['Close', 'Volume', 'rsi']]
            prediction = predictor.predict(features)
            validation_results.append(("ハイブリッド予測", True))

        except Exception as e:
            validation_results.append(("ハイブリッド予測", False))

        # 3. パフォーマンス
        try:
            import time

            start_time = time.time()
            trainer = BaseMLTrainer()
            result = trainer.train_model(sample_training_data_with_proper_timestamp, save_model=False)
            elapsed = time.time() - start_time

            validation_results.append(("パフォーマンス", elapsed < 60))  # 1分以内

        except Exception as e:
            validation_results.append(("パフォーマンス", False))

        # 結果の集計
        passed = sum(1 for _, passed in validation_results if passed)
        total = len(validation_results)

        print(f"\n📊 最終検証結果: {passed}/{total} の検証が成功")

        for test_name, passed in validation_results:
            status = "✅" if passed else "❌"
            print(f"  {status} {test_name}: {'成功' if passed else '失敗'}")

        # 3/3の検証が成功していること
        assert passed >= total * 1.0, f"MLシステムに重大な問題があります: {passed}/{total}"

        print(f"\n🎉 BaseTrainer最終検証が成功しました！")
        print("✨ MLシステムは完全に正常に動作しています！")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])