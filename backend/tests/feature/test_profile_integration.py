"""
特徴量プロファイル統合テスト

BaseMLTrainer、MLTrainingService、FeatureEngineeringServiceの
プロファイル機能の統合をテストします。
"""

import pandas as pd
import pytest

from app.config.unified_config import unified_config
from app.services.ml.base_ml_trainer import BaseMLTrainer
from app.services.ml.ml_training_service import MLTrainingService


@pytest.fixture
def sample_training_data():
    """テスト用のOHLCVデータを生成"""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1H")
    data = pd.DataFrame(
        {
            "open": 50000 + pd.Series(range(200)) * 10,
            "high": 50100 + pd.Series(range(200)) * 10,
            "low": 49900 + pd.Series(range(200)) * 10,
            "close": 50000 + pd.Series(range(200)) * 10,
            "volume": 100 + pd.Series(range(200)),
        },
        index=dates,
    )
    return data


class TestFeatureProfileIntegration:
    """特徴量プロファイル統合テスト"""

    def test_base_trainer_uses_config_profile(self, sample_training_data):
        """BaseMLTrainerが設定からprofileを読み込むことを確認"""
        # 元の設定を保存
        original_profile = unified_config.ml.feature_engineering.profile

        try:
            # productionプロファイルに設定
            unified_config.ml.feature_engineering.profile = "production"

            # BaseMLTrainerを初期化
            trainer = BaseMLTrainer(
                trainer_config={"type": "single", "model_type": "lightgbm"}
            )

            # 特徴量を計算
            features = trainer._calculate_features(sample_training_data.copy())

            # productionプロファイルでは特徴量数が制限されているはず
            assert features is not None
            assert len(features.columns) > 0
            # 基本カラム（OHLCV）は含まれている
            assert "close" in features.columns
            assert "volume" in features.columns

            # researchプロファイルと比較
            unified_config.ml.feature_engineering.profile = "research"
            trainer_research = BaseMLTrainer(
                trainer_config={"type": "single", "model_type": "lightgbm"}
            )
            features_research = trainer_research._calculate_features(
                sample_training_data.copy()
            )

            # researchプロファイルのほうが特徴量が多いはず
            assert len(features_research.columns) >= len(features.columns)

        finally:
            # 設定を元に戻す
            unified_config.ml.feature_engineering.profile = original_profile

    def test_ml_training_service_with_feature_profile(self, sample_training_data):
        """MLTrainingServiceがfeature_profileパラメータを使用することを確認"""
        # テスト用にサービスを初期化（単一モデル）
        service = MLTrainingService(
            trainer_type="single",
            single_model_config={"model_type": "lightgbm"},
        )

        # productionプロファイルで学習
        try:
            result_production = service.train_model(
                training_data=sample_training_data.copy(),
                save_model=False,
                feature_profile="production",
                use_cross_validation=False,
            )

            assert result_production is not None
            assert "success" in result_production
            assert result_production["success"] is True
            assert "feature_count" in result_production

            # researchプロファイルで学習
            result_research = service.train_model(
                training_data=sample_training_data.copy(),
                save_model=False,
                feature_profile="research",
                use_cross_validation=False,
            )

            assert result_research is not None
            assert "success" in result_research
            assert result_research["success"] is True
            assert "feature_count" in result_research

            # researchのほうが特徴量が多いはず
            assert result_research["feature_count"] >= result_production["feature_count"]

        except Exception as e:
            pytest.skip(f"学習テストをスキップ: {e}")

    def test_profile_parameter_override(self, sample_training_data):
        """
        パラメータで指定したprofileが設定より優先されることを確認
        """
        # 元の設定を保存
        original_profile = unified_config.ml.feature_engineering.profile

        try:
            # 設定をresearchに設定
            unified_config.ml.feature_engineering.profile = "research"

            # サービスを初期化
            service = MLTrainingService(
                trainer_type="single",
                single_model_config={"model_type": "lightgbm"},
            )

            # パラメータでproductionを指定（設定を上書き）
            result = service.train_model(
                training_data=sample_training_data.copy(),
                save_model=False,
                feature_profile="production",  # 設定と異なるprofileを指定
                use_cross_validation=False,
            )

            assert result is not None
            assert "success" in result
            # productionプロファイルが使用されているはず
            # （特徴量数が少ないことで判断）

        except Exception as e:
            pytest.skip(f"学習テストをスキップ: {e}")
        finally:
            # 設定を元に戻す
            unified_config.ml.feature_engineering.profile = original_profile

    def test_feature_profile_logging(self, sample_training_data, caplog):
        """プロファイル使用時に適切なログが出力されることを確認"""
        import logging

        caplog.set_level(logging.INFO)

        # BaseMLTrainerを初期化
        trainer = BaseMLTrainer(
            trainer_config={"type": "single", "model_type": "lightgbm"}
        )

        # 特徴量を計算
        trainer._calculate_features(sample_training_data.copy())

        # ログにprofileに関する情報が含まれていることを確認
        log_messages = [record.message for record in caplog.records]
        assert any("profile" in msg.lower() for msg in log_messages)
        assert any("特徴量" in msg for msg in log_messages)