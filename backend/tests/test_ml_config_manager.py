"""
ML設定管理機能のテスト

ML設定の更新、リセット、バリデーション機能をテストします。
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.ml.config.ml_config_manager import MLConfigManager
from app.utils.unified_error_handler import UnifiedValidationError


class TestMLConfigManager:
    """ML設定管理機能のテストクラス"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化処理"""
        # 一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_ml_config.json")
        
        # テスト用のMLConfigManagerを作成
        self.config_manager = MLConfigManager(config_file_path=self.config_file)
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ処理"""
        # 一時ファイルを削除
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_get_config_dict(self):
        """設定辞書の取得テスト"""
        config_dict = self.config_manager.get_config_dict()
        
        # 必要なセクションが存在することを確認
        assert "data_processing" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
        assert "prediction" in config_dict
        assert "ensemble" in config_dict
        assert "retraining" in config_dict
        
        # データ処理設定の確認
        dp_config = config_dict["data_processing"]
        assert "max_ohlcv_rows" in dp_config
        assert "feature_calculation_timeout" in dp_config
        assert isinstance(dp_config["max_ohlcv_rows"], int)
        assert dp_config["max_ohlcv_rows"] > 0
    
    def test_save_and_load_config(self):
        """設定の保存と読み込みテスト"""
        # 設定を保存
        save_result = self.config_manager.save_config()
        assert save_result is True
        assert os.path.exists(self.config_file)
        
        # 新しいマネージャーで設定を読み込み
        new_manager = MLConfigManager(config_file_path=self.config_file)
        loaded_config = new_manager.get_config_dict()
        original_config = self.config_manager.get_config_dict()
        
        # 設定が一致することを確認
        assert loaded_config["data_processing"]["max_ohlcv_rows"] == original_config["data_processing"]["max_ohlcv_rows"]
        assert loaded_config["prediction"]["default_up_prob"] == original_config["prediction"]["default_up_prob"]
    
    def test_update_config_valid(self):
        """有効な設定更新のテスト"""
        updates = {
            "data_processing": {
                "max_ohlcv_rows": 500000,
                "feature_calculation_timeout": 1800
            },
            "prediction": {
                "default_up_prob": 0.4,
                "default_down_prob": 0.3,
                "default_range_prob": 0.3
            }
        }
        
        result = self.config_manager.update_config(updates)
        assert result is True
        
        # 更新された設定を確認
        config_dict = self.config_manager.get_config_dict()
        assert config_dict["data_processing"]["max_ohlcv_rows"] == 500000
        assert config_dict["data_processing"]["feature_calculation_timeout"] == 1800
        assert config_dict["prediction"]["default_up_prob"] == 0.4
    
    def test_update_config_invalid(self):
        """無効な設定更新のテスト"""
        # 無効な値での更新
        invalid_updates = {
            "data_processing": {
                "max_ohlcv_rows": -1,  # 負の値は無効
                "feature_calculation_timeout": "invalid"  # 文字列は無効
            },
            "prediction": {
                "default_up_prob": 1.5  # 1.0を超える値は無効
            }
        }
        
        result = self.config_manager.update_config(invalid_updates)
        assert result is False
    
    def test_reset_config(self):
        """設定リセットのテスト"""
        # 設定を変更
        updates = {
            "data_processing": {
                "max_ohlcv_rows": 123456
            }
        }
        self.config_manager.update_config(updates)
        
        # 変更されたことを確認
        config_dict = self.config_manager.get_config_dict()
        assert config_dict["data_processing"]["max_ohlcv_rows"] == 123456
        
        # リセット実行
        result = self.config_manager.reset_config()
        assert result is True
        
        # デフォルト値に戻ったことを確認
        config_dict = self.config_manager.get_config_dict()
        assert config_dict["data_processing"]["max_ohlcv_rows"] == 1000000  # デフォルト値
    
    def test_validate_data_processing_config(self):
        """データ処理設定のバリデーションテスト"""
        # 有効な設定
        valid_config = {
            "max_ohlcv_rows": 100000,
            "feature_calculation_timeout": 3600,
            "log_level": "INFO"
        }
        errors = self.config_manager._validate_data_processing_config(valid_config)
        assert len(errors) == 0
        
        # 無効な設定
        invalid_config = {
            "max_ohlcv_rows": -1,
            "feature_calculation_timeout": 0,
            "log_level": "INVALID_LEVEL"
        }
        errors = self.config_manager._validate_data_processing_config(invalid_config)
        assert len(errors) > 0
        assert any("max_ohlcv_rows" in error for error in errors)
        assert any("feature_calculation_timeout" in error for error in errors)
        assert any("log_level" in error for error in errors)
    
    def test_validate_prediction_config(self):
        """予測設定のバリデーションテスト"""
        # 有効な設定
        valid_config = {
            "default_up_prob": 0.33,
            "default_down_prob": 0.33,
            "default_range_prob": 0.34,
            "probability_sum_min": 0.8,
            "probability_sum_max": 1.2
        }
        errors = self.config_manager._validate_prediction_config(valid_config)
        assert len(errors) == 0
        
        # 無効な設定
        invalid_config = {
            "default_up_prob": 1.5,  # 1.0を超える
            "default_down_prob": -0.1,  # 負の値
            "probability_sum_min": 1.0,
            "probability_sum_max": 0.5  # min > max
        }
        errors = self.config_manager._validate_prediction_config(invalid_config)
        assert len(errors) > 0
        assert any("default_up_prob" in error for error in errors)
        assert any("default_down_prob" in error for error in errors)
        assert any("probability_sum_min" in error for error in errors)
    
    def test_validate_training_config(self):
        """学習設定のバリデーションテスト"""
        # 有効な設定
        valid_config = {
            "train_test_split": 0.8,
            "cross_validation_folds": 5,
            "prediction_horizon": 24,
            "threshold_up": 0.02,
            "threshold_down": -0.02
        }
        errors = self.config_manager._validate_training_config(valid_config)
        assert len(errors) == 0
        
        # 無効な設定
        invalid_config = {
            "train_test_split": 1.5,  # 1.0を超える
            "cross_validation_folds": 1,  # 2未満
            "prediction_horizon": -1,  # 負の値
            "threshold_up": -0.01,  # 負の値
            "threshold_down": 0.01  # 正の値
        }
        errors = self.config_manager._validate_training_config(invalid_config)
        assert len(errors) > 0
        assert any("train_test_split" in error for error in errors)
        assert any("cross_validation_folds" in error for error in errors)
        assert any("prediction_horizon" in error for error in errors)
        assert any("threshold_up" in error for error in errors)
        assert any("threshold_down" in error for error in errors)
    
    def test_validate_ensemble_config(self):
        """アンサンブル設定のバリデーションテスト"""
        # 有効な設定
        valid_config = {
            "algorithms": ["lightgbm", "xgboost"],
            "voting_method": "soft",
            "stacking_cv_folds": 5
        }
        errors = self.config_manager._validate_ensemble_config(valid_config)
        assert len(errors) == 0
        
        # 無効な設定
        invalid_config = {
            "algorithms": ["invalid_algo"],
            "voting_method": "invalid_method",
            "stacking_cv_folds": 1
        }
        errors = self.config_manager._validate_ensemble_config(invalid_config)
        assert len(errors) > 0
        assert any("invalid_algo" in error for error in errors)
        assert any("voting_method" in error for error in errors)
        assert any("stacking_cv_folds" in error for error in errors)
    
    def test_backup_creation(self):
        """バックアップ作成のテスト"""
        # 初期設定を保存
        self.config_manager.save_config()
        
        # バックアップディレクトリが存在することを確認
        backup_dir = self.config_manager.backup_dir
        assert backup_dir.exists()
        
        # 設定を更新（バックアップが作成される）
        updates = {"data_processing": {"max_ohlcv_rows": 999999}}
        self.config_manager.update_config(updates)
        
        # バックアップファイルが作成されたことを確認
        backup_files = list(backup_dir.glob("ml_config_backup_*.json"))
        assert len(backup_files) > 0


if __name__ == "__main__":
    pytest.main([__file__])
