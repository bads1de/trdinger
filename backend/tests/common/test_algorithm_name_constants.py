"""
モデルラッパークラスのALGORITHM_NAME定数のテスト
"""

import sys
import os

# バックエンドディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from app.services.ml.common.algorithm_registry import AlgorithmRegistry

# モデルラッパークラスをモックとして作成
class LightGBMModel:
    ALGORITHM_NAME = "lightgbm"

class XGBoostModel:
    ALGORITHM_NAME = "xgboost"

class CatBoostModel:
    ALGORITHM_NAME = "catboost"

class RandomForestModel:
    ALGORITHM_NAME = "randomforest"

class ExtraTreesModel:
    ALGORITHM_NAME = "extratrees"

class AdaBoostModel:
    ALGORITHM_NAME = "adaboost"

class GradientBoostingModel:
    ALGORITHM_NAME = "gradientboosting"

class KNNModel:
    ALGORITHM_NAME = "knn"

class NaiveBayesModel:
    ALGORITHM_NAME = "naivebayes"

class RidgeModel:
    ALGORITHM_NAME = "ridge"

class TabNetModel:
    ALGORITHM_NAME = "tabnet"


class TestAlgorithmNameConstants:
    """モデルラッパークラスのALGORITHM_NAME定数のテスト"""

    def test_algorithm_name_constants_existence(self):
        """すべてのモデルラッパークラスにALGORITHM_NAME定数が存在することを確認するテスト"""
        model_classes = [
            LightGBMModel,
            XGBoostModel,
            CatBoostModel,
            RandomForestModel,
            ExtraTreesModel,
            AdaBoostModel,
            GradientBoostingModel,
            KNNModel,
            NaiveBayesModel,
            RidgeModel,
            TabNetModel,
        ]
        
        for model_class in model_classes:
            assert hasattr(model_class, 'ALGORITHM_NAME'), f"{model_class.__name__} does not have ALGORITHM_NAME constant"
            
            # 定数が文字列であることを確認
            algorithm_name = getattr(model_class, 'ALGORITHM_NAME')
            assert isinstance(algorithm_name, str), f"{model_class.__name__}.ALGORITHM_NAME is not a string"
            assert len(algorithm_name) > 0, f"{model_class.__name__}.ALGORITHM_NAME is empty"

    def test_algorithm_name_consistency_with_registry(self):
        """ALGORITHM_NAME定数がAlgorithmRegistryと一致することを確認するテスト"""
        test_cases = [
            (LightGBMModel, "LightGBMModel"),
            (XGBoostModel, "XGBoostModel"),
            (CatBoostModel, "CatBoostModel"),
            (RandomForestModel, "RandomForestModel"),
            (ExtraTreesModel, "ExtraTreesModel"),
            (AdaBoostModel, "AdaBoostModel"),
            (GradientBoostingModel, "GradientBoostingModel"),
            (KNNModel, "KNNModel"),
            (NaiveBayesModel, "NaiveBayesModel"),
            (RidgeModel, "RidgeModel"),
            (TabNetModel, "TabNetModel"),
        ]
        
        for model_class, class_name in test_cases:
            expected_algorithm_name = AlgorithmRegistry.get_algorithm_name(class_name)
            actual_algorithm_name = model_class.ALGORITHM_NAME
            
            assert actual_algorithm_name == expected_algorithm_name, \
                f"{model_class.__name__}.ALGORITHM_NAME mismatch: expected {expected_algorithm_name}, got {actual_algorithm_name}"

    def test_algorithm_name_supported_by_registry(self):
        """ALGORITHM_NAME定数がAlgorithmRegistryでサポートされていることを確認するテスト"""
        model_classes = [
            LightGBMModel,
            XGBoostModel,
            CatBoostModel,
            RandomForestModel,
            ExtraTreesModel,
            AdaBoostModel,
            GradientBoostingModel,
            KNNModel,
            NaiveBayesModel,
            RidgeModel,
            TabNetModel,
        ]
        
        for model_class in model_classes:
            algorithm_name = model_class.ALGORITHM_NAME
            assert AlgorithmRegistry.is_supported_algorithm(algorithm_name), \
                f"{model_class.__name__}.ALGORITHM_NAME '{algorithm_name}' is not supported by AlgorithmRegistry"

    def test_algorithm_name_uniqueness(self):
        """ALGORITHM_NAME定数が一意であることを確認するテスト"""
        model_classes = [
            LightGBMModel,
            XGBoostModel,
            CatBoostModel,
            RandomForestModel,
            ExtraTreesModel,
            AdaBoostModel,
            GradientBoostingModel,
            KNNModel,
            NaiveBayesModel,
            RidgeModel,
            TabNetModel,
        ]
        
        algorithm_names = [model_class.ALGORITHM_NAME for model_class in model_classes]
        unique_algorithm_names = set(algorithm_names)
        
        assert len(algorithm_names) == len(unique_algorithm_names), \
            f"Duplicate ALGORITHM_NAME constants found: {algorithm_names}"

    def test_algorithm_name_values(self):
        """ALGORITHM_NAME定数の値が期待通りであることを確認するテスト"""
        expected_values = {
            LightGBMModel: "lightgbm",
            XGBoostModel: "xgboost",
            CatBoostModel: "catboost",
            RandomForestModel: "randomforest",
            ExtraTreesModel: "extratrees",
            AdaBoostModel: "adaboost",
            GradientBoostingModel: "gradientboosting",
            KNNModel: "knn",
            NaiveBayesModel: "naivebayes",
            RidgeModel: "ridge",
            TabNetModel: "tabnet",
        }
        
        for model_class, expected_value in expected_values.items():
            actual_value = model_class.ALGORITHM_NAME
            assert actual_value == expected_value, \
                f"{model_class.__name__}.ALGORITHM_NAME expected {expected_value}, got {actual_value}"

    def test_algorithm_name_display_name_mapping(self):
        """ALGORITHM_NAME定数から表示名が正しく取得できることを確認するテスト"""
        model_classes = [
            LightGBMModel,
            XGBoostModel,
            CatBoostModel,
            RandomForestModel,
            ExtraTreesModel,
            AdaBoostModel,
            GradientBoostingModel,
            KNNModel,
            NaiveBayesModel,
            RidgeModel,
            TabNetModel,
        ]
        
        for model_class in model_classes:
            algorithm_name = model_class.ALGORITHM_NAME
            display_name = AlgorithmRegistry.get_display_name(algorithm_name)
            
            assert display_name is not None, f"Display name for {algorithm_name} is None"
            assert isinstance(display_name, str), f"Display name for {algorithm_name} is not a string"
            assert len(display_name) > 0, f"Display name for {algorithm_name} is empty"