import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from app.services.ml.common.ml_utils import (
    optimize_dtypes,
    generate_cache_key,
    calculate_price_change,
    validate_training_inputs,
    get_feature_importance_unified,
    prepare_data_for_prediction,
    predict_class_from_proba
)

class TestMLUtils:
    def test_optimize_dtypes(self):
        """データ型最適化のテスト"""
        df = pd.DataFrame({
            "float_col": [1.0, 2.0, 3.0],
            "int_col": [1, 2, 3],
            "big_int": [2**40, 2**40 + 1, 2**40 + 2],
            "timestamp": [1672531200, 1672617600, 1672704000]
        })
        # float_col: float64 -> float32
        # int_col: int64 -> int32
        # big_int: int64 -> int64 (範囲外)
        # timestamp: 除外
        
        optimized = optimize_dtypes(df)
        assert optimized["float_col"].dtype == "float32"
        assert optimized["int_col"].dtype == "int32"
        assert optimized["big_int"].dtype == "int64"
        # timestampはint64のまま(明示的に除外されているため)
        assert optimized["timestamp"].dtype == "int64"

    def test_generate_cache_key(self):
        """キャッシュキー生成のテスト"""
        df1 = pd.DataFrame({"close": [100, 101]}, index=pd.date_range("2023-01-01", periods=2))
        df2 = pd.DataFrame({"close": [100, 102]}, index=pd.date_range("2023-01-01", periods=2))
        
        key1 = generate_cache_key(df1, extra_params={"p": 1})
        key1_same = generate_cache_key(df1, extra_params={"p": 1})
        key2 = generate_cache_key(df2, extra_params={"p": 1})
        key3 = generate_cache_key(df1, extra_params={"p": 2})
        
        assert key1 == key1_same
        assert key1 != key2
        assert key1 != key3
        assert key1.startswith("features_")

    def test_calculate_price_change(self):
        """価格変化率計算のテスト"""
        s = pd.Series([100, 110, 121, 108.9])
        # 10%, 10%, -10%
        
        # デフォルト (periods=1, shift=0, fill_na=True)
        res = calculate_price_change(s)
        assert res.iloc[0] == 0.0
        assert pytest.approx(float(res.iloc[1])) == 0.1
        
        # shift=1 (過去を参照)
        res_shifted = calculate_price_change(s, shift=1)
        assert pytest.approx(float(res_shifted.iloc[2])) == 0.1 # 1つ前の変化率

    def test_validate_training_inputs(self):
        """学習入力検証のテスト"""
        X = pd.DataFrame({"f": [1, 2]})
        y = pd.Series([0, 1])
        
        # 正常系
        validate_training_inputs(X, y)
        
        # 異常系: 長さ不一致
        with pytest.raises(ValueError, match="特徴量とターゲットの長さが一致しません"):
            validate_training_inputs(X, pd.Series([0]))
            
        # 異常系: 空
        with pytest.raises(ValueError, match="ターゲットデータが空です"):
            validate_training_inputs(X, pd.Series([], dtype=int))

    def test_get_feature_importance_unified(self):
        """特徴量重要度統一取得のテスト"""
        cols = ["f1", "f2", "f3"]

        # 1. feature_importances_ 属性 (sklearn style)
        # MagicMockは全ての属性に反応してしまうため、属性を明示的に削除するか、specを指定する
        model_sk = MagicMock(spec=["feature_importances_"])
        model_sk.feature_importances_ = np.array([0.1, 0.5, 0.4])
        
        res = get_feature_importance_unified(model_sk, cols, top_n=2)
        assert len(res) == 2
        assert list(res.keys()) == ["f2", "f3"]
        
        # 2. feature_importance メソッド (LGBM style)
        model_lgb = MagicMock(spec=["feature_importance"])
        model_lgb.feature_importance.return_value = np.array([10.0, 50.0, 40.0])
        res = get_feature_importance_unified(model_lgb, cols, top_n=1)
        assert list(res.keys()) == ["f2"]

        # 3. get_feature_importance メソッド
        model_custom = MagicMock(spec=["get_feature_importance"])
        model_custom.get_feature_importance.return_value = {"f3": 0.9, "f1": 0.1}
        res = get_feature_importance_unified(model_custom, cols, top_n=1)
        assert list(res.keys()) == ["f3"]

    def test_prepare_data_for_prediction(self):
        """予測データ準備のテスト"""
        df = pd.DataFrame({"f1": [1, 2], "f3": [3, 4]})
        expected = ["f1", "f2", "f3"]
        
        # f2が欠損しているが、0で補完され、順序が維持されるはず
        processed = prepare_data_for_prediction(df, expected)
        assert list(processed.columns) == ["f1", "f2", "f3"]
        assert (processed["f2"] == 0.0).all()
        
        # スケーラーあり
        scaler = MagicMock()
        scaler.transform.return_value = np.array([[10, 20, 30], [40, 50, 60]])
        processed_scaled = prepare_data_for_prediction(df, expected, scaler=scaler)
        assert processed_scaled.iloc[0, 0] == 10

    def test_predict_class_from_proba(self):
        """確率からクラスへの変換テスト"""
        # バイナリ
        probs = np.array([0.1, 0.6, 0.4, 0.9])
        classes = predict_class_from_proba(probs, threshold=0.5)
        np.testing.assert_array_equal(classes, [0, 1, 0, 1])
        
        # 多クラス
        probs_multi = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1]
        ])
        classes_multi = predict_class_from_proba(probs_multi)
        np.testing.assert_array_equal(classes_multi, [0, 1])
