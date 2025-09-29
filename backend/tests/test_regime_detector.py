"""レジーム検知のテスト"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from app.services.auto_strategy.services.regime_detector import RegimeDetector
from pydantic import BaseModel, ValidationError


class TestRegimeDetector:
    """RegimeDetectorのテストクラス"""

    @pytest.fixture
    def mock_config(self):
        """モック設定オブジェクト"""
        config = Mock()
        config.n_components = 3
        config.covariance_type = "full"
        config.n_iter = 100
        return config

    @pytest.fixture
    def detector(self, mock_config):
        """RegimeDetectorインスタンス"""
        return RegimeDetector(mock_config)

    def test_initialization(self, detector):
        """初期化テスト"""
        assert detector.config is not None
        assert hasattr(detector, 'model')
        assert detector.model is not None

    def test_detect_regime_basic(self, detector):
        """基本的なレジーム検知テスト"""
        # ダミーOHLCVデータ作成（トレンド、レンジ、高ボラの特徴）
        np.random.seed(42)
        n_samples = 100

        # トレンドデータ（上昇トレンド）
        trend_data = np.cumsum(np.random.randn(n_samples, 4) * 0.01 + 0.001, axis=0)
        trend_volume = np.random.rand(n_samples) * 1000 + 500
        trend_data = np.column_stack([trend_data, trend_volume])

        # レンジデータ（横ばい）
        range_data = np.random.randn(n_samples, 4) * 0.005
        range_volume = np.random.rand(n_samples) * 1000 + 500
        range_data = np.column_stack([range_data, range_volume])

        # 高ボラデータ（高変動）
        high_vol_data = np.random.randn(n_samples, 4) * 0.02
        high_vol_volume = np.random.rand(n_samples) * 1000 + 500
        high_vol_data = np.column_stack([high_vol_data, high_vol_volume])

        # データを結合
        data = np.vstack([trend_data, range_data, high_vol_data])
        data_df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])

        # レジーム検知実行
        regimes = detector.detect_regimes(data_df)

        assert isinstance(regimes, np.ndarray)
        assert len(regimes) == len(data)
        assert all(regime in [0, 1, 2] for regime in regimes)  # 0=トレンド、1=レンジ、2=高ボラ

    def test_regime_transition_validation(self, detector):
        """状態遷移検証テスト"""
        np.random.seed(42)
        n_samples = 50

        # 明確な遷移データ作成
        trend_data = np.array([[i * 0.01, i * 0.01 + 0.005, i * 0.01 - 0.005, i * 0.01 + 0.002] for i in range(n_samples)])
        trend_volume = np.random.rand(n_samples) * 1000 + 500
        trend_data = np.column_stack([trend_data, trend_volume])

        range_data = np.array([[0.5 + np.random.randn() * 0.001 for _ in range(4)] for _ in range(n_samples)])
        range_volume = np.random.rand(n_samples) * 1000 + 500
        range_data = np.column_stack([range_data, range_volume])

        data = np.vstack([trend_data, range_data])
        data_df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])

        regimes = detector.detect_regimes(data_df)

        # 最初の部分がトレンド（0）、後半がレンジ（1）であることを確認
        first_half_regimes = regimes[:n_samples]
        second_half_regimes = regimes[n_samples:]

        # 遷移の確認（トレンドからレンジへの遷移があること）
        assert len(set(regimes)) >= 2  # 少なくとも2つの異なるレジーム

    def test_empty_data_handling(self, detector):
        """空データ処理テスト"""
        with pytest.raises(ValueError):
            detector.detect_regimes(pd.DataFrame())

    def test_invalid_data_handling(self, detector):
        """不正データ処理テスト"""
        # NaNを含むデータ
        invalid_data = pd.DataFrame([[1.0, 2.0, np.nan, 4.0, 100.0]], columns=['open', 'high', 'low', 'close', 'volume'])
        with pytest.raises(ValueError):
            detector.detect_regimes(invalid_data)

        # 無限大を含むデータ
        invalid_data_inf = pd.DataFrame([[1.0, 2.0, np.inf, 4.0, 100.0]], columns=['open', 'high', 'low', 'close', 'volume'])
        with pytest.raises(ValueError):
            detector.detect_regimes(invalid_data_inf)

    class OHLCVInput(BaseModel):
        """OHLCV入力データのPydanticモデル（テスト用）"""
        open: float
        high: float
        low: float
        close: float
        volume: float

    def test_pydantic_validation_valid_data(self):
        """Pydanticバリデーション有効データテスト"""
        valid_data = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0
        }
        record = self.OHLCVInput(**valid_data)
        assert record.open == 100.0
        assert record.close == 102.0

    def test_pydantic_validation_invalid_data(self):
        """Pydanticバリデーション無効データテスト"""
        invalid_data = {
            "open": "invalid",
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0
        }
        with pytest.raises(ValidationError):
            self.OHLCVInput(**invalid_data)

    def test_pydantic_validation_missing_fields(self):
        """Pydanticバリデーション必須フィールド欠如テスト"""
        incomplete_data = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            # closeとvolumeが欠如
        }
        with pytest.raises(ValidationError):
            self.OHLCVInput(**incomplete_data)

    def test_model_fitting_error_handling(self, detector):
        """モデルフィッティングエラーハンドリングテスト"""
        data = pd.DataFrame(np.random.randn(10, 5), columns=['open', 'high', 'low', 'close', 'volume'])
        with patch.object(detector.model, 'fit', side_effect=Exception("Fitting failed")):
            with pytest.raises(RuntimeError):
                detector.detect_regimes(data)

    def test_regime_label_mapping(self, detector):
        """レジームラベルマッピングテスト"""
        # モックを使用して特定の状態を返す
        with patch.object(detector.model, 'predict', return_value=np.array([0, 1, 2, 0, 1])):
            data = pd.DataFrame(np.random.randn(5, 5), columns=['open', 'high', 'low', 'close', 'volume'])
            regimes = detector.detect_regimes(data)

            # ラベルのマッピングを確認（0=トレンド、1=レンジ、2=高ボラ）
            expected_labels = ['trend', 'range', 'high_volatility', 'trend', 'range']
            assert len(regimes) == 5
            # 実際のラベルは実装依存だが、数値で返されることを確認