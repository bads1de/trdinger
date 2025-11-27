"""
HybridFeatureAdapterのテストモジュール

StrategyGene → 特徴量DataFrame変換のテスト
TDD: テストファースト
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.models.condition import Condition
from app.services.auto_strategy.models.indicator_gene import IndicatorGene
from app.services.auto_strategy.models.strategy_gene import StrategyGene
from app.services.ml.exceptions import MLFeatureError


class TestHybridFeatureAdapter:
    """HybridFeatureAdapterのテストクラス"""

    @pytest.fixture
    def sample_strategy_gene(self):
        """サンプルStrategyGene"""
        indicator1 = IndicatorGene(
            type="SMA",
            parameters={"period": 20},
        )
        indicator2 = IndicatorGene(
            type="RSI",
            parameters={"period": 14},
        )

        condition1 = Condition(
            left_operand="close",
            operator=">",
            right_operand="SMA",
        )

        gene = StrategyGene(
            id="test_gene_001",
            indicators=[indicator1, indicator2],
            entry_conditions=[condition1],
            exit_conditions=[],
            long_entry_conditions=[condition1],
            short_entry_conditions=[],
        )

        return gene

    @pytest.fixture
    def sample_ohlcv_data(self):
        """サンプルOHLCVデータ"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.uniform(40000, 41000, 100),
                "high": np.random.uniform(41000, 42000, 100),
                "low": np.random.uniform(39000, 40000, 100),
                "close": np.random.uniform(40000, 41000, 100),
                "volume": np.random.uniform(100, 1000, 100),
            }
        )
        return data

    def test_gene_to_features_basic(self, sample_strategy_gene, sample_ohlcv_data):
        """
        基本的なGene→特徴量変換テスト

        検証項目:
        - 変換後のDataFrameが正しい形状を持つ
        - 必要な特徴量カラムが存在する
        """

        from app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        adapter = HybridFeatureAdapter()
        features_df = adapter.gene_to_features(sample_strategy_gene, sample_ohlcv_data)

        # DataFrameの形状チェック
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        assert features_df.shape[1] > 5  # 最低限の特徴量数

        # 基本的なカラムの存在チェック
        expected_columns = ["open", "high", "low", "close", "volume"]
        for col in expected_columns:
            assert col in features_df.columns

    def test_gene_to_features_with_indicators(
        self, sample_strategy_gene, sample_ohlcv_data
    ):
        """
        インジケータ情報を含む特徴量変換テスト

        検証項目:
        - インジケータパラメータが特徴量に反映される
        - SMA, RSIなどのインジケータ特徴が生成される
        """

        from app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        adapter = HybridFeatureAdapter()
        features_df = adapter.gene_to_features(sample_strategy_gene, sample_ohlcv_data)

        # インジケータ関連の特徴量が含まれることを確認
        assert "indicator_count" in features_df.columns
        assert features_df["indicator_count"].iloc[0] == 2

    def test_gene_to_features_with_conditions(
        self, sample_strategy_gene, sample_ohlcv_data
    ):
        """
        条件情報を含む特徴量変換テスト

        検証項目:
        - エントリー条件が特徴量に反映される
        - 条件の複雑さが特徴量化される
        """
        from app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        adapter = HybridFeatureAdapter()
        features_df = adapter.gene_to_features(sample_strategy_gene, sample_ohlcv_data)

        # 条件関連の特徴量が含まれることを確認
        assert "condition_count" in features_df.columns
        assert features_df["condition_count"].iloc[0] >= 1

    @patch("app.services.ml.base_ml_trainer.BaseMLTrainer._preprocess_data")
    def test_gene_to_features_with_preprocessing(
        self, mock_preprocess, sample_strategy_gene, sample_ohlcv_data
    ):
        """
        前処理を含む特徴量変換テスト

        検証項目:
        - BaseMLTrainer._preprocess_dataが呼ばれる
        - 前処理後のデータが返される
        """
        # モックの設定
        mock_preprocess.return_value = sample_ohlcv_data.copy()

        from app.services.ml.base_ml_trainer import BaseMLTrainer
        from unittest.mock import MagicMock
        
        mock_trainer_instance = MagicMock(spec=BaseMLTrainer)
        mock_trainer_instance._preprocess_data.return_value = (sample_ohlcv_data.copy(), sample_ohlcv_data.copy())

        from app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        adapter = HybridFeatureAdapter(preprocess_handler=mock_trainer_instance._preprocess_data)
        adapter.gene_to_features(
            sample_strategy_gene, sample_ohlcv_data, apply_preprocessing=True
        )

        # 前処理が呼ばれたことを確認
        assert mock_trainer_instance._preprocess_data.called

    def test_gene_to_features_invalid_gene(self, sample_ohlcv_data):
        """
        無効なGeneでのエラーハンドリングテスト

        検証項目:
        - 無効なGeneでMLFeatureErrorが発生する
        """
        from app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        adapter = HybridFeatureAdapter()

        # 無効なGene（Noneや空）でエラーが発生することを確認
        with pytest.raises(MLFeatureError):
            adapter.gene_to_features(None, sample_ohlcv_data)

    def test_gene_to_features_empty_ohlcv(self, sample_strategy_gene):
        """
        空のOHLCVデータでのエラーハンドリングテスト

        検証項目:
        - 空のOHLCVデータでMLFeatureErrorが発生する
        """
        from app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        adapter = HybridFeatureAdapter()
        empty_data = pd.DataFrame()

        # 空のデータでエラーが発生することを確認
        with pytest.raises(MLFeatureError):
            adapter.gene_to_features(sample_strategy_gene, empty_data)

    def test_gene_to_features_with_wavelet_config(
        self, sample_strategy_gene, sample_ohlcv_data
    ):
        """
        ウェーブレット設定を使用した特徴量変換テスト

        検証項目:
        - ウェーブレット設定が特徴量生成に反映される
        - ウェーブレット派生特徴量が含まれる
        """
        from app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        wavelet_config = {
            "enabled": True,
            "base_wavelet": "haar",
            "scales": [2, 4],
        }

        adapter = HybridFeatureAdapter(wavelet_config=wavelet_config)
        features_df = adapter.gene_to_features(sample_strategy_gene, sample_ohlcv_data)

        # ウェーブレット特徴量が含まれることを確認
        assert features_df.shape[1] > 10  # ウェーブレット特徴量で増加

    def test_gene_to_features_batch(self, sample_ohlcv_data):
        """
        複数Geneのバッチ変換テスト

        検証項目:
        - 複数のGeneを一度に変換できる
        - 各Geneの特徴量が正しく生成される
        """
        from app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        # 複数のGeneを作成
        genes = []
        for i in range(3):
            indicator = IndicatorGene(
                type="SMA",
                parameters={"period": 20 + i * 10},
            )
            gene = StrategyGene(
                id=f"gene_{i}",
                indicators=[indicator],
                entry_conditions=[],
                exit_conditions=[],
            )
            genes.append(gene)

        adapter = HybridFeatureAdapter()
        features_list = adapter.genes_to_features_batch(genes, sample_ohlcv_data)

        # バッチ変換の結果を確認
        assert len(features_list) == 3
        for features_df in features_list:
            assert isinstance(features_df, pd.DataFrame)
            assert len(features_df) > 0

    def test_gene_serialization_integration(self, sample_strategy_gene):
        """
        Gene SerializationとFeature変換の統合テスト

        検証項目:
        - GeneSerializer.from_listで復元したGeneが使える
        - シリアライズ→デシリアライズ→特徴量変換が動作する
        """
        from app.services.auto_strategy.serializers.gene_serialization import (
            GeneSerializer,
        )
        from app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        # Geneをシリアライズ→デシリアライズ
        serializer = GeneSerializer()
        gene_list = serializer.to_list(sample_strategy_gene)
        restored_gene = serializer.from_list(gene_list, StrategyGene)

        # 復元したGeneで特徴量変換
        adapter = HybridFeatureAdapter()
        ohlcv_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1h"),
                "open": np.random.uniform(40000, 41000, 100),
                "high": np.random.uniform(41000, 42000, 100),
                "low": np.random.uniform(39000, 40000, 100),
                "close": np.random.uniform(40000, 41000, 100),
                "volume": np.random.uniform(100, 1000, 100),
            }
        )

        features_df = adapter.gene_to_features(restored_gene, ohlcv_data)
        assert isinstance(features_df, pd.DataFrame)
