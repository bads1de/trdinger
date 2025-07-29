"""
データ前処理改善テスト

SimpleImputerを使用した高品質なデータ前処理機能を検証します。

テスト項目:
1. DataPreprocessorの基本機能
2. 欠損値補完の品質
3. 外れ値除去機能
4. 統合前処理機能
5. MLサービスでの使用確認
"""

import pytest
import logging
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from app.utils.data_preprocessing import DataPreprocessor, data_preprocessor

logger = logging.getLogger(__name__)


class TestDataPreprocessing:
    """データ前処理テストクラス"""

    def setup_method(self):
        """テストセットアップ"""
        # テスト用のサンプルデータを作成（欠損値と外れ値を含む）
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        self.sample_data = pd.DataFrame({
            'feature1': np.random.normal(100, 10, 100),
            'feature2': np.random.normal(50, 5, 100),
            'feature3': np.random.normal(0, 1, 100),
            'feature4': np.random.exponential(2, 100),
        }, index=dates)
        
        # 意図的に欠損値を作成
        self.sample_data.loc[self.sample_data.index[10:15], 'feature1'] = np.nan
        self.sample_data.loc[self.sample_data.index[20:25], 'feature2'] = np.nan
        self.sample_data.loc[self.sample_data.index[30:35], 'feature3'] = np.nan
        
        # 意図的に外れ値を作成
        self.sample_data.loc[self.sample_data.index[5], 'feature1'] = 1000  # 外れ値
        self.sample_data.loc[self.sample_data.index[15], 'feature2'] = -100  # 外れ値

    def test_basic_imputation(self):
        """1. 基本的な欠損値補完テスト"""
        logger.info("=== 基本的な欠損値補完テスト ===")

        preprocessor = DataPreprocessor()
        
        # 欠損値の数を確認
        missing_before = self.sample_data.isnull().sum().sum()
        assert missing_before > 0, "テストデータに欠損値が必要です"
        
        # 欠損値補完を実行
        result_df = preprocessor.transform_missing_values(
            self.sample_data, strategy="median"
        )
        
        # 結果の検証
        missing_after = result_df.isnull().sum().sum()
        assert missing_after == 0, "欠損値が完全に補完されていません"
        assert len(result_df) == len(self.sample_data), "行数が変わってはいけません"
        assert list(result_df.columns) == list(self.sample_data.columns), "カラムが変わってはいけません"
        
        # 補完統計情報を確認
        stats = preprocessor.get_imputation_stats()
        assert len(stats) > 0, "補完統計情報が記録されていません"
        
        logger.info(f"✅ 基本的な欠損値補完: 成功 ({missing_before} → {missing_after}個)")

    def test_different_strategies(self):
        """2. 異なる補完戦略のテスト"""
        logger.info("=== 異なる補完戦略のテスト ===")

        preprocessor = DataPreprocessor()
        strategies = ["mean", "median", "most_frequent"]
        
        for strategy in strategies:
            try:
                result_df = preprocessor.transform_missing_values(
                    self.sample_data, strategy=strategy
                )
                
                missing_count = result_df.isnull().sum().sum()
                assert missing_count == 0, f"戦略 {strategy} で欠損値が残存"
                
                logger.info(f"✅ 戦略 {strategy}: 成功")
                
            except Exception as e:
                pytest.fail(f"戦略 {strategy} でエラー: {e}")

    def test_outlier_removal(self):
        """3. 外れ値除去機能テスト"""
        logger.info("=== 外れ値除去機能テスト ===")

        preprocessor = DataPreprocessor()
        
        # 外れ値除去前の統計
        feature1_before = self.sample_data['feature1'].describe()
        
        # 外れ値除去を実行
        result_df = preprocessor._remove_outliers(
            self.sample_data, 
            columns=['feature1', 'feature2'], 
            threshold=3.0
        )
        
        # 外れ値除去後の統計
        feature1_after = result_df['feature1'].describe()
        
        # 外れ値が除去されていることを確認（最大値が小さくなっている）
        assert feature1_after['max'] < feature1_before['max'], "外れ値が除去されていません"
        
        logger.info("✅ 外れ値除去機能: 成功")

    def test_comprehensive_preprocessing(self):
        """4. 包括的前処理機能テスト"""
        logger.info("=== 包括的前処理機能テスト ===")

        preprocessor = DataPreprocessor()
        
        # 包括的前処理を実行
        result_df = preprocessor.preprocess_features(
            self.sample_data,
            imputation_strategy="median",
            scale_features=False,
            remove_outliers=True,
            outlier_threshold=3.0
        )
        
        # 結果の検証
        assert result_df.isnull().sum().sum() == 0, "欠損値が残存しています"
        assert len(result_df) == len(self.sample_data), "行数が変わってはいけません"
        assert not result_df.isin([np.inf, -np.inf]).any().any(), "無限値が残存しています"
        
        logger.info("✅ 包括的前処理機能: 成功")

    def test_scaling_feature(self):
        """5. 特徴量スケーリング機能テスト"""
        logger.info("=== 特徴量スケーリング機能テスト ===")

        preprocessor = DataPreprocessor()
        
        # スケーリングありで前処理を実行
        result_df = preprocessor.preprocess_features(
            self.sample_data,
            imputation_strategy="median",
            scale_features=True,
            remove_outliers=False
        )
        
        # スケーリング後の統計を確認
        for col in result_df.select_dtypes(include=[np.number]).columns:
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            
            # 標準化されていることを確認（平均≈0、標準偏差≈1）
            assert abs(mean_val) < 0.1, f"カラム {col} の平均が0に近くありません: {mean_val}"
            assert abs(std_val - 1.0) < 0.1, f"カラム {col} の標準偏差が1に近くありません: {std_val}"
        
        logger.info("✅ 特徴量スケーリング機能: 成功")

    def test_empty_data_handling(self):
        """6. 空データの処理テスト"""
        logger.info("=== 空データの処理テスト ===")

        preprocessor = DataPreprocessor()
        
        # 空のDataFrame
        empty_df = pd.DataFrame()
        result_df = preprocessor.transform_missing_values(empty_df)
        assert result_df.empty, "空のDataFrameは空のまま返されるべきです"
        
        # Noneの場合
        result_none = preprocessor.transform_missing_values(None)
        assert result_none is None, "Noneはそのまま返されるべきです"
        
        logger.info("✅ 空データの処理: 成功")

    def test_all_missing_column(self):
        """7. 全て欠損値のカラム処理テスト"""
        logger.info("=== 全て欠損値のカラム処理テスト ===")

        preprocessor = DataPreprocessor()
        
        # 全て欠損値のカラムを持つデータを作成
        test_data = self.sample_data.copy()
        test_data['all_missing'] = np.nan
        
        # 前処理を実行
        result_df = preprocessor.transform_missing_values(test_data, strategy="median")
        
        # 全て欠損値のカラムは0で補完されることを確認
        assert result_df['all_missing'].isnull().sum() == 0, "全て欠損値のカラムが補完されていません"
        assert (result_df['all_missing'] == 0).all(), "全て欠損値のカラムは0で補完されるべきです"
        
        logger.info("✅ 全て欠損値のカラム処理: 成功")

    def test_cache_functionality(self):
        """8. キャッシュ機能テスト"""
        logger.info("=== キャッシュ機能テスト ===")

        preprocessor = DataPreprocessor()
        
        # 最初の学習
        preprocessor.fit_imputers(self.sample_data, strategy="median")
        assert len(preprocessor.imputers) > 0, "Imputerが学習されていません"
        assert len(preprocessor.imputation_stats) > 0, "統計情報が記録されていません"
        
        # キャッシュクリア
        preprocessor.clear_cache()
        assert len(preprocessor.imputers) == 0, "Imputerキャッシュがクリアされていません"
        assert len(preprocessor.scalers) == 0, "Scalerキャッシュがクリアされていません"
        assert len(preprocessor.imputation_stats) == 0, "統計情報がクリアされていません"
        
        logger.info("✅ キャッシュ機能: 成功")

    def test_global_instance(self):
        """9. グローバルインスタンステスト"""
        logger.info("=== グローバルインスタンステスト ===")

        # グローバルインスタンスが利用可能であることを確認
        assert data_preprocessor is not None, "グローバルインスタンスが利用できません"
        assert isinstance(data_preprocessor, DataPreprocessor), "グローバルインスタンスの型が正しくありません"
        
        # グローバルインスタンスで前処理を実行
        result_df = data_preprocessor.transform_missing_values(
            self.sample_data, strategy="median"
        )
        
        assert result_df.isnull().sum().sum() == 0, "グローバルインスタンスで欠損値補完が失敗"
        
        logger.info("✅ グローバルインスタンス: 成功")

    def test_data_preprocessing_summary(self):
        """10. データ前処理機能の総合確認"""
        logger.info("=== データ前処理機能の総合確認 ===")

        summary = {
            "basic_imputation": True,
            "multiple_strategies": True,
            "outlier_removal": True,
            "comprehensive_preprocessing": True,
            "feature_scaling": True,
            "edge_case_handling": True,
            "cache_management": True,
            "global_instance": True
        }

        try:
            preprocessor = DataPreprocessor()
            
            # 基本機能の確認
            result_df = preprocessor.preprocess_features(
                self.sample_data,
                imputation_strategy="median",
                scale_features=False,
                remove_outliers=True
            )
            
            # 品質確認
            assert result_df.isnull().sum().sum() == 0, "欠損値が残存"
            assert not result_df.isin([np.inf, -np.inf]).any().any(), "無限値が残存"
            assert len(result_df) == len(self.sample_data), "データサイズが変更"
            
            # 統計情報の確認
            stats = preprocessor.get_imputation_stats()
            assert isinstance(stats, dict), "統計情報の形式が正しくない"
            
            logger.info("✅ データ前処理機能の総合確認: 成功")
            logger.info(f"機能サマリー: {summary}")
            
        except Exception as e:
            pytest.fail(f"データ前処理機能の確認でエラーが発生しました: {e}")


if __name__ == "__main__":
    # テストを直接実行する場合
    pytest.main([__file__, "-v", "--tb=short"])
