"""
改善されたデータ前処理パイプラインのテスト

IQRベースの外れ値検出により
金融時系列データの重要なシグナルが保持されるかを検証する。
"""

import pytest
import pandas as pd
import numpy as np
import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class TestImprovedDataPreprocessing:
    """改善されたデータ前処理パイプラインのテストクラス"""

    def generate_financial_data_with_events(self):
        """金融イベントを含むテスト用データを生成"""
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
        np.random.seed(42)
        
        # 基本的な価格変動
        base_returns = np.random.normal(0, 0.01, len(dates))  # 1%の標準偏差
        
        # 重要な金融イベントを追加
        # 1. 市場クラッシュ（-15%の急落）
        crash_index = 200
        base_returns[crash_index] = -0.15
        
        # 2. 急激な回復（+12%の上昇）
        recovery_index = 205
        base_returns[recovery_index] = 0.12
        
        # 3. ボラティリティスパイク（±8%の変動）
        spike_indices = [400, 401, 402]
        base_returns[spike_indices] = [0.08, -0.08, 0.08]
        
        # 4. ニュースイベント（±5%の変動）
        news_indices = [600, 700, 800]
        base_returns[news_indices] = [0.05, -0.05, 0.05]
        
        # 価格を計算
        prices = [50000]
        for ret in base_returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # 特徴量データを生成
        data = {
            'price_change': base_returns,
            'volume': np.random.lognormal(10, 1, len(dates)),  # 対数正規分布
            'spread': np.random.exponential(0.001, len(dates)),  # 指数分布
            'volatility': np.abs(base_returns) * np.random.uniform(0.8, 1.2, len(dates)),
            'momentum': pd.Series(base_returns).rolling(5).mean().fillna(0),
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # 重要なイベントのインデックスを記録
        important_events = {
            'crash': crash_index,
            'recovery': recovery_index,
            'volatility_spikes': spike_indices,
            'news_events': news_indices
        }
        
        return df, important_events

    def test_iqr_vs_zscore_outlier_detection(self):
        """IQRベースとZ-scoreベースの外れ値検出を比較"""
        logger.info("=== IQRベース vs Z-scoreベース外れ値検出の比較 ===")
        
        # 金融イベントを含むデータを生成
        financial_data, important_events = self.generate_financial_data_with_events()
        
        preprocessor = DataPreprocessor()
        
        # Z-scoreベースの外れ値検出
        data_zscore = preprocessor.preprocess_features(
            financial_data.copy(),
            remove_outliers=True,
            outlier_threshold=3.0,
            outlier_method="zscore",
            scale_features=False
        )
        
        # IQRベースの外れ値検出
        data_iqr = preprocessor.preprocess_features(
            financial_data.copy(),
            remove_outliers=True,
            outlier_threshold=3.0,
            outlier_method="iqr",
            scale_features=False
        )
        
        # 重要なイベントの保持率を計算
        def calculate_event_preservation(original_data, processed_data, events):
            preservation_stats = {}
            
            for event_type, indices in events.items():
                if isinstance(indices, list):
                    preserved_count = 0
                    total_count = len(indices)
                    
                    for idx in indices:
                        if idx < len(processed_data):
                            # price_changeカラムで確認
                            if not pd.isna(processed_data.iloc[idx]['price_change']):
                                preserved_count += 1
                    
                    preservation_rate = preserved_count / total_count if total_count > 0 else 0
                    preservation_stats[event_type] = {
                        'preserved': preserved_count,
                        'total': total_count,
                        'rate': preservation_rate
                    }
                else:
                    # 単一のインデックス
                    idx = indices
                    if idx < len(processed_data):
                        preserved = not pd.isna(processed_data.iloc[idx]['price_change'])
                        preservation_stats[event_type] = {
                            'preserved': 1 if preserved else 0,
                            'total': 1,
                            'rate': 1.0 if preserved else 0.0
                        }
            
            return preservation_stats
        
        # 各方法での重要イベント保持率を計算
        zscore_preservation = calculate_event_preservation(financial_data, data_zscore, important_events)
        iqr_preservation = calculate_event_preservation(financial_data, data_iqr, important_events)
        
        logger.info("重要な金融イベントの保持率:")
        logger.info("Z-scoreベース:")
        total_zscore_preserved = 0
        total_zscore_events = 0
        for event_type, stats in zscore_preservation.items():
            logger.info(f"  {event_type}: {stats['preserved']}/{stats['total']} ({stats['rate']*100:.1f}%)")
            total_zscore_preserved += stats['preserved']
            total_zscore_events += stats['total']
        
        logger.info("IQRベース:")
        total_iqr_preserved = 0
        total_iqr_events = 0
        for event_type, stats in iqr_preservation.items():
            logger.info(f"  {event_type}: {stats['preserved']}/{stats['total']} ({stats['rate']*100:.1f}%)")
            total_iqr_preserved += stats['preserved']
            total_iqr_events += stats['total']
        
        # 全体的な保持率
        zscore_overall_rate = total_zscore_preserved / total_zscore_events if total_zscore_events > 0 else 0
        iqr_overall_rate = total_iqr_preserved / total_iqr_events if total_iqr_events > 0 else 0
        
        logger.info(f"全体的な重要イベント保持率:")
        logger.info(f"  Z-score: {zscore_overall_rate*100:.1f}%")
        logger.info(f"  IQR: {iqr_overall_rate*100:.1f}%")
        
        # 通常データの除去率も確認
        original_count = len(financial_data)
        zscore_remaining = data_zscore.dropna().shape[0]
        iqr_remaining = data_iqr.dropna().shape[0]
        
        zscore_removal_rate = (original_count - zscore_remaining) / original_count
        iqr_removal_rate = (original_count - iqr_remaining) / original_count
        
        logger.info(f"データ除去率:")
        logger.info(f"  Z-score: {zscore_removal_rate*100:.1f}%")
        logger.info(f"  IQR: {iqr_removal_rate*100:.1f}%")
        
        # IQRの方が重要イベントを保持しているかを確認
        if iqr_overall_rate > zscore_overall_rate:
            logger.info("✅ IQRベースの外れ値検出により、重要な金融イベントがより多く保持されました")
        else:
            logger.warning("⚠️ IQRベースの改善効果が期待より低いです")
        
        return {
            'zscore_preservation': zscore_preservation,
            'iqr_preservation': iqr_preservation,
            'zscore_overall_rate': zscore_overall_rate,
            'iqr_overall_rate': iqr_overall_rate,
            'zscore_removal_rate': zscore_removal_rate,
            'iqr_removal_rate': iqr_removal_rate
        }

    def test_financial_data_characteristics(self):
        """金融データの特性に適した前処理をテスト"""
        logger.info("=== 金融データ特性に適した前処理のテスト ===")
        
        # 異なる分布を持つ金融データを生成
        np.random.seed(42)
        n_samples = 1000
        
        financial_features = pd.DataFrame({
            # 正規分布に近い特徴量（価格変化率）
            'returns': np.random.normal(0, 0.02, n_samples),
            
            # 対数正規分布（出来高）
            'volume': np.random.lognormal(10, 1, n_samples),
            
            # 指数分布（スプレッド）
            'spread': np.random.exponential(0.001, n_samples),
            
            # 重い尾を持つ分布（ボラティリティ）
            'volatility': np.random.standard_t(3, n_samples) * 0.01,
            
            # 歪んだ分布（建玉残高変化）
            'oi_change': np.random.gamma(2, 0.01, n_samples) - 0.02,
        })
        
        preprocessor = DataPreprocessor()
        
        # 異なる外れ値検出方法でテスト
        methods = ['iqr', 'zscore']
        results = {}
        
        for method in methods:
            processed_data = preprocessor.preprocess_features(
                financial_features.copy(),
                remove_outliers=True,
                outlier_threshold=3.0,
                outlier_method=method,
                scale_features=False
            )
            
            # 各特徴量の保持率を計算
            preservation_rates = {}
            for col in financial_features.columns:
                original_count = financial_features[col].notna().sum()
                preserved_count = processed_data[col].notna().sum()
                preservation_rate = preserved_count / original_count if original_count > 0 else 0
                preservation_rates[col] = preservation_rate
            
            results[method] = {
                'data': processed_data,
                'preservation_rates': preservation_rates,
                'overall_preservation': np.mean(list(preservation_rates.values()))
            }
            
            logger.info(f"{method.upper()}法の特徴量保持率:")
            for col, rate in preservation_rates.items():
                logger.info(f"  {col}: {rate*100:.1f}%")
            logger.info(f"  全体平均: {results[method]['overall_preservation']*100:.1f}%")
        
        # IQRの方が金融データに適しているかを確認
        iqr_preservation = results['iqr']['overall_preservation']
        zscore_preservation = results['zscore']['overall_preservation']
        
        if iqr_preservation > zscore_preservation:
            logger.info("✅ IQRベースの方が金融データの特性に適しています")
        else:
            logger.warning("⚠️ 期待された改善効果が見られません")
        
        return results

    def test_data_quality_monitoring(self):
        """データ品質モニタリング機能をテスト"""
        logger.info("=== データ品質モニタリングのテスト ===")
        
        # 品質問題を含むデータを生成
        np.random.seed(42)
        problematic_data = pd.DataFrame({
            'normal_feature': np.random.normal(0, 1, 100),
            'missing_heavy': np.concatenate([np.random.normal(0, 1, 50), [np.nan] * 50]),
            'outlier_heavy': np.concatenate([np.random.normal(0, 1, 90), [100, -100] * 5]),
            'zero_variance': [1.0] * 100,
            'infinite_values': np.concatenate([np.random.normal(0, 1, 95), [np.inf, -np.inf, np.nan, np.inf, -np.inf]])
        })
        
        preprocessor = DataPreprocessor()
        
        # 前処理前の品質チェック
        def analyze_data_quality(df, label):
            logger.info(f"{label}のデータ品質:")
            
            for col in df.columns:
                series = df[col]
                total_count = len(series)
                
                missing_count = series.isna().sum()
                infinite_count = np.isinf(series).sum()
                zero_var = series.var() == 0 if series.notna().sum() > 1 else True
                
                logger.info(f"  {col}:")
                logger.info(f"    欠損値: {missing_count}/{total_count} ({missing_count/total_count*100:.1f}%)")
                logger.info(f"    無限値: {infinite_count}")
                logger.info(f"    分散ゼロ: {zero_var}")
                
                if series.notna().sum() > 0:
                    logger.info(f"    範囲: {series.min():.3f} - {series.max():.3f}")
        
        analyze_data_quality(problematic_data, "前処理前")
        
        # IQRベースで前処理
        processed_data = preprocessor.preprocess_features(
            problematic_data.copy(),
            remove_outliers=True,
            outlier_threshold=3.0,
            outlier_method="iqr",
            scale_features=True,
            scaling_method="robust"
        )
        
        analyze_data_quality(processed_data, "前処理後")
        
        # 品質改善の評価
        quality_improvements = {}
        
        for col in problematic_data.columns:
            if col in processed_data.columns:
                # 欠損値の改善
                original_missing = problematic_data[col].isna().sum()
                processed_missing = processed_data[col].isna().sum()
                
                # 無限値の改善
                original_infinite = np.isinf(problematic_data[col]).sum()
                processed_infinite = np.isinf(processed_data[col]).sum()
                
                quality_improvements[col] = {
                    'missing_improved': original_missing > processed_missing,
                    'infinite_improved': original_infinite > processed_infinite,
                    'has_finite_values': processed_data[col].notna().sum() > 0
                }
        
        # 全体的な品質改善を評価
        total_improvements = sum([
            sum([
                improvements['missing_improved'],
                improvements['infinite_improved'],
                improvements['has_finite_values']
            ]) for improvements in quality_improvements.values()
        ])
        
        max_possible_improvements = len(quality_improvements) * 3
        improvement_rate = total_improvements / max_possible_improvements if max_possible_improvements > 0 else 0
        
        logger.info(f"データ品質改善率: {improvement_rate*100:.1f}%")
        
        if improvement_rate > 0.7:
            logger.info("✅ データ品質が大幅に改善されました")
        elif improvement_rate > 0.5:
            logger.info("✅ データ品質が改善されました")
        else:
            logger.warning("⚠️ データ品質の改善が限定的です")
        
        return {
            'original_data': problematic_data,
            'processed_data': processed_data,
            'quality_improvements': quality_improvements,
            'improvement_rate': improvement_rate
        }

    def test_overall_preprocessing_improvement(self):
        """全体的な前処理改善をテスト"""
        logger.info("=== 全体的な前処理改善の検証 ===")
        
        # 各テストを実行
        outlier_results = self.test_iqr_vs_zscore_outlier_detection()
        financial_results = self.test_financial_data_characteristics()
        quality_results = self.test_data_quality_monitoring()
        
        # 改善スコアを計算
        improvement_score = 0
        
        # 重要イベント保持（最大40点）
        iqr_event_preservation = outlier_results['iqr_overall_rate']
        zscore_event_preservation = outlier_results['zscore_overall_rate']
        
        if iqr_event_preservation > zscore_event_preservation:
            improvement_score += 40
        elif iqr_event_preservation >= zscore_event_preservation * 0.9:
            improvement_score += 30
        elif iqr_event_preservation >= zscore_event_preservation * 0.8:
            improvement_score += 20
        
        # 金融データ適応性（最大30点）
        iqr_financial_preservation = financial_results['iqr']['overall_preservation']
        zscore_financial_preservation = financial_results['zscore']['overall_preservation']
        
        if iqr_financial_preservation > zscore_financial_preservation:
            improvement_score += 30
        elif iqr_financial_preservation >= zscore_financial_preservation * 0.95:
            improvement_score += 20
        elif iqr_financial_preservation >= zscore_financial_preservation * 0.9:
            improvement_score += 10
        
        # データ品質改善（最大30点）
        quality_improvement_rate = quality_results['improvement_rate']
        if quality_improvement_rate > 0.8:
            improvement_score += 30
        elif quality_improvement_rate > 0.6:
            improvement_score += 20
        elif quality_improvement_rate > 0.4:
            improvement_score += 10
        
        logger.info(f"データ前処理改善スコア: {improvement_score}/100")
        
        if improvement_score >= 80:
            logger.info("🎉 優秀な改善効果が確認されました")
        elif improvement_score >= 60:
            logger.info("✅ 良好な改善効果が確認されました")
        elif improvement_score >= 40:
            logger.info("⚠️ 部分的な改善効果が確認されました")
        else:
            logger.warning("❌ 改善効果が不十分です")
        
        return {
            'improvement_score': improvement_score,
            'outlier_results': outlier_results,
            'financial_results': financial_results,
            'quality_results': quality_results
        }


if __name__ == "__main__":
    # テストを直接実行する場合
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_instance = TestImprovedDataPreprocessing()
    
    # 全体的な改善効果を検証
    results = test_instance.test_overall_preprocessing_improvement()
    
    print(f"\n=== データ前処理改善結果サマリー ===")
    print(f"改善スコア: {results['improvement_score']}/100")
    print(f"重要イベント保持率（IQR）: {results['outlier_results']['iqr_overall_rate']*100:.1f}%")
    print(f"重要イベント保持率（Z-score）: {results['outlier_results']['zscore_overall_rate']*100:.1f}%")
    print(f"データ品質改善率: {results['quality_results']['improvement_rate']*100:.1f}%")
