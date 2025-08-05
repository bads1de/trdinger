"""
改善されたラベル生成システムのテスト

動的ボラティリティベースのラベル生成が
クラス不均衡問題を解決するかを検証する。
"""

import pytest
import pandas as pd
import numpy as np
import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.label_generation import LabelGenerator, ThresholdMethod
from app.services.ml.config.ml_config import TrainingConfig

logger = logging.getLogger(__name__)


class TestImprovedLabelGeneration:
    """改善されたラベル生成システムのテストクラス"""

    def generate_sample_data(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
        np.random.seed(42)
        
        # 現実的な価格データを生成
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, len(dates))  # 2%の標準偏差
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # OHLCV データを生成
        data = {
            'timestamp': dates,
            'Open': prices * np.random.uniform(0.995, 1.005, len(prices)),
            'High': prices * np.random.uniform(1.001, 1.02, len(prices)),
            'Low': prices * np.random.uniform(0.98, 0.999, len(prices)),
            'Close': prices,
            'Volume': np.random.uniform(100, 1000, len(prices))
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def test_dynamic_volatility_label_generation(self):
        """動的ボラティリティベースのラベル生成をテスト"""
        logger.info("=== 動的ボラティリティベースのラベル生成テスト ===")
        
        # サンプルデータを生成
        sample_data = self.generate_sample_data()
        
        # ラベル生成器を初期化
        label_generator = LabelGenerator()
        
        # 動的ボラティリティベースでラベルを生成
        labels_dynamic, threshold_info_dynamic = label_generator.generate_labels(
            sample_data['Close'],
            method=ThresholdMethod.DYNAMIC_VOLATILITY,
            volatility_window=24,
            threshold_multiplier=0.5,
            min_threshold=0.005,
            max_threshold=0.05
        )
        
        # 固定閾値でラベルを生成（比較用）
        labels_fixed, threshold_info_fixed = label_generator.generate_labels(
            sample_data['Close'],
            method=ThresholdMethod.FIXED,
            threshold=0.02
        )
        
        # ラベル分布を分析
        def analyze_distribution(labels, method_name):
            label_counts = labels.value_counts().sort_index()
            total = len(labels)
            
            distribution = {
                'down': label_counts.get(0, 0) / total,
                'range': label_counts.get(1, 0) / total,
                'up': label_counts.get(2, 0) / total
            }
            
            # クラス不均衡の度合いを計算
            ratios = [distribution['down'], distribution['range'], distribution['up']]
            max_ratio = max(ratios)
            min_ratio = min([r for r in ratios if r > 0])
            imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
            
            logger.info(f"{method_name}ラベル分布:")
            logger.info(f"  下落: {distribution['down']:.3f} ({label_counts.get(0, 0)}個)")
            logger.info(f"  レンジ: {distribution['range']:.3f} ({label_counts.get(1, 0)}個)")
            logger.info(f"  上昇: {distribution['up']:.3f} ({label_counts.get(2, 0)}個)")
            logger.info(f"  クラス不均衡比率: {imbalance_ratio:.2f}")
            
            return distribution, imbalance_ratio
        
        # 分布を分析
        dynamic_dist, dynamic_imbalance = analyze_distribution(labels_dynamic, "動的ボラティリティ")
        fixed_dist, fixed_imbalance = analyze_distribution(labels_fixed, "固定閾値")
        
        # 改善効果を評価
        improvement_ratio = fixed_imbalance / dynamic_imbalance
        logger.info(f"クラス不均衡改善比率: {improvement_ratio:.2f}倍")
        
        # 閾値情報を表示
        logger.info(f"動的閾値情報: {threshold_info_dynamic['description']}")
        logger.info(f"固定閾値情報: {threshold_info_fixed['description']}")
        
        # 動的閾値の統計情報を表示
        if 'volatility_stats' in threshold_info_dynamic:
            vol_stats = threshold_info_dynamic['volatility_stats']
            logger.info(f"ボラティリティ統計:")
            logger.info(f"  平均: {vol_stats['mean_volatility']:.6f}")
            logger.info(f"  標準偏差: {vol_stats['std_volatility']:.6f}")
            logger.info(f"  範囲: {vol_stats['min_volatility']:.6f} - {vol_stats['max_volatility']:.6f}")
        
        if 'threshold_stats' in threshold_info_dynamic:
            thresh_stats = threshold_info_dynamic['threshold_stats']
            logger.info(f"閾値統計:")
            logger.info(f"  平均閾値: {thresh_stats['mean_threshold']:.6f}")
            logger.info(f"  閾値範囲: {thresh_stats['min_threshold_used']:.6f} - {thresh_stats['max_threshold_used']:.6f}")
        
        # 改善の検証
        assert dynamic_imbalance < fixed_imbalance, f"動的閾値でクラス不均衡が改善されませんでした（動的: {dynamic_imbalance:.2f}, 固定: {fixed_imbalance:.2f}）"
        assert dynamic_imbalance < 2.0, f"動的閾値でもクラス不均衡が残っています（比率: {dynamic_imbalance:.2f}）"
        
        logger.info("✅ 動的ボラティリティベースのラベル生成により、クラス不均衡が改善されました")
        
        return {
            'dynamic_distribution': dynamic_dist,
            'fixed_distribution': fixed_dist,
            'dynamic_imbalance': dynamic_imbalance,
            'fixed_imbalance': fixed_imbalance,
            'improvement_ratio': improvement_ratio,
            'threshold_info_dynamic': threshold_info_dynamic,
            'threshold_info_fixed': threshold_info_fixed
        }

    def test_config_integration(self):
        """設定ファイルとの統合をテスト"""
        logger.info("=== 設定ファイル統合テスト ===")
        
        # 設定を読み込み
        config = TrainingConfig()
        
        # 新しい設定項目が正しく設定されているかを確認
        assert hasattr(config, 'LABEL_METHOD'), "LABEL_METHOD設定が見つかりません"
        assert hasattr(config, 'VOLATILITY_WINDOW'), "VOLATILITY_WINDOW設定が見つかりません"
        assert hasattr(config, 'THRESHOLD_MULTIPLIER'), "THRESHOLD_MULTIPLIER設定が見つかりません"
        assert hasattr(config, 'MIN_THRESHOLD'), "MIN_THRESHOLD設定が見つかりません"
        assert hasattr(config, 'MAX_THRESHOLD'), "MAX_THRESHOLD設定が見つかりません"
        
        logger.info(f"ラベル生成方法: {config.LABEL_METHOD}")
        logger.info(f"ボラティリティウィンドウ: {config.VOLATILITY_WINDOW}")
        logger.info(f"閾値乗数: {config.THRESHOLD_MULTIPLIER}")
        logger.info(f"最小閾値: {config.MIN_THRESHOLD}")
        logger.info(f"最大閾値: {config.MAX_THRESHOLD}")
        
        # デフォルト値が動的ボラティリティベースになっているかを確認
        assert config.LABEL_METHOD == "dynamic_volatility", f"デフォルトのラベル生成方法が期待値と異なります: {config.LABEL_METHOD}"
        
        logger.info("✅ 設定ファイルの統合が正常に動作しています")
        
        return {
            'label_method': config.LABEL_METHOD,
            'volatility_window': config.VOLATILITY_WINDOW,
            'threshold_multiplier': config.THRESHOLD_MULTIPLIER,
            'min_threshold': config.MIN_THRESHOLD,
            'max_threshold': config.MAX_THRESHOLD
        }

    def test_different_market_conditions(self):
        """異なる市場状況でのラベル生成をテスト"""
        logger.info("=== 異なる市場状況でのテスト ===")
        
        label_generator = LabelGenerator()
        results = {}
        
        # 1. 低ボラティリティ市場
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        np.random.seed(42)
        low_vol_prices = 50000 * (1 + np.cumsum(np.random.normal(0, 0.005, len(dates))))  # 0.5%標準偏差
        low_vol_data = pd.Series(low_vol_prices, index=dates)
        
        labels_low_vol, info_low_vol = label_generator.generate_labels(
            low_vol_data, method=ThresholdMethod.DYNAMIC_VOLATILITY
        )
        
        # 2. 高ボラティリティ市場
        high_vol_prices = 50000 * (1 + np.cumsum(np.random.normal(0, 0.05, len(dates))))  # 5%標準偏差
        high_vol_data = pd.Series(high_vol_prices, index=dates)
        
        labels_high_vol, info_high_vol = label_generator.generate_labels(
            high_vol_data, method=ThresholdMethod.DYNAMIC_VOLATILITY
        )
        
        # 分布を分析
        for market_type, labels, info in [
            ("低ボラティリティ", labels_low_vol, info_low_vol),
            ("高ボラティリティ", labels_high_vol, info_high_vol)
        ]:
            label_counts = labels.value_counts().sort_index()
            total = len(labels)
            
            distribution = {
                'down': label_counts.get(0, 0) / total,
                'range': label_counts.get(1, 0) / total,
                'up': label_counts.get(2, 0) / total
            }
            
            ratios = [distribution['down'], distribution['range'], distribution['up']]
            max_ratio = max(ratios)
            min_ratio = min([r for r in ratios if r > 0])
            imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
            
            logger.info(f"{market_type}市場:")
            logger.info(f"  分布: 下落={distribution['down']:.3f}, レンジ={distribution['range']:.3f}, 上昇={distribution['up']:.3f}")
            logger.info(f"  不均衡比率: {imbalance_ratio:.2f}")
            logger.info(f"  平均閾値: {info.get('threshold_stats', {}).get('mean_threshold', 'N/A')}")
            
            results[market_type] = {
                'distribution': distribution,
                'imbalance_ratio': imbalance_ratio,
                'threshold_info': info
            }
        
        # 動的閾値が市場状況に適応していることを確認
        low_vol_threshold = results["低ボラティリティ"]['threshold_info'].get('threshold_stats', {}).get('mean_threshold', 0)
        high_vol_threshold = results["高ボラティリティ"]['threshold_info'].get('threshold_stats', {}).get('mean_threshold', 0)
        
        if low_vol_threshold and high_vol_threshold:
            logger.info(f"閾値適応: 低ボラ={low_vol_threshold:.6f}, 高ボラ={high_vol_threshold:.6f}")
            assert high_vol_threshold > low_vol_threshold, "高ボラティリティ市場で閾値が適応的に調整されていません"
        
        logger.info("✅ 動的閾値が異なる市場状況に適応しています")
        
        return results

    def test_overall_improvement(self):
        """全体的な改善効果をテスト"""
        logger.info("=== 全体的な改善効果の検証 ===")
        
        # 各テストを実行
        label_test_results = self.test_dynamic_volatility_label_generation()
        config_test_results = self.test_config_integration()
        market_test_results = self.test_different_market_conditions()
        
        # 改善スコアを計算
        improvement_score = 0
        
        # ラベル生成改善（最大40点）
        improvement_ratio = label_test_results['improvement_ratio']
        if improvement_ratio > 3:
            improvement_score += 40
        elif improvement_ratio > 2:
            improvement_score += 30
        elif improvement_ratio > 1.5:
            improvement_score += 20
        elif improvement_ratio > 1:
            improvement_score += 10
        
        # クラス不均衡解消（最大30点）
        dynamic_imbalance = label_test_results['dynamic_imbalance']
        if dynamic_imbalance < 1.5:
            improvement_score += 30
        elif dynamic_imbalance < 2:
            improvement_score += 20
        elif dynamic_imbalance < 3:
            improvement_score += 10
        
        # 設定統合（最大15点）
        if config_test_results['label_method'] == "dynamic_volatility":
            improvement_score += 15
        
        # 市場適応性（最大15点）
        market_adaptability = len([r for r in market_test_results.values() if r['imbalance_ratio'] < 2])
        if market_adaptability == 2:
            improvement_score += 15
        elif market_adaptability == 1:
            improvement_score += 10
        
        logger.info(f"ラベル生成改善スコア: {improvement_score}/100")
        
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
            'label_test_results': label_test_results,
            'config_test_results': config_test_results,
            'market_test_results': market_test_results
        }


if __name__ == "__main__":
    # テストを直接実行する場合
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_instance = TestImprovedLabelGeneration()
    
    # 全体的な改善効果を検証
    results = test_instance.test_overall_improvement()
    
    print(f"\n=== ラベル生成改善結果サマリー ===")
    print(f"改善スコア: {results['improvement_score']}/100")
    print(f"クラス不均衡改善比率: {results['label_test_results']['improvement_ratio']:.2f}倍")
    print(f"動的閾値の不均衡比率: {results['label_test_results']['dynamic_imbalance']:.2f}")
    print(f"固定閾値の不均衡比率: {results['label_test_results']['fixed_imbalance']:.2f}")
