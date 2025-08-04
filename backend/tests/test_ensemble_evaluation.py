#!/usr/bin/env python3
"""
アンサンブル評価ロジックの統合テスト

実際のアンサンブルクラスで評価ロジックの統一が正常に動作するかテストします。
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from app.services.ml.ensemble.stacking import StackingEnsemble
from app.services.ml.ensemble.bagging import BaggingEnsemble

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_simple_test_data(n_samples=50, n_features=3, n_classes=3):
    """シンプルなテストデータを作成"""
    np.random.seed(42)
    
    # 特徴量データ
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # ターゲットデータ
    y = np.random.choice(range(n_classes), size=n_samples, p=[0.5, 0.3, 0.2])
    y = pd.Series(y, name='target')
    
    return X, y


def test_stacking_ensemble_evaluation():
    """StackingEnsembleの評価ロジックテスト"""
    logger.info("=== StackingEnsemble評価ロジックテスト ===")
    
    try:
        # テストデータ作成
        X, y = create_simple_test_data(n_samples=30, n_features=3, n_classes=3)
        
        # StackingEnsembleのインスタンス作成
        ensemble = StackingEnsemble()
        
        # _evaluate_predictionsメソッドをテスト
        y_pred = np.random.choice([0, 1, 2], size=len(y))
        y_pred_proba = np.random.dirichlet([1, 1, 1], size=len(y))
        
        metrics = ensemble._evaluate_predictions(y, y_pred, y_pred_proba)
        
        # 結果確認
        assert "accuracy" in metrics, "accuracy指標が見つかりません"
        assert "balanced_accuracy" in metrics, "balanced_accuracy指標が見つかりません"
        assert "f1_score" in metrics, "f1_score指標が見つかりません"
        
        logger.info(f"✅ StackingEnsemble評価成功")
        logger.info(f"   accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"   balanced_accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        logger.info(f"   評価指標数: {len(metrics)}")
        
        return True
        
    except Exception as e:
        logger.error(f"StackingEnsemble評価テストエラー: {e}")
        return False


def test_bagging_ensemble_evaluation():
    """BaggingEnsembleの評価ロジックテスト"""
    logger.info("=== BaggingEnsemble評価ロジックテスト ===")
    
    try:
        # テストデータ作成
        X, y = create_simple_test_data(n_samples=30, n_features=3, n_classes=3)
        
        # BaggingEnsembleのインスタンス作成
        ensemble = BaggingEnsemble()
        
        # _evaluate_predictionsメソッドをテスト
        y_pred = np.random.choice([0, 1, 2], size=len(y))
        y_pred_proba = np.random.dirichlet([1, 1, 1], size=len(y))
        
        metrics = ensemble._evaluate_predictions(y, y_pred, y_pred_proba)
        
        # 結果確認
        assert "accuracy" in metrics, "accuracy指標が見つかりません"
        assert "balanced_accuracy" in metrics, "balanced_accuracy指標が見つかりません"
        assert "f1_score" in metrics, "f1_score指標が見つかりません"
        
        logger.info(f"✅ BaggingEnsemble評価成功")
        logger.info(f"   accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"   balanced_accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        logger.info(f"   評価指標数: {len(metrics)}")
        
        return True
        
    except Exception as e:
        logger.error(f"BaggingEnsemble評価テストエラー: {e}")
        return False


def test_evaluation_consistency():
    """評価ロジックの一貫性テスト"""
    logger.info("=== 評価ロジック一貫性テスト ===")
    
    try:
        # 同じテストデータで複数のアンサンブルをテスト
        X, y = create_simple_test_data(n_samples=30, n_features=3, n_classes=3)
        
        # 同じ予測結果を使用
        y_pred = np.random.choice([0, 1, 2], size=len(y))
        y_pred_proba = np.random.dirichlet([1, 1, 1], size=len(y))
        
        # 複数のアンサンブルで評価
        stacking = StackingEnsemble()
        bagging = BaggingEnsemble()
        
        stacking_metrics = stacking._evaluate_predictions(y, y_pred, y_pred_proba)
        bagging_metrics = bagging._evaluate_predictions(y, y_pred, y_pred_proba)
        
        # 同じ予測結果なので、評価指標も同じになるはず
        assert abs(stacking_metrics['accuracy'] - bagging_metrics['accuracy']) < 1e-10, "accuracy指標が一致しません"
        assert abs(stacking_metrics['balanced_accuracy'] - bagging_metrics['balanced_accuracy']) < 1e-10, "balanced_accuracy指標が一致しません"
        
        logger.info("✅ 評価ロジック一貫性確認成功")
        logger.info(f"   Stacking accuracy: {stacking_metrics['accuracy']:.6f}")
        logger.info(f"   Bagging accuracy: {bagging_metrics['accuracy']:.6f}")
        logger.info(f"   差分: {abs(stacking_metrics['accuracy'] - bagging_metrics['accuracy']):.10f}")
        
        return True
        
    except Exception as e:
        logger.error(f"評価ロジック一貫性テストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    logger.info("=" * 60)
    logger.info("🚀 アンサンブル評価ロジック統合テスト開始")
    logger.info("=" * 60)
    
    results = {}
    
    try:
        # 1. StackingEnsemble評価テスト
        results['stacking'] = test_stacking_ensemble_evaluation()
        
        # 2. BaggingEnsemble評価テスト
        results['bagging'] = test_bagging_ensemble_evaluation()
        
        # 3. 評価ロジック一貫性テスト
        results['consistency'] = test_evaluation_consistency()
        
        # 結果サマリー
        logger.info("=" * 60)
        logger.info("📊 テスト結果サマリー")
        logger.info("=" * 60)
        
        success_count = sum(results.values())
        total_count = len(results)
        
        for test_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失敗"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if success_count == total_count:
            logger.info("🎉 全テスト成功！評価ロジック統一は正常に動作しています。")
            return True
        else:
            logger.warning(f"⚠️ 一部テストが失敗しました。({total_count-success_count}個の失敗)")
            return False
            
    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
