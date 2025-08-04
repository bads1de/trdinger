"""
メトリクス統合テスト
"""

import sys
sys.path.append('.')

import numpy as np
from app.services.ml.evaluation.enhanced_metrics import (
    EnhancedMetricsCalculator,
    enhanced_metrics_calculator,
    record_metric,
    record_performance,
    evaluate_and_record_model,
    MLMetricsCollector,  # 後方互換性エイリアス
    metrics_collector,   # 後方互換性エイリアス
)

def test_enhanced_metrics_calculator():
    """統合されたEnhancedMetricsCalculatorのテスト"""
    print("=== 統合メトリクス計算器テスト ===")
    
    # インスタンス作成
    calculator = EnhancedMetricsCalculator()
    print("✅ EnhancedMetricsCalculator インスタンス作成成功")
    
    # テストデータ作成
    np.random.seed(42)
    y_true = np.random.choice([0, 1, 2], size=50)
    y_pred = np.random.choice([0, 1, 2], size=50)
    y_proba = np.random.dirichlet([1, 1, 1], size=50)
    
    # 包括的メトリクス計算
    metrics = calculator.calculate_comprehensive_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Down", "Hold", "Up"]
    )
    
    print(f"計算されたメトリクス数: {len(metrics)}")
    print(f"精度: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"F1スコア: {metrics.get('f1_score', 'N/A'):.4f}")
    print("✅ 包括的メトリクス計算成功")
    
    # メトリクス記録機能テスト
    calculator.record_metric("test_metric", 0.85, tags={"test": "true"})
    calculator.record_performance("test_operation", 100.0, memory_mb=50.0, success=True)
    print("✅ メトリクス記録機能テスト成功")
    
    # 統合評価・記録機能テスト
    result = calculator.evaluate_and_record_model(
        model_name="test_model",
        model_type="test_type",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        training_time=5.0,
        memory_usage=100.0
    )
    
    print(f"統合評価結果: accuracy={result.get('accuracy', 'N/A'):.4f}")
    print("✅ 統合評価・記録機能テスト成功")

def test_backward_compatibility():
    """後方互換性テスト"""
    print("=== 後方互換性テスト ===")
    
    # MLMetricsCollectorエイリアスのテスト
    collector = MLMetricsCollector()
    print("✅ MLMetricsCollector エイリアス動作確認")
    
    # グローバルインスタンスのテスト
    print(f"グローバルインスタンス: {type(metrics_collector).__name__}")
    print("✅ グローバルインスタンス動作確認")
    
    # 便利関数のテスト
    record_metric("compat_test", 0.95)
    record_performance("compat_operation", 50.0)
    print("✅ 便利関数動作確認")
    
    # 統合評価関数のテスト
    np.random.seed(42)
    y_true = np.random.choice([0, 1, 2], size=30)
    y_pred = np.random.choice([0, 1, 2], size=30)
    y_proba = np.random.dirichlet([1, 1, 1], size=30)
    
    result = evaluate_and_record_model(
        model_name="compat_model",
        model_type="compat_type",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba
    )
    
    print(f"後方互換評価結果: accuracy={result.get('accuracy', 'N/A'):.4f}")
    print("✅ 統合評価関数動作確認")

def test_feature_completeness():
    """機能完全性テスト"""
    print("=== 機能完全性テスト ===")
    
    calculator = enhanced_metrics_calculator
    
    # 必要なメソッドが存在することを確認
    required_methods = [
        'calculate_comprehensive_metrics',
        'record_metric',
        'record_performance', 
        'record_error',
        'record_model_evaluation_metrics',
        'evaluate_and_record_model'
    ]
    
    for method_name in required_methods:
        if hasattr(calculator, method_name):
            print(f"✅ {method_name} メソッド存在確認")
        else:
            print(f"❌ {method_name} メソッド不足")
    
    # 必要な属性が存在することを確認
    required_attributes = [
        '_metrics',
        '_performance_metrics',
        '_model_evaluation_metrics',
        '_error_counts',
        '_operation_counts'
    ]
    
    for attr_name in required_attributes:
        if hasattr(calculator, attr_name):
            print(f"✅ {attr_name} 属性存在確認")
        else:
            print(f"❌ {attr_name} 属性不足")

def main():
    """メインテスト実行"""
    print("🚀 メトリクス統合テスト開始")
    
    test_enhanced_metrics_calculator()
    test_backward_compatibility()
    test_feature_completeness()
    
    print("✅ メトリクス統合テスト完了")

if __name__ == "__main__":
    main()
