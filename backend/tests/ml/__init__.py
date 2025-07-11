"""
ML機能包括テストモジュール

オートストラテジーのML機能に対する包括的なテストを提供します。

テストモジュール構成:
- test_ml_core_functionality.py: ML基本機能テスト
- test_feature_engineering_comprehensive.py: 特徴量エンジニアリング詳細テスト
- test_ml_model_training.py: MLモデル学習・予測テスト
- test_ml_auto_strategy_integration.py: ML-オートストラテジー統合テスト
- test_ml_performance.py: パフォーマンス・スケーラビリティテスト
- test_ml_error_handling.py: エラーハンドリング・エッジケーステスト
- test_ml_prediction_accuracy.py: 予測精度・信頼性テスト
- test_ml_data_quality.py: データ品質・前処理テスト

テスト実行:
- 個別テスト: python -m pytest backend/tests/ml/test_*.py
- 全MLテスト: python backend/tests/ml/run_comprehensive_ml_tests.py
"""

__version__ = "1.0.0"
__author__ = "Trading System Development Team"

# テスト共通ユーティリティのインポート
from .utils import (
    create_sample_ohlcv_data,
    create_sample_funding_rate_data,
    create_sample_open_interest_data,
    MLTestConfig,
    PerformanceMetrics
)

__all__ = [
    "create_sample_ohlcv_data",
    "create_sample_funding_rate_data", 
    "create_sample_open_interest_data",
    "MLTestConfig",
    "PerformanceMetrics"
]
