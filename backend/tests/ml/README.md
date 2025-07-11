# ML機能包括テストスイート

オートストラテジーのML機能に対する包括的なテストを提供します。

## 📁 テストファイル構成

### 🔧 基本機能テスト
- **`test_ml_core_functionality.py`** - ML基本機能テスト
  - MLIndicatorService, MLSignalGenerator, FeatureEngineeringServiceの基本動作
  - モデル初期化、予測、指標計算の正確性検証

### 🧪 詳細機能テスト
- **`test_feature_engineering_comprehensive.py`** - 特徴量エンジニアリング詳細テスト
  - 特徴量生成の正確性、データ品質、欠損値処理、異常値検出
- **`test_ml_model_training.py`** - MLモデル学習・予測テスト
  - モデル学習プロセス、予測精度、モデル保存・読み込み、バージョン管理

### 🔗 統合テスト
- **`test_ml_auto_strategy_integration.py`** - ML-オートストラテジー統合テスト
  - ML指標とGA戦略生成の統合、条件生成でのML指標使用

### ⚡ パフォーマンステスト
- **`test_ml_performance.py`** - パフォーマンス・スケーラビリティテスト
  - 大量データ処理速度、メモリ使用量、並列処理、リアルタイム予測性能
- **`test_ml_prediction_accuracy.py`** - 予測精度・信頼性テスト
  - 予測精度の統計的検証、信頼区間計算、予測の一貫性

### 🛡️ 堅牢性テスト
- **`test_ml_error_handling.py`** - エラーハンドリング・エッジケーステスト
  - 不正データ入力、モデル未学習状態、メモリ不足、ネットワークエラー
- **`test_ml_data_quality.py`** - データ品質・前処理テスト
  - 入力データ検証、前処理パイプライン、データ正規化、外れ値処理

### 🎯 統合実行
- **`run_comprehensive_ml_tests.py`** - 包括テストランナー
  - 全MLテストの統合実行、詳細レポート生成、パフォーマンス測定

## 🚀 テスト実行方法

### 個別テスト実行
```bash
# 基本機能テスト
python -m pytest backend/tests/ml/test_ml_core_functionality.py -v

# 特定のテストクラス
python -m pytest backend/tests/ml/test_ml_performance.py::MLPerformanceTestSuite -v

# 特定のテストメソッド
python -m pytest backend/tests/ml/test_ml_error_handling.py::test_invalid_data_handling -v
```

### 全MLテスト実行
```bash
# 包括テストランナー使用（推奨）
python backend/tests/ml/run_comprehensive_ml_tests.py

# pytest使用
python -m pytest backend/tests/ml/ -v
```

### パフォーマンステスト実行
```bash
# パフォーマンス測定付き
python backend/tests/ml/test_ml_performance.py --benchmark

# メモリ使用量監視付き
python backend/tests/ml/test_ml_performance.py --memory-profile
```

## 📊 テスト対象範囲

### ✅ 機能テスト
- [x] ML指標計算の正確性
- [x] 特徴量エンジニアリング品質
- [x] モデル学習・予測性能
- [x] オートストラテジー統合

### ✅ 品質テスト
- [x] データ品質保証
- [x] エラーハンドリング
- [x] 予測精度・信頼性
- [x] 一貫性・安定性

### ✅ パフォーマンステスト
- [x] 処理速度・スループット
- [x] メモリ使用効率
- [x] スケーラビリティ
- [x] リアルタイム性能

## 🔧 テスト設定

### 設定ファイル
テスト設定は `utils.py` の `MLTestConfig` クラスで管理：

```python
@dataclass
class MLTestConfig:
    sample_size: int = 1000          # サンプルデータサイズ
    prediction_horizon: int = 24     # 予測期間（時間）
    threshold_up: float = 0.02       # 上昇判定閾値
    threshold_down: float = -0.02    # 下落判定閾値
    test_train_split: float = 0.8    # 学習・テスト分割比
    random_seed: int = 42            # 乱数シード
    performance_timeout: float = 30.0 # パフォーマンステストタイムアウト
    memory_limit_mb: int = 1000      # メモリ使用量上限
```

### 共通ユーティリティ
- `create_sample_ohlcv_data()` - サンプルOHLCVデータ生成
- `create_sample_funding_rate_data()` - サンプルファンディングレートデータ生成
- `create_sample_open_interest_data()` - サンプル建玉残高データ生成
- `measure_performance()` - パフォーマンス測定
- `validate_ml_predictions()` - ML予測結果検証

## 📈 テスト結果レポート

包括テストランナーは以下の詳細レポートを生成：

1. **テスト実行サマリー** - 成功/失敗数、実行時間
2. **パフォーマンスメトリクス** - 処理速度、メモリ使用量、スループット
3. **予測精度レポート** - 精度統計、信頼区間、一貫性指標
4. **エラー分析** - 失敗テストの詳細、推奨対策
5. **可視化グラフ** - パフォーマンス推移、精度分布

## 🔄 CI/CD統合

### GitHub Actions設定例
```yaml
- name: Run ML Tests
  run: |
    python backend/tests/ml/run_comprehensive_ml_tests.py --ci-mode
    
- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: ml-test-reports
    path: backend/tests/ml/reports/
```

## 🎯 テスト品質基準

### 成功基準
- **基本機能テスト**: 100% 成功
- **統合テスト**: 95% 以上成功
- **パフォーマンステスト**: 設定閾値内
- **予測精度**: ベースライン以上
- **メモリ使用量**: 制限値以下

### 警告基準
- **実行時間**: 設定タイムアウトの80%超過
- **メモリ使用量**: 制限値の80%超過
- **予測精度**: ベースラインの90%未満

## 📞 サポート

テストに関する質問や問題は開発チームまでお問い合わせください。

---
*最終更新: 2025-07-11*
