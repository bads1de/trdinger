# アンサンブル学習ハイパーパラメータ最適化 実装サマリー

## 実装概要

アンサンブル学習（LightGBM、XGBoost、RandomForest、CatBoost、TabNet）に対応したハイパーパラメータ最適化機能を実装しました。

## 実装されたファイル

### 1. コア実装

#### `backend/app/core/services/optimization/ensemble_parameter_space.py`

- **目的**: アンサンブル学習用パラメータ空間の定義
- **機能**:
  - 各モデル（LightGBM、XGBoost、RandomForest、CatBoost、TabNet）のパラメータ空間
  - バギング・スタッキング固有のパラメータ空間
  - 動的なパラメータ空間構築

#### `backend/app/core/services/optimization/optuna_optimizer.py` (拡張)

- **追加機能**: `get_ensemble_parameter_space()` メソッド
- **目的**: アンサンブル用パラメータ空間の取得

#### `backend/app/core/services/ml/ml_training_service.py` (拡張)

- **追加機能**: アンサンブルトレーナー検出時の自動パラメータ空間選択
- **改善点**: 最適化設定でのアンサンブル対応

#### `backend/app/core/services/ml/ensemble/ensemble_trainer.py` (拡張)

- **追加機能**:
  - `_extract_optimized_parameters()` メソッド
  - 最適化されたパラメータの分離と適用
- **対応モデル**: LightGBM、XGBoost、RandomForest、CatBoost、TabNet

#### `backend/app/core/services/ml/ensemble/bagging.py` (拡張)

- **追加機能**: `base_model_params` パラメータ対応

#### `backend/app/core/services/ml/ensemble/stacking.py` (拡張)

- **追加機能**: `base_model_params` パラメータ対応

### 2. テスト実装

#### `backend/tests/unit/test_ensemble_parameter_space.py`

- **目的**: EnsembleParameterSpace のユニットテスト
- **カバレッジ**: 全モデルのパラメータ空間、統合機能

#### `backend/tests/unit/test_ensemble_trainer_optimization.py`

- **目的**: EnsembleTrainer の最適化機能テスト
- **カバレッジ**: パラメータ抽出、分離機能

#### `backend/tests/integration/test_ensemble_optimization.py`

- **目的**: アンサンブル最適化の統合テスト
- **カバレッジ**: バギング、スタッキング、複数モデル組み合わせ

### 3. ドキュメント

#### `backend/docs/ensemble_optimization_config_example.md`

- **内容**: 設定例、パラメータ説明、最適化戦略

#### `backend/docs/ensemble_optimization_implementation_summary.md`

- **内容**: 実装サマリー（このファイル）

## パラメータ命名規則

### ベースモデルパラメータ

- **LightGBM**: `lgb_` プレフィックス（例: `lgb_num_leaves`）
- **XGBoost**: `xgb_` プレフィックス（例: `xgb_max_depth`）
- **RandomForest**: `rf_` プレフィックス（例: `rf_n_estimators`）
- **CatBoost**: `cat_` プレフィックス（例: `cat_iterations`）
- **TabNet**: `tab_` プレフィックス（例: `tab_n_d`）

### アンサンブル固有パラメータ

- **バギング**: `bagging_` プレフィックス（例: `bagging_n_estimators`）
- **スタッキング**: `stacking_` プレフィックス（例: `stacking_meta_C`）

## 対応モデル組み合わせ

### 推奨組み合わせ

1. **バランス型**: `["lightgbm", "xgboost", "catboost"]`
2. **多様性重視**: `["lightgbm", "randomforest", "tabnet"]`
3. **高性能型**: `["lightgbm", "xgboost", "catboost", "tabnet"]`
4. **全モデル**: `["lightgbm", "xgboost", "randomforest", "catboost", "tabnet"]`

## テスト結果

### ユニットテスト

- ✅ `test_ensemble_parameter_space.py`: 11/11 テスト成功
- ✅ `test_ensemble_trainer_optimization.py`: 11/11 テスト成功

### 統合テスト

- ✅ 基本機能テスト成功
- ✅ パラメータ空間統合テスト成功

## 使用方法

### 基本的な使用例

```python
from app.core.services.ml.ml_training_service import MLTrainingService, OptimizationSettings

# アンサンブル設定
ensemble_config = {
    "method": "bagging",
    "models": ["lightgbm", "xgboost", "catboost"],
    "bagging_params": {
        "base_model_type": "mixed"
    }
}

# MLTrainingService初期化
service = MLTrainingService(
    trainer_type="ensemble",
    ensemble_config=ensemble_config
)

# 最適化設定
optimization_settings = OptimizationSettings(
    enabled=True,
    n_calls=50,  # 試行回数
)

# 学習実行
result = service.train_model(
    training_data=training_data,
    optimization_settings=optimization_settings,
    save_model=True
)
```

### カスタムパラメータ空間

```python
custom_parameter_space = {
    "lgb_num_leaves": {"type": "integer", "low": 20, "high": 80},
    "xgb_max_depth": {"type": "integer", "low": 4, "high": 12},
    "cat_iterations": {"type": "integer", "low": 200, "high": 800},
    "bagging_n_estimators": {"type": "integer", "low": 3, "high": 7},
}

optimization_settings = OptimizationSettings(
    enabled=True,
    n_calls=30,
    parameter_space=custom_parameter_space
)
```

## パフォーマンス考慮事項

### 最適化試行回数の目安

- **3 モデル**: 50-80 回
- **4 モデル**: 80-120 回
- **5 モデル**: 100-150 回

### メモリ使用量

- アンサンブル学習は複数モデルを同時に保持するため、メモリ使用量が増加
- 大きなデータセットでは`bagging_max_samples`を調整してメモリ使用量を制御

### 計算時間

- 最適化時間は使用モデル数と試行回数に比例
- 並列処理を活用してパフォーマンスを向上

## 今後の拡張可能性

1. **新しいモデルの追加**: パラメータ空間定義とトレーナー対応で容易に拡張可能
2. **最適化アルゴリズムの追加**: Optuna 以外の最適化手法の統合
3. **動的パラメータ調整**: 学習進行に応じたパラメータ空間の動的調整
4. **マルチ目的最適化**: 精度とモデルサイズの同時最適化

## 注意事項

1. **依存関係**: 各モデルライブラリ（CatBoost、TabNet 等）のインストールが必要
2. **GPU 使用**: TabNet は GPU 使用時にパラメータ調整が必要
3. **データサイズ**: 小さなデータセットでは過学習に注意
4. **最適化時間**: 大規模な最適化は時間がかかるため、段階的に試行回数を増やすことを推奨
