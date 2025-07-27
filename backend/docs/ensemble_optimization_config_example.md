# アンサンブル学習用ハイパーパラメータ最適化設定例

アンサンブル学習を使用する際のハイパーパラメータ最適化設定の例を示します。

## バギングアンサンブルの最適化設定例

```json
{
  "optimization_settings": {
    "enabled": true,
    "n_calls": 50,
    "parameter_space": {
      // LightGBMパラメータ
      "lgb_num_leaves": { "type": "integer", "low": 10, "high": 100 },
      "lgb_learning_rate": { "type": "real", "low": 0.01, "high": 0.3 },
      "lgb_feature_fraction": { "type": "real", "low": 0.5, "high": 1.0 },
      "lgb_bagging_fraction": { "type": "real", "low": 0.5, "high": 1.0 },

      // XGBoostパラメータ
      "xgb_max_depth": { "type": "integer", "low": 3, "high": 15 },
      "xgb_learning_rate": { "type": "real", "low": 0.01, "high": 0.3 },
      "xgb_subsample": { "type": "real", "low": 0.5, "high": 1.0 },

      // RandomForestパラメータ
      "rf_n_estimators": { "type": "integer", "low": 50, "high": 300 },
      "rf_max_depth": { "type": "integer", "low": 3, "high": 20 },

      // CatBoostパラメータ
      "cat_iterations": { "type": "integer", "low": 100, "high": 1000 },
      "cat_learning_rate": { "type": "real", "low": 0.01, "high": 0.3 },
      "cat_depth": { "type": "integer", "low": 3, "high": 10 },
      "cat_l2_leaf_reg": { "type": "real", "low": 1.0, "high": 10.0 },

      // TabNetパラメータ
      "tab_n_d": { "type": "integer", "low": 8, "high": 64 },
      "tab_n_a": { "type": "integer", "low": 8, "high": 64 },
      "tab_n_steps": { "type": "integer", "low": 3, "high": 10 },
      "tab_gamma": { "type": "real", "low": 1.0, "high": 2.0 },

      // バギング固有パラメータ
      "bagging_n_estimators": { "type": "integer", "low": 3, "high": 10 },
      "bagging_max_samples": { "type": "real", "low": 0.5, "high": 1.0 }
    }
  },
  "ensemble_config": {
    "method": "bagging",
    "models": ["lightgbm", "xgboost", "randomforest", "catboost", "tabnet"],
    "bagging_params": {
      "base_model_type": "mixed"
    }
  }
}
```

## スタッキングアンサンブルの最適化設定例

```json
{
  "optimization_settings": {
    "enabled": true,
    "n_calls": 100,
    "parameter_space": {
      // ベースモデルパラメータ（上記と同様）
      "lgb_num_leaves": { "type": "integer", "low": 10, "high": 100 },
      "xgb_max_depth": { "type": "integer", "low": 3, "high": 15 },
      "rf_n_estimators": { "type": "integer", "low": 50, "high": 300 },
      "cat_iterations": { "type": "integer", "low": 100, "high": 1000 },
      "tab_n_d": { "type": "integer", "low": 8, "high": 64 },

      // スタッキング固有パラメータ
      "stacking_meta_C": { "type": "real", "low": 0.01, "high": 10.0 },
      "stacking_meta_penalty": {
        "type": "categorical",
        "categories": ["l1", "l2"]
      },
      "stacking_cv_folds": { "type": "integer", "low": 3, "high": 10 }
    }
  },
  "ensemble_config": {
    "method": "stacking",
    "models": ["lightgbm", "xgboost", "randomforest", "catboost", "tabnet"],
    "stacking_params": {
      "meta_model": "logistic_regression"
    }
  }
}
```

## 最適化のポイント

### 1. パラメータ空間の設計

- **ベースモデル別**: 各モデル（LightGBM、XGBoost、RandomForest）のパラメータを個別に最適化
- **アンサンブル固有**: バギングやスタッキングの手法固有パラメータも最適化対象

### 2. 最適化試行回数の調整

- **バギング**: 比較的少ない試行回数（30-50 回）でも効果的
- **スタッキング**: より多くの試行回数（50-100 回）が推奨（メタモデルの最適化も含むため）

### 3. パフォーマンス考慮事項

- **並列処理**: アンサンブル学習は計算コストが高いため、最適化時間を考慮
- **早期停止**: 収束が見られた場合の早期停止機能の活用
- **メモリ管理**: 複数モデルの同時学習によるメモリ使用量の監視

### 4. 評価指標の選択

- **分類問題**: F1 スコア、AUC-ROC、精度のバランス
- **アンサンブル特有**: 個別モデルの多様性も考慮した評価

## 実装上の注意点

1. **パラメータ命名規則**: プレフィックス（lgb*, xgb*, rf*, bagging*, stacking\_）で区別
2. **設定の継承**: 基本設定からアンサンブル固有設定への適切な継承
3. **エラーハンドリング**: 個別モデルの学習失敗時の適切な処理
4. **ログ出力**: 最適化プロセスの詳細な追跡とデバッグ情報

## CatBoost と TabNet の特徴的なパラメータ

### CatBoost パラメータの説明

- **cat_iterations**: ブースティングのイテレーション数（100-1000）
- **cat_depth**: 木の深さ（3-10、CatBoost は浅い木を好む）
- **cat_l2_leaf_reg**: L2 正則化パラメータ（1.0-10.0）
- **cat_border_count**: 数値特徴量の境界数（32-255）
- **cat_bagging_temperature**: バギングの温度パラメータ（0.0-1.0）
- **cat_random_strength**: ランダム性の強度（0.0-10.0）
- **cat_subsample**: サブサンプリング率（0.5-1.0）
- **cat_colsample_bylevel**: レベル別特徴量サンプリング率（0.5-1.0）

### TabNet パラメータの説明

- **tab_n_d**: 決定ステップの次元数（8-64）
- **tab_n_a**: アテンション機構の次元数（8-64）
- **tab_n_steps**: 決定ステップ数（3-10）
- **tab_gamma**: スパース性制御パラメータ（1.0-2.0）
- **tab_lambda_sparse**: スパース正則化の重み（1e-6-1e-3）
- **tab_optimizer_lr**: 学習率（0.005-0.05）
- **tab_scheduler_step_size**: 学習率スケジューラのステップサイズ（10-50）
- **tab_scheduler_gamma**: 学習率減衰率（0.8-0.99）
- **tab_n_independent**: 独立 GLU 層数（1-5）
- **tab_n_shared**: 共有 GLU 層数（1-5）
- **tab_momentum**: モメンタム（0.01-0.4）

## 最適化戦略の推奨事項

### モデル組み合わせの推奨

1. **バランス型**: `["lightgbm", "xgboost", "catboost"]` - 勾配ブースティング系の組み合わせ
2. **多様性重視**: `["lightgbm", "randomforest", "tabnet"]` - 異なるアルゴリズム系統の組み合わせ
3. **高性能型**: `["lightgbm", "xgboost", "catboost", "tabnet"]` - 4 モデルでの高精度追求

### 最適化試行回数の目安

- **3 モデル**: 50-80 回
- **4 モデル**: 80-120 回
- **5 モデル**: 100-150 回

### パフォーマンス最適化

- **TabNet**: GPU 使用時は`tab_optimizer_lr`を低めに設定
- **CatBoost**: カテゴリ特徴量が多い場合は`cat_depth`を浅めに
- **メモリ使用量**: 大きなデータセットでは`bagging_max_samples`を調整
