# MLハイパーパラメータ最適化API

## 概要

MLハイパーパラメータ最適化APIは、機械学習モデルのハイパーパラメータを自動的に最適化する機能を提供します。ベイジアン最適化、グリッドサーチ、ランダムサーチの3つの最適化手法をサポートしています。

## エンドポイント

### POST /api/ml-training/train

MLモデルのトレーニングを実行します。最適化設定が有効な場合、ハイパーパラメータの最適化も同時に実行されます。

#### リクエスト

```json
{
  "symbol": "BTC/USDT:USDT",
  "timeframe": "1h",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "save_model": true,
  "train_test_split": 0.8,
  "random_state": 42,
  "optimization_settings": {
    "enabled": true,
    "method": "bayesian",
    "n_calls": 50,
    "parameter_space": {
      "num_leaves": {
        "type": "integer",
        "low": 10,
        "high": 100
      },
      "learning_rate": {
        "type": "real",
        "low": 0.01,
        "high": 0.3
      },
      "feature_fraction": {
        "type": "real",
        "low": 0.5,
        "high": 1.0
      }
    }
  }
}
```

#### パラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|---|------|------|
| symbol | string | ✓ | 取引ペア |
| timeframe | string | ✓ | 時間足 |
| start_date | string | ✓ | 開始日 (YYYY-MM-DD) |
| end_date | string | ✓ | 終了日 (YYYY-MM-DD) |
| save_model | boolean | ✓ | モデルを保存するか |
| train_test_split | number | ✓ | 訓練データの割合 (0.0-1.0) |
| random_state | integer | ✓ | 乱数シード |
| optimization_settings | object | - | 最適化設定 |

#### optimization_settings

| パラメータ | 型 | 必須 | 説明 |
|-----------|---|------|------|
| enabled | boolean | ✓ | 最適化を有効にするか |
| method | string | ✓ | 最適化手法 ("bayesian", "grid", "random") |
| n_calls | integer | ✓ | 最適化試行回数 |
| parameter_space | object | ✓ | パラメータ空間設定 |

#### parameter_space

各パラメータは以下の形式で定義します：

**実数パラメータ:**
```json
{
  "parameter_name": {
    "type": "real",
    "low": 0.01,
    "high": 1.0
  }
}
```

**整数パラメータ:**
```json
{
  "parameter_name": {
    "type": "integer",
    "low": 10,
    "high": 100
  }
}
```

**カテゴリカルパラメータ:**
```json
{
  "parameter_name": {
    "type": "categorical",
    "categories": ["option1", "option2", "option3"]
  }
}
```

#### レスポンス

```json
{
  "accuracy": 0.85,
  "f1_score": 0.82,
  "precision": 0.83,
  "recall": 0.81,
  "classification_report": {
    "0": {
      "precision": 0.80,
      "recall": 0.78,
      "f1-score": 0.79,
      "support": 1000
    },
    "1": {
      "precision": 0.86,
      "recall": 0.84,
      "f1-score": 0.85,
      "support": 1200
    },
    "macro avg": {
      "precision": 0.83,
      "recall": 0.81,
      "f1-score": 0.82,
      "support": 2200
    }
  },
  "optimization_result": {
    "method": "bayesian",
    "best_params": {
      "num_leaves": 45,
      "learning_rate": 0.12,
      "feature_fraction": 0.8
    },
    "best_score": 0.82,
    "total_evaluations": 50,
    "optimization_time": 125.6,
    "convergence_info": {
      "converged": true,
      "best_iteration": 38
    }
  }
}
```

## 最適化手法

### ベイジアン最適化 (bayesian)

- **特徴**: ガウス過程を用いた効率的な最適化
- **適用場面**: 評価コストが高い場合、少ない試行回数で良い結果を得たい場合
- **推奨試行回数**: 20-100回

### グリッドサーチ (grid)

- **特徴**: パラメータ空間を網羅的に探索
- **適用場面**: パラメータ数が少ない場合、確実に最適解を見つけたい場合
- **推奨試行回数**: パラメータ組み合わせ数に依存

### ランダムサーチ (random)

- **特徴**: パラメータ空間をランダムに探索
- **適用場面**: パラメータ数が多い場合、ベースライン性能を素早く確認したい場合
- **推奨試行回数**: 50-200回

## エラーレスポンス

### 400 Bad Request

```json
{
  "detail": "無効な最適化手法が指定されました: invalid_method"
}
```

### 422 Unprocessable Entity

```json
{
  "detail": [
    {
      "loc": ["body", "optimization_settings", "parameter_space", "num_leaves", "low"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt",
      "ctx": {"limit_value": 0}
    }
  ]
}
```

### 500 Internal Server Error

```json
{
  "detail": "最適化処理中にエラーが発生しました"
}
```

## 使用例

### 基本的な使用例

```python
import requests

# 基本的なトレーニング（最適化なし）
response = requests.post("/api/ml-training/train", json={
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "save_model": True,
    "train_test_split": 0.8,
    "random_state": 42
})
```

### ベイジアン最適化を使用した例

```python
# ベイジアン最適化付きトレーニング
response = requests.post("/api/ml-training/train", json={
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "save_model": True,
    "train_test_split": 0.8,
    "random_state": 42,
    "optimization_settings": {
        "enabled": True,
        "method": "bayesian",
        "n_calls": 30,
        "parameter_space": {
            "num_leaves": {"type": "integer", "low": 20, "high": 80},
            "learning_rate": {"type": "real", "low": 0.05, "high": 0.2},
            "feature_fraction": {"type": "real", "low": 0.7, "high": 1.0}
        }
    }
})
```

### グリッドサーチを使用した例

```python
# グリッドサーチ最適化
response = requests.post("/api/ml-training/train", json={
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h", 
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "save_model": True,
    "train_test_split": 0.8,
    "random_state": 42,
    "optimization_settings": {
        "enabled": True,
        "method": "grid",
        "n_calls": 50,
        "parameter_space": {
            "num_leaves": {"type": "integer", "low": 30, "high": 50},
            "learning_rate": {"type": "real", "low": 0.1, "high": 0.15}
        }
    }
})
```

## ベストプラクティス

1. **最適化手法の選択**
   - 初回実行時はランダムサーチでベースライン確認
   - 精度重視の場合はベイジアン最適化
   - 確実性重視の場合はグリッドサーチ

2. **パラメータ空間の設定**
   - 適切な範囲設定（狭すぎず広すぎず）
   - 重要なパラメータを優先的に含める
   - カテゴリカルパラメータは選択肢を絞る

3. **試行回数の設定**
   - ベイジアン最適化: 20-50回
   - グリッドサーチ: 組み合わせ数に応じて
   - ランダムサーチ: 50-100回

4. **パフォーマンス考慮**
   - 大きなデータセットでは試行回数を制限
   - 並列実行は避ける（リソース競合防止）
   - 定期的な進捗確認
