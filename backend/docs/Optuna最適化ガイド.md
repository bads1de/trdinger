# Optuna HPO（ハイパーパラメータ最適化）ガイド

## 1. 概要

このシステムでは[Optuna](https://optuna.readthedocs.io/)を使用した効率的なハイパーパラメータ最適化（HPO）を実装しています。

本ガイドでは、以下の2つの主要な機能について説明します。

1.  **トライアル数調整**: 精度と計算時間のバランスを取るために、最適化の試行回数を柔軟に設定します。
2.  **プルーニング（Pruning）**: 見込みのないハイパーパラメータ試行を早期終了し、計算リソースを有望な試行に集中させます。

これらの機能により、限られたリソースでも効率的に高性能なモデルを発見することが可能になります。

---

## 2. アーキテクチャと動作原理

### 2.1 コンポーネント構成

HPOは以下のコンポーネントによって実現されています。

```
backend/app/
├── services/
│   ├── ml/
│   │   └── ml_training_service.py # MLTrainingService
│   └── optimization/
│       └── optuna_optimizer.py     # OptunaOptimizer
```

-   **`MLTrainingService`**: HPOのワークフロー全体を管理します。`train_model`メソッドが`OptimizationSettings`を受け取ると、最適化プロセスを開始します。
-   **`OptunaOptimizer`**: Optunaの`study`オブジェクトを管理し、`optimize`メソッドで実際の最適化ループを実行します。
-   **`Pruner` (e.g., `MedianPruner`)**: `OptunaOptimizer`内部で使用され、各トライアルの中間報告を監視し、成績の悪いトライアルを早期に打ち切ります（プルーニング）。

### 2.2 最適化プロセスフロー

```
MLTrainingService.train_model(optimization_settings=...)
  ↓
_train_with_optimization()
  │ 1. `OptunaOptimizer` を作成
  │ 2. `_create_objective_function` で目的関数を定義
  │ 3. `optimizer.optimize()` を呼び出し
  ↓
OptunaOptimizer.optimize(n_calls=...)
  │ 1. `optuna.create_study` で探索セッションを開始
  │    └ Prunerとして `MedianPruner` を設定
  │ 2. `study.optimize()` で目的関数をn_calls回実行
  │    └ 各トライアルで目的関数が中間スコアを報告
  │    └ `MedianPruner` が中間スコアを評価し、見込みがなければトライアルを早期終了
  ↓
最適化完了後、ベストパラメータを返す
  ↓
MLTrainingServiceがベストパラメータで最終モデルを再学習
```

### 2.3 プルーニングの動作原理

プルーニングは、HPOの計算コストを大幅に削減する鍵となります。

**▼ Pruningなし（従来）**
すべてのトライアルを最後まで実行するため、結果の悪い試行にも多くの時間を費やします。
```
Trial 1: [========================================] 100% 完了（悪い結果）
Trial 2: [========================================] 100% 完了（悪い結果）
Trial 3: [========================================] 100% 完了（良い結果）
```

**▼ Pruning有効時**
中間評価値が他の試行の中央値より悪い場合、その試行は早期に打ち切られます。
```
Trial 1: [=====> PRUNED] 10% で早期終了（見込みなし）
Trial 2: [====> PRUNED] 8% で早期終了（見込みなし）
Trial 3: [========================================] 100% 完了（良い結果）
Trial 4: [========================================] 100% 完了（良い結果）
```
これにより、計算リソースが有望なハイパーパラメータの探索に集中され、効率が向上します。

---

## 3. 設定方法

### 3.1 HPOの有効化とトライアル数

HPOは`MLTrainingService.train_model`メソッドの`optimization_settings`引数で制御します。

```python
from app.services.ml.ml_training_service import MLTrainingService, OptimizationSettings

# MLTrainingServiceインスタンス作成
service = MLTrainingService()

# 最適化設定を作成
opt_settings = OptimizationSettings(
    enabled=True,   # HPOを有効化
    n_calls=100,    # トライアル数を100回に設定
    parameter_space={} # パラメータ空間（省略するとデフォルト）
)

# MLトレーニング実行
result = service.train_model(
    training_data=training_data,
    optimization_settings=opt_settings,
    save_model=True
)
```

| パラメータ | 型 | 説明 |
| :--- | :--- | :--- |
| `enabled` | `bool` | `True`に設定するとHPOが実行されます。 |
| `n_calls` | `int` | Optunaが試行するハイパーパラメータの組み合わせの数。 |

#### パフォーマンスへの影響

トライアル数と精度のトレードオフを考慮して設定します。

| トライアル数 | 所要時間目安 | 精度向上 | 推奨用途 |
| :--- | :--- | :--- | :--- |
| 10-30 | 短い | 低 | 動作確認、プロトタイピング |
| **50-100** | **中〜長い** | **中〜高** | **通常開発、本番前調整** |
| 200+ | 非常に長い | 最高 | 研究目的 |

### 3.2 プルーニングの設定

プルーニングは`OptunaOptimizer`内で自動的に有効化されますが、その動作は環境変数で調整可能です。

**環境変数**
```bash
# Pruningを有効化（デフォルト）
ML__OPTIMIZATION__ENABLE_PRUNING=true

# プルーニング開始前のウォームアップステップ数（デフォルト: 10）
# 初期N回のイテレーションではプルーニングを実行しない
ML__OPTIMIZATION__PRUNING_WARMUP_STEPS=10
```

これらの設定は、`ML改善実装設計.md`で示された`enable_pruning`や`pruning_warmup_steps`のようなPythonコード内の設定と連動します（現在は環境変数経由での設定が主）。

---

## 4. ベストプラクティス

### 4.1 開発フェーズ別の使い分け

-   **開発初期（プロトタイピング）**: `n_calls=30`程度で素早く反復。
-   **開発中期（機能実装）**: `n_calls=50`で効率と精度のバランスを取る。
-   **開発後期（品質検証）**: `n_calls=100`以上で精度を重視。

### 4.2 プルーニングのウォームアップ設定

-   `pruning_warmup_steps`は`10`〜`15`程度を推奨。
-   これにより、プルーニングの判断基準となる中央値を適切に計算できます。

### 4.3 LightGBMPruningCallbackの活用

`OptunaOptimizer`の内部では、`LightGBMPruningCallback`が自動的に適用されるように目的関数が構築されます。これにより、LightGBMの学習プロセスとOptunaが密に連携し、効率的なプルーニングが実現されます。

**▼ コールバック連携のポイント**
```python
# 目的関数内でtrialオブジェクトをモデル学習に渡すことで連携が確立される
model.fit(
    ...,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_logloss", # プルーニング対象のメトリック
    callbacks=[pruning_callback],
)
```

### 4.4 リソース管理

-   **並列実行**: `study.optimize(n_jobs=4)`のように設定すると複数プロセスで並列最適化できますが、メモリ使用量が増加するため注意が必要です。
-   **メモリ監視**: 大量のトライアルを実行する際は、メモリ使用率を監視し、必要であればデータサイズを削減する（サンプリングするなど）ことを検討してください。

---

## 5. トラブルシューティング

### 問題: 最適化が途中で停止する

-   **原因**: メモリ不足、またはプルーニング設定が厳しすぎる。
-   **対策**:
    1.  システムメモリを監視し、必要であれば学習データの一部をサンプリングして使用する。
    2.  プルーニングの`n_startup_trials`（初期にプルーニングしない試行数）や`n_warmup_steps`（評価ステップ数）を増やし、より寛容な設定にする。

### 問題: プルーニングが全く動作しない

-   **原因1**: `enable_pruning`設定が無効になっている。環境変数を確認してください。
-   **原因2**: 目的関数から`trial`オブジェクトがモデルの学習コールバックに正しく渡されていない。
-   **原因3**: `n_warmup_steps`が長すぎて、プルーニングが開始される前にトライアルが終了している。

### 問題: 精度が期待ほど向上しない

-   **原因**: パラメータ空間が適切でない、またはデータの質に問題がある。
-   **対策**:
    1.  `OptunaOptimizer`内の`get_default_parameter_space`などで定義されているパラメータの探索範囲を見直す。
    2.  特徴量エンジニアリングやクラス不均衡対策など、データの質を改善する他の施策を検討する。

---

## 6. まとめ

-   HPOは`MLTrainingService`を通じて、`OptimizationSettings`で制御します。
-   **トライアル数 (`n_calls`)** は、計算時間と精度のトレードオフを決定する主要な要素です。
-   **プルーニング**は、見込みのない試行を早期に打ち切ることでHPOを大幅に高速化する重要な機能であり、デフォルトで有効化されています。

このガイドを参考に、プロジェクトの要件とリソースに応じて最適なハイパーパラメータ最適化を実現してください。
