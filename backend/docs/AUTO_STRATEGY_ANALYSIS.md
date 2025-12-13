# オートストラテジーシステム分析と改善提案

このドキュメントは、`auto_strategy` モジュールの現状を分析し、システムアップグレードのための課題と改善案を提示します。

---

## 1. システム概要

### 1.1 アーキテクチャ全体像

```
┌───────────────────────────────────────────────────────────────────────┐
│                         GeneticAlgorithmEngine                        │
│                       (app/services/auto_strategy/core/ga_engine.py)  │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐     ┌──────────────────────┐    ┌─────────────┐ │
│  │ RandomGene      │     │  IndividualEvaluator │    │ Genetic     │ │
│  │ Generator       │────▶│   または              │────│ Operators   │ │
│  │                 │     │ HybridIndividual     │    │ (crossover, │ │
│  │                 │     │ Evaluator            │    │  mutation)  │ │
│  └─────────────────┘     └──────────────────────┘    └─────────────┘ │
│          │                         │                                  │
│          ▼                         ▼                                  │
│  ┌─────────────────┐     ┌──────────────────────┐                    │
│  │ StrategyGene    │     │  BacktestService     │                    │
│  │ (遺伝子オブジェ  │     │  (バックテスト実行)   │                    │
│  │ クト)            │     └──────────────────────┘                    │
│  └─────────────────┘               │                                  │
│                                    ▼                                  │
│                          ┌──────────────────────┐                    │
│                          │ UniversalStrategy    │                    │
│                          │ (backtesting.py互換)  │                    │
│                          └──────────────────────┘                    │
└───────────────────────────────────────────────────────────────────────┘
```

### 1.2 主要コンポーネント

| コンポーネント                | ファイルパス                          | 責務                                           |
| ----------------------------- | ------------------------------------- | ---------------------------------------------- |
| **GeneticAlgorithmEngine**    | `core/ga_engine.py`                   | DEAP 使用した GA 実行の統括                    |
| **RandomGeneGenerator**       | `generators/random_gene_generator.py` | ランダム遺伝子生成                             |
| **IndividualEvaluator**       | `core/individual_evaluator.py`        | 個体評価（バックテスト実行＋フィットネス計算） |
| **HybridIndividualEvaluator** | `core/hybrid_individual_evaluator.py` | ML 予測スコア統合版評価器                      |
| **UniversalStrategy**         | `strategies/universal_strategy.py`    | backtesting.py 互換の戦略クラス                |
| **HybridPredictor**           | `core/hybrid_predictor.py`            | ML 予測器（アンサンブル対応）                  |
| **HybridFeatureAdapter**      | `utils/hybrid_feature_adapter.py`     | StrategyGene→ML 特徴量変換                     |

---

## 2. 現在の GA 設定パラメータ

### 2.1 探索規模パラメータ（`config/ga.py`）

```python
GA_DEFAULT_CONFIG = {
    "population_size": 100,    # 1世代の個体数
    "generations": 50,         # 進化世代数
    "crossover_rate": 0.8,     # 交叉確率
    "mutation_rate": 0.1,      # 突然変異確率
    "elite_size": 10,          # エリート保存数
    "max_indicators": 3,       # 使用可能インジケータ最大数
}
```

### 2.2 フィットネス評価パラメータ

```python
FITNESS_WEIGHT_PROFILES = {
    "balanced": {
        "total_return": 0.2,           # リターン重視
        "sharpe_ratio": 0.25,          # リスク調整済みリターン
        "max_drawdown": 0.15,          # 最大ドローダウンペナルティ
        "win_rate": 0.1,               # 勝率
        "balance_score": 0.1,          # ロング/ショートバランス
        "ulcer_index_penalty": 0.15,   # ストレス指標ペナルティ
        "trade_frequency_penalty": 0.05, # 取引頻度ペナルティ
    },
}
```

### 2.3 高度な設定（`config/ga_runtime.py`の`GAConfig`）

| 設定項目                 | デフォルト値 | 説明                           |
| ------------------------ | ------------ | ------------------------------ |
| `enable_walk_forward`    | `False`      | Walk-Forward Analysis 有効化   |
| `wfa_n_folds`            | `5`          | WFA のフォールド数             |
| `wfa_train_ratio`        | `0.7`        | 各フォールドの学習期間比率     |
| `oos_split_ratio`        | `0.0`        | Out-of-Sample 分割比率         |
| `enable_multi_timeframe` | `False`      | マルチタイムフレーム有効化     |
| `hybrid_mode`            | `False`      | GA+ML ハイブリッドモード       |
| `enable_fitness_sharing` | `True`       | フィットネス共有（多様性維持） |

---

## 3. 課題分析

### 3.1 ✅ 課題 1: ML フィルターが「フィルター」として機能していない【修正済み】

**修正日:** 2024-12-13

**対応内容:**
`UniversalStrategy` に ML フィルター機能を実装し、エントリー条件成立時にリアルタイムで ML 予測を確認し、危険な相場でのエントリーを拒否できるようにしました。

**実装:**

1. `UniversalStrategy.__init__()` に `ml_predictor` と `ml_filter_threshold` パラメータを追加
2. `_ml_allows_entry(direction)` メソッドを追加: ML 予測がエントリー方向と一致するかを判定
3. `_prepare_current_features()` メソッドを追加: 現在のバーから ML 用特徴量を準備
4. `next()` メソッドで ML フィルター判定を追加: エントリー条件成立後、ML で許可/拒否を判定

```python
# UniversalStrategy.next() 改善後
def next(self):
    # ... 既存のエントリー条件チェック ...

    if long_signal or short_signal or stateful_direction is not None:
        direction = 1.0 if long_signal else (-1.0 if short_signal else stateful_direction)

        # === ML フィルター判定 ===
        if direction != 0.0 and self.ml_predictor is not None:
            if not self._ml_allows_entry(direction):
                return  # MLがエントリーを拒否

        # 通常のエントリー処理
        self.buy(size=position_size) if direction > 0 else self.sell(size=position_size)

def _ml_allows_entry(self, direction: float) -> bool:
    """MLがエントリーを許可するかチェック"""
    if self.ml_predictor is None:
        return True

    features = self._prepare_current_features()
    prediction = self.ml_predictor.predict(features)

    up_score = prediction.get("up", 0.33)
    down_score = prediction.get("down", 0.33)

    if direction > 0:  # Long
        return up_score > down_score + self.ml_filter_threshold
    else:  # Short
        return down_score > up_score + self.ml_filter_threshold
```

**効果:**

- GA は「ML が OK を出した相場でのみ勝てる戦略」を探す
- 役割分担が明確化（GA=テクニカル構造、ML=相場環境判断）
- 「ML が危険と判断した場面でのエントリー」を防止

### 3.2 ✅ 課題 2: Optuna は ML モデル最適化用に限定されている【修正済み】

**修正日:** 2024-12-13

**対応内容:**
`StrategyParameterTuner` と `StrategyParameterSpace` を実装し、GA で発見された戦略構造のパラメータを Optuna で最適化できるようにしました。

**実装:**

1. `StrategyParameterSpace`: StrategyGene から Optuna パラメータ空間を動的に構築
2. `StrategyParameterTuner`: Optuna を使用した戦略パラメータ最適化
3. `GAConfig` にチューニング設定を追加（`enable_parameter_tuning` など）
4. `GeneticAlgorithmEngine._tune_elite_parameters()` メソッドを追加

```python
# GAConfig でチューニングを有効化
config = GAConfig(
    enable_parameter_tuning=True,  # チューニング有効化
    tuning_n_trials=30,            # Optuna試行回数
    tuning_use_wfa=True,           # WFA評価を使用
)
```

**効果:**

- GA は「大まかに良さそうな構造」の発見に集中
- Optuna は「構造内の最適パラメータ」を高精度に特定
- WFA 評価との連携で過学習を抑制

### 3.3 🟡 課題 3: 過学習対策機能は実装済みだが、デフォルト無効

**現状:**
以下の過学習対策機能が`IndividualEvaluator`に実装されていますが、**デフォルトでは無効**です。

| 機能                  | 設定項目                 | デフォルト      |
| --------------------- | ------------------------ | --------------- |
| Out-of-Sample 検証    | `oos_split_ratio`        | `0.0`（無効）   |
| Walk-Forward Analysis | `enable_walk_forward`    | `False`（無効） |
| フィットネス共有      | `enable_fitness_sharing` | `True`（有効）  |

**問題点:**

- デフォルト設定では、過去データ全体へのカーブフィッティングが発生しやすい
- WFA を有効化しても、「WFA スコアを最大化する Optuna」との連携がない

---

## 4. ML 統合の現状詳細

### 4.1 HybridPredictor の役割

`HybridPredictor`は`MLTrainingService`をラップし、GA 評価時に ML 予測を提供します。

```python
# HybridPredictor.predict より
def predict(self, features_df: pd.DataFrame) -> Dict[str, float]:
    # 複数モデルの場合は平均化
    if len(self.services) > 1:
        predictions = [service.generate_signals(features_df) for service in self.services]
        ml_prediction = {
            "up": np.mean([p.get("up", 0.0) for p in predictions]),
            "down": np.mean([p.get("down", 0.0) for p in predictions]),
            "range": np.mean([p.get("range", 0.0) for p in predictions]),
        }
    else:
        ml_prediction = self.services[0].generate_signals(features_df)

    return self._normalise_prediction(ml_prediction)
```

**出力形式:**

- **方向予測:** `{"up": 0.4, "down": 0.3, "range": 0.3}`
- **ボラティリティ予測:** `{"trend": 0.6, "range": 0.4}`

### 4.2 HybridFeatureAdapter の役割

`StrategyGene` → `ML特徴量DataFrame` への変換を担当。

**抽出される特徴量:**

```python
# 戦略構造特徴
"indicator_count": 3
"condition_count": 5
"has_tpsl": 1
"take_profit_ratio": 0.02
"stop_loss_ratio": 0.01

# OHLCVからの派生特徴
"close_return_1", "close_return_5"
"close_rolling_mean_5", "close_rolling_std_5"
"oi_pct_change", "funding_rate_change"

# ウェーブレット特徴（オプション）
"wavelet_close_scale_2", "wavelet_close_scale_4"
```

### 4.3 現在の ML フィルター処理フロー（問題あり）

```
1. GAが戦略遺伝子（StrategyGene）を生成
         │
         ▼
2. HybridIndividualEvaluator.evaluate_individual()
         │
         ├─▶ バックテスト実行（UniversalStrategy.next()でエントリー判断）
         │       └─ この時点ではML予測は使われていない！
         │
         ▼
3. ML予測スコアを取得（HybridPredictor.predict()）
         │
         ▼
4. フィットネス計算（バックテスト結果 + ML予測スコア）
         │
         ▼
5. フィットネスをGAに返却
```

**問題:**

- ステップ 2 で ML が介入していない
- ステップ 4 で「加点」しても、すでに損失が発生している

---

## 5. 改善提案

### 5.1 提案 1: ML フィルターの「真のフィルター化」（優先度: 高）

**目的:** ML が「危険」と判断した相場でエントリーを拒否できるようにする。

**実装:**

1. `UniversalStrategy`の`__init__`で`HybridPredictor`をオプション受け取り
2. `next()`メソッド内で条件成立時に ML 予測を取得
3. ML 予測スコアが閾値以下ならエントリーをスキップ

```python
# UniversalStrategy.next() 改善案
def next(self):
    # ... 既存のエントリー条件チェック ...

    if long_signal or short_signal:
        # MLフィルターによる拒否判定
        if self.ml_predictor and not self._ml_allows_entry(direction):
            logger.debug(f"ML Filter: エントリー拒否 (direction={direction})")
            return  # エントリーしない

        # 通常のエントリー処理
        self.buy(size=position_size) if direction > 0 else self.sell(size=position_size)

def _ml_allows_entry(self, direction: float) -> bool:
    """MLがエントリーを許可するかチェック"""
    features = self._prepare_current_features()
    prediction = self.ml_predictor.predict(features)

    # 方向予測の場合
    if direction > 0:  # Long
        return prediction.get("up", 0) > prediction.get("down", 0) + 0.1
    else:  # Short
        return prediction.get("down", 0) > prediction.get("up", 0) + 0.1
```

**効果:**

- GA は「ML が OK を出した相場でのみ勝てる戦略」を探す
- 役割分担が明確化（GA=テクニカル構造、ML=相場環境判断）

### 5.2 ✅ 提案 2: GA×Optuna ハイブリッド化（優先度: 中）【実装完了】

**実装完了日:** 2024-12-13

**目的:** GA で発見した戦略構造に対し、Optuna でパラメータチューニングを行う。

**実装:**

1. GA のフィットネス評価時に、上位 N 個体に対して Optuna 最適化を実施
2. 最適化の評価関数を WFA スコアに設定（過学習防止）

```python
# 疑似コード: Optunaによる戦略パラメータ最適化
def optimize_strategy_parameters(gene: StrategyGene, wfa_config: GAConfig) -> StrategyGene:
    def objective(trial: optuna.Trial) -> float:
        # インジケータパラメータを提案
        for indicator in gene.indicators:
            if indicator.type == "RSI":
                indicator.parameters["period"] = trial.suggest_int("rsi_period", 5, 50)

        # WFAスコアで評価
        wfa_fitness = individual_evaluator._evaluate_with_walk_forward(
            gene, backtest_config, wfa_config
        )
        return wfa_fitness[0]  # weighted_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    # 最適パラメータを適用
    apply_params(gene, study.best_params)
    return gene
```

**効果:**

- GA は「大まかに良さそうな構造」を発見することに集中
- Optuna は「構造内の最適パラメータ」を高精度に特定
- WFA 評価により過学習を抑制

### 5.3 提案 3: デフォルト設定で WFA/OOS を有効化（優先度: 中）

**目的:** 過学習を防ぐため、デフォルトで OOS 検証を有効にする。

**実装:**
`config/ga_runtime.py`のデフォルト値を変更：

```python
# 変更前
oos_split_ratio: float = 0.0
enable_walk_forward: bool = False

# 変更後
oos_split_ratio: float = 0.2  # 20%をOOSに
enable_walk_forward: bool = True
wfa_n_folds: int = 3  # 3フォールド（計算コストと精度のバランス）
```

**効果:**

- 新規ユーザーでも過学習しにくい設定で開始できる
- 「全期間でたまたまフィットした戦略」の淘汰

---

## 6. 追加アイデア: 市場レジーム検出との連携

### 6.1 コンセプト

「全ての相場で勝てる単一戦略」を目指すのではなく、**相場環境（レジーム）ごとに最適な戦略を使い分ける**アプローチ。

### 6.2 実装案

1. **レジーム分類器の作成:**

   - 既存の ML 機能を活用（`label_generation/`配下のモジュール）
   - `trend_scanning.py`や`event_driven.py`でレジームラベルを生成

2. **レジーム別 GA の実行:**

   ```python
   regimes = ["TREND_UP", "TREND_DOWN", "RANGE"]
   strategies = {}

   for regime in regimes:
       filtered_data = filter_by_regime(ohlcv_data, regime)
       ga_result = ga_engine.run_evolution(config, {"data": filtered_data})
       strategies[regime] = ga_result["best_strategy"]
   ```

3. **メタ戦略（戦略の戦略）:**
   - リアルタイムでレジームを判定
   - 適切な戦略を動的に切り替え

---

## 7. まとめと推奨アクション

| 優先度     | アクション                      | 期待効果                                          | 実装コスト | 状態      |
| ---------- | ------------------------------- | ------------------------------------------------- | ---------- | --------- |
| ~~**高**~~ | ML フィルターの真のフィルター化 | 無駄なエントリーの排除、GA と ML の役割分担明確化 | 中         | ✅ 完了   |
| **中**     | WFA/OOS のデフォルト有効化      | 過学習防止、初期設定の改善                        | 低         | 📋 未着手 |
| **中**     | GA×Optuna ハイブリッド化        | パラメータ探索効率向上、WFA 連携                  | 高         | ✅ 完了   |
| **低**     | 市場レジーム連携                | 相場適応型戦略ポートフォリオ                      | 高         | 📋 未着手 |

---

## 8. 関連ファイル一覧

```
backend/app/services/auto_strategy/
├── config/
│   ├── ga.py                    # GA基本設定・定数
│   ├── ga_runtime.py            # GAConfig（実行時設定）
│   └── tpsl.py                  # TPSL設定
├── core/
│   ├── ga_engine.py             # GAエンジン本体
│   ├── individual_evaluator.py  # 個体評価器（OOS/WFA対応）
│   ├── hybrid_individual_evaluator.py  # ハイブリッド評価器
│   ├── hybrid_predictor.py      # ML予測器
│   ├── genetic_operators.py     # 交叉・突然変異
│   └── evolution_runner.py      # 進化実行
├── generators/
│   ├── random_gene_generator.py # ランダム遺伝子生成
│   ├── condition_generator.py   # 条件生成
│   └── strategy_factory.py      # 戦略ファクトリー
├── strategies/
│   └── universal_strategy.py    # backtesting.py互換戦略
├── utils/
│   └── hybrid_feature_adapter.py # Gene→特徴量変換
└── serializers/
    └── gene_serialization.py    # 遺伝子シリアライゼーション

backend/app/services/ml/
├── optimization/
│   ├── optuna_optimizer.py      # Optuna最適化エンジン
│   └── optimization_service.py  # 最適化サービス
├── label_generation/
│   ├── trend_scanning.py        # トレンドスキャン
│   └── event_driven.py          # イベント駆動ラベル
└── ml_training_service.py       # ML学習サービス
```

---

_ドキュメント作成日: 2024-12-12_
_作成者: Antigravity AI Assistant_
