# auto_strategy リファクタリング計画

## 目的

`backend/app/services/auto_strategy` 配下の責務分割を整理し、以下を改善する。

- 変更の影響範囲が広すぎる箇所の分解
- 実行時依存と互換レイヤの整理
- 未使用または冗長な API surface の削減
- テストしやすい境界への再設計

このドキュメントは 2026-03-27 時点の静的レビューに基づく。

## 進捗

2026-03-27 時点の実装反映状況:

- [x] `UniversalStrategy` に `StrategyRuntimeState` を導入し、`OrderManager` / `PositionManager` の state 更新を集約
- [x] `UniversalStrategy` の entry / exit 判定を `EntryDecisionEngine` / `PositionExitEngine` に分離
- [x] `StrategyGene` から factory / mutate / crossover ロジックを分離
- [x] `IndividualEvaluator` から `BacktestDataProvider` と `RunConfigBuilder` を分離
- [x] `ExperimentManager` の engine registry を `ExperimentEngineRegistry` として分離
- [x] `ga_engine` から evaluator の private API 依存を除去
- [x] serializer の内部 helper 分割
- [x] 未使用 config / 冗長 export の整理 (`utils.__init__` の eager export 整理、`config.settings` と未使用 settings class を削除)

## 結論

現状の `auto_strategy` は機能追加は可能だが、アーキテクチャとしては以下の問題が蓄積している。

- ドメインモデルに演算責務が入り込みすぎている
- strategy / evaluator / engine / persistence の境界が曖昧
- helper class を導入しても private state 共有が強く、分離効果が弱い
- 互換 import と再 export が多く、構造改善の阻害要因になっている
- 一部に実質未使用の設定クラスや冗長 API が残っている

優先度としては、まず巨大クラスの責務分離、その次に互換レイヤ削減、その後に serializer と persistence の整理を進めるのが妥当。

## 優先度付きの指摘

### 1. `StrategyGene` が多機能すぎる

対象:

- `backend/app/services/auto_strategy/genes/strategy.py`

現状:

- `StrategyGene` はデータモデルであるにもかかわらず、以下を内包している
- デフォルト生成
- assemble
- validate
- clone
- mutate
- adaptive_mutate
- crossover

問題:

- データ構造の変更が GA 演算全体の変更に直結する
- model と operator の責務が分離されていない
- テストの単位が大きくなり、演算ごとの差分検証がしづらい

方針:

- `StrategyGene` は状態保持に寄せる
- `operators/strategy_mutation.py`
- `operators/strategy_crossover.py`
- `factories/strategy_factory.py`
- `validators/strategy_validator.py`

期待効果:

- 演算ロジックを独立してテストできる
- ドメインモデルの変更時に GA 演算の巻き込みを減らせる

### 2. `UniversalStrategy` が god object 化している

対象:

- `backend/app/services/auto_strategy/strategies/universal_strategy.py`
- `backend/app/services/auto_strategy/strategies/order_manager.py`
- `backend/app/services/auto_strategy/strategies/position_manager.py`

現状:

- `UniversalStrategy` が indicator init、MTF、ML filter、tool gating、entry 判定、pending order、TP/SL、position sizing を保持
- `OrderManager` と `PositionManager` が `strategy._sl_price` などの private state を直接更新

問題:

- 分離されたように見えて実態は shared mutable state
- 戦略実行の局所的変更でも複数クラスを横断する
- state の整合性を型や interface で担保できていない

方針:

- `StrategyRuntimeState` を導入し、価格・方向・TP/SL・pending orders を集約する
- `UniversalStrategy` は orchestration のみに寄せる
- `OrderManager` / `PositionManager` は `StrategyRuntimeState` を操作する
- indicator 初期化は `IndicatorBootstrapper`
- ML 判定は `EntryFilterPipeline`
- entry 実行は `EntryDecisionEngine`

期待効果:

- strategy helper が private field に直接触れなくなる
- state mutation の追跡が容易になる

### 3. `IndividualEvaluator` と `GeneticAlgorithmEngine` の境界が崩れている

対象:

- `backend/app/services/auto_strategy/core/evaluation/individual_evaluator.py`
- `backend/app/services/auto_strategy/core/engine/ga_engine.py`

現状:

- `IndividualEvaluator` が設定保持、キャッシュ、データ取得、run config 構築、バックテスト、fitness 計算まで担当
- `ga_engine` が evaluator の private field や private method に依存している

問題:

- evaluator が差し替えにくい
- engine 側が内部実装に依存しているため、抽象化が成立していない
- 並列評価や hybrid evaluator の拡張点が汚染されやすい

方針:

- `BacktestDataProvider`
- `EvaluationCache`
- `RunConfigBuilder`
- `FitnessService`
- `EvaluationPipeline`

上記に分割し、`IndividualEvaluator` は facade にする。

また、`GeneticAlgorithmEngine` から evaluator の private API 呼び出しをなくし、公開 interface のみ使う。

期待効果:

- 並列評価、walk-forward、OOS の実装が差し替えやすくなる
- hybrid evaluator を同一 interface で扱いやすくなる

### 4. 実験管理が API / runtime / persistence をまたいでいる

対象:

- `backend/app/services/auto_strategy/services/auto_strategy_service.py`
- `backend/app/services/auto_strategy/services/experiment_manager.py`
- `backend/app/services/auto_strategy/services/experiment_persistence_service.py`

現状:

- `AutoStrategyService` が FastAPI の `BackgroundTasks` を知っている
- `ExperimentManager` が class-level registry を持つ
- persistence service が保存 payload 整形まで持つ

問題:

- framework 依存と domain workflow が混ざっている
- in-memory registry の扱いが暗黙的
- 将来的に worker 化や queue 化しづらい

方針:

- `AutoStrategyService` は API adapter に寄せる
- `ExperimentApplicationService` を作り、開始・停止・状態遷移を集中管理する
- `ActiveExperimentRegistry` を独立サービス化する
- `ExperimentPersistenceService` は repository 呼び出しに専念させる

期待効果:

- FastAPI 背景タスクから job queue に移行しやすくなる
- 実行中エンジンの管理責務が明確になる

### 5. serializer 層が肥大化している

対象:

- `backend/app/services/auto_strategy/serializers/serialization.py`

現状:

- `DictConverter` と `GeneSerializer` が strategy、condition、stateful condition、sub-gene、JSON 変換を一括処理

問題:

- 変更点が 1 ファイルに集中する
- 一部の責務は persistence 用、別の責務は transport 用で混ざっている

方針:

- `strategy_gene_serializer.py`
- `condition_serializer.py`
- `stateful_condition_serializer.py`
- `gene_json_codec.py`

公開 facade が必要なら `GeneSerializer` は残して中で委譲する。

期待効果:

- 保存形式変更や API 形式変更の影響を局所化できる

### 6. 互換 shim と再 export が多い

対象:

- `backend/app/services/auto_strategy/core/__init__.py`
- `backend/app/services/auto_strategy/config/__init__.py`
- `backend/app/services/auto_strategy/config/settings.py`
- `backend/app/services/auto_strategy/utils/__init__.py`

現状:

- `core.__init__` が旧 path を `sys.modules` に差し込んでいる
- config と utils に再 export レイヤが複数ある

問題:

- 現在の構造が実際には改善されていても、旧 path が残り続ける
- import surface が増え、依存追跡が難しい

方針:

- app 本体コードでは新 path に統一する
- テストも段階的に新 path へ移行する
- 互換 shim は deprecation 期間を決めて削除する
- `settings.py` のような re-export 専用モジュールは用途がなければ削除する

期待効果:

- package 構造と import 構造が一致する
- リファクタ時の見通しが良くなる

### 7. fitness sharing が 1 クラスに寄りすぎている

対象:

- `backend/app/services/auto_strategy/core/fitness/fitness_sharing.py`

現状:

- 特徴ベクトル化
- 類似度計算
- KD-tree 近傍探索
- silhouette clustering
- individual へのキャッシュ書き込み

問題:

- アルゴリズム変更の影響範囲が広い
- vectorizer と sharing policy の独立検証がしづらい

方針:

- `StrategyVectorizer`
- `SimilarityMetric`
- `SharingPolicy`
- `FitnessSharingService`

のように分ける。

期待効果:

- diversity policy の比較実験がしやすくなる

## 不要または冗長な候補

### 1. 未使用に見える設定クラス群

候補:

- `TradingSettings`
- `IndicatorSettings`
- `TPSLSettings`
- `PositionSizingSettings`
- `config/settings.py`

理由:

- repo 内検索では app 本体からの利用が見当たらない
- `ConfigValidator` 側にも将来利用前提のコメントだけが残っている

対応:

- 外部利用がなければ削除候補
- すぐ消せない場合は deprecated 扱いにする

### 2. `inject_seeds_into_population`

対象:

- `backend/app/services/auto_strategy/generators/seed_strategy_factory.py`

理由:

- runtime では `ga_engine` 側が独自に seed 注入しており、この関数は主に export と test のために残っている

対応:

- `ga_engine` からこの関数を使う形に寄せるか、逆に helper として deprecated にして削除する
- 二重実装を避ける

### 3. `utils.__init__` の二重 export

対象:

- `backend/app/services/auto_strategy/utils/__init__.py`

理由:

- lazy import と eager import が混在
- `operand_grouping` まで utils から再 export している

対応:

- 役割を `normalization` の export のみに絞る
- それ以外は本来の package から import させる

### 4. 小さな重複

候補:

- `IndicatorCalculator._calculate_indicator_from_dataframe`
- `IndicatorCalculator._calculate_indicator_from_df`
- `ExperimentPersistenceService.complete_experiment`
- `ExperimentPersistenceService.fail_experiment`
- `ExperimentPersistenceService.stop_experiment`
- `save_experiment_result` の未使用 `backtest_config`

対応:

- この種の重複は第一段階の大規模分割後にまとめて掃除する

## 残すべきもの

以下は現時点では削除対象ではない。

- `tools/*`
- `hybrid/*`
- `parallel_evaluator`

理由:

- 実行経路またはテスト経路があり、機能としてまだ生きている
- 問題は存在そのものではなく境界の持ち方

## 推奨ターゲット構成

一例:

```text
auto_strategy/
├── application/
│   ├── experiment_application_service.py
│   └── active_experiment_registry.py
├── domain/
│   ├── genes/
│   ├── operators/
│   ├── validators/
│   └── value_objects/
├── evaluation/
│   ├── evaluation_pipeline.py
│   ├── backtest_data_provider.py
│   ├── evaluation_cache.py
│   ├── run_config_builder.py
│   └── fitness_service.py
├── execution/
│   ├── universal_strategy.py
│   ├── strategy_runtime_state.py
│   ├── entry_decision_engine.py
│   ├── order_manager.py
│   └── position_manager.py
├── infrastructure/
│   ├── serializers/
│   ├── persistence/
│   └── backtest/
└── api/
    └── auto_strategy_service.py
```

重要なのはディレクトリ名そのものではなく、以下の境界を明確にすること。

- domain model
- domain operator
- evaluation pipeline
- experiment orchestration
- infrastructure

## 段階的な実施順

### Phase 1: 境界を壊さずに分離する

- [x] `StrategyRuntimeState` を追加し、strategy helper が直接 private field を触らないようにする
- [x] `ga_engine` から evaluator の private API 依存を外す
- [x] serializer の内部 helper を分割し、公開 API は維持する

成功条件:

- 既存の import path を大きく変えずに内部委譲へ置き換えられる

### Phase 2: 冗長 surface を整理する

- `inject_seeds_into_population` の扱いを一本化
- 未使用設定クラスと `settings.py` の削除可否を判定
- `utils.__init__` の再 export を縮小

成功条件:

- app 本体の import graph が短くなる

### Phase 3: 互換レイヤを縮小する

- test を新 path に移行
- `core.__init__` の shim を段階的に削除

成功条件:

- package 構造と import path が一致する

### Phase 4: 実験実行基盤を再設計する

- [x] `ExperimentManager` の global registry を独立サービス化
- [x] framework 依存を application 層に寄せる

成功条件:

- queue 実行や外部 worker に移しやすい形になる

## テスト方針

リファクタリング時は以下を必須にする。

- strategy 実行の回帰テスト
- evaluator のキャッシュ・walk-forward・OOS テスト
- serializer の round-trip テスト
- seed strategy 注入の挙動テスト
- experiment 開始・停止・完了の統合テスト

特に、互換 shim を剥がすタイミングでは import path 変更だけのテスト修正で終わらせず、public API をどこに固定するかを明示すること。

## 着手順の提案

最初の着手順は以下が良い。

1. [x] `UniversalStrategy` に `StrategyRuntimeState` を導入
2. [x] `StrategyGene` から mutate / crossover を分離
3. [x] `IndividualEvaluator` を pipeline 構成へ分割
4. [x] 冗長 export と未使用 config を削減

この順なら、リスクの高い箇所を先に改善しつつ、公開面の破壊を最小限に抑えられる。
