# Auto Strategy リファクタリング提案書

## 概要

`backend/app/services/auto_strategy` パッケージ全体のコード品質向上のためのリファクタリング案です。
DRY (Don't Repeat Yourself)、単一責任の原則 (SRP)、メンテナンス性の向上を主な目的とします。

---

## 1. 設定管理 (`config` ディレクトリ)

### 課題

- **設定の重複**: `ga.py` (GASettings) と `ga_runtime.py` (GAConfig) に、`population_size` や `generations` などの GA 関連設定が重複して定義されています。
- **検証ロジックの重複**: `BaseConfig` と `AutoStrategyConfig` の `validate` メソッドに、必須フィールドチェック、範囲チェック、型チェックなどの同じロジックが繰り返し実装されています。
- **定数の散在**: 設定値に関連する定数が `constants.py` と各設定ファイルに分散しており、管理が煩雑です。

### 提案

1. **設定クラスの責務分離 (SRP/DRY)**

   - `GASettings` を GA の静的な設定を集約する唯一のクラスとします。
   - `GAConfig` (実行時設定) は `GASettings` のインスタンスを内包し、実行時固有のパラメータ（`progress_callback`など）のみを管理するように変更します。これにより、設定の重複を完全に排除します。

   ```python
   # 変更案: ga_runtime.py
   @dataclass
   class GAConfig(BaseConfig):
       ga_settings: GASettings = field(default_factory=GASettings)
       # ... 実行時固有のパラメータ ...
   ```

2. **検証ロジックの共通化 (DRY)**

   - `BaseConfig` の `validate` メソッドを拡張し、より汎用的な検証ヘルパー（範囲、型、必須項目）を提供します。
   - 他のすべての設定クラス (`AutoStrategyConfig`, `GASettings` など) は、この共通検証ロジックを呼び出すように統一します。

3. **定数の一元管理**
   - `constants.py` に散在する設定のデフォルト値を、それが使用される各設定クラス (`GASettings`, `TPSLSettings`など) のクラス変数または `default_factory` 内に移動させます。これにより、設定とそのデフォルト値の関連性が明確になります。

---

## 2. コアロジック (`core` ディレクトリ)

### 課題

- **責務の混在**: `genetic_operators.py` 内の交叉・突然変異関数が、ビジネスロジック（`StrategyGene`）とフレームワーク（DEAP の`list`）の両方を扱っており、複雑性が増しています。
- **巨大なメソッド**: `ga_engine.py` の `run_evolution` メソッドが長大で、単一目的最適化と多目的最適化（NSGA-II）のロジックが混在しています。
- **ロジックの重複**: `individual_evaluator.py` のフィットネス計算メソッド内に、複数のメトリクス取得ロジックが重複しています。

### 提案

1. **遺伝的演算子の責務分離 (SRP)**

   - `crossover_strategy_genes` と `mutate_strategy_gene` は、`StrategyGene` オブジェクトのみを扱うように修正します。
   - DEAP のツールボックスに登録する際に、`list` と `StrategyGene` の相互変換を行うラッパー関数を用意します。これにより、コアロジックが DEAP ライブラリから独立し、テストが容易になります。

2. **進化アルゴリズムの分割 (SRP/メンテナンス性)**

   - `ga_engine.py` 内に `EvolutionRunner` のようなヘルパークラスを作成します。
   - `run_single_objective` と `run_multi_objective` のようにメソッドを分割し、それぞれのアルゴリズムに特化したロジックをカプセル化します。`run_evolution` は、設定に応じて適切なメソッドを呼び出すファサードとして機能します。

3. **フィットネス計算の共通化 (DRY)**
   - `individual_evaluator.py` に、バックテスト結果から主要なパフォーマンスメトリクスを抽出するプライベートメソッド `_extract_performance_metrics` を作成します。
   - `_calculate_fitness` と `_calculate_multi_objective_fitness` は、この共通メソッドを呼び出してフィットネスを計算するようにします。

---

## 3. 戦略生成 (`generators` ディレクトリ)

### 課題

- **巨大なメソッド**: `condition_generator.py` の `generate_balanced_conditions` メソッドが、巨大な if-elif 文で戦略タイプを分岐しており、新しい戦略タイプの追加が困難です。
- **複雑なロジック**: `random_gene_generator.py` の `_generate_random_indicators` メソッド内に、指標の構成を調整するロジック（トレンド系指標の強制追加など）が複雑に絡み合っています。
- **クラス間の密結合**: `gene_factory.py` の `SmartGeneGenerator` が `RandomGeneGenerator` を直接インスタンス化しており、結合度が高くなっています。

### 提案

1. **ストラテジーパターンの適用 (SRP/メンテナンス性)**

   - `condition_generator.py` をリファクタリングし、各戦略タイプ（`DifferentIndicatorsStrategy`, `ComplexConditionsStrategy`など）を個別のクラスとして実装します。
   - `ConditionGenerator` は、これらの戦略クラスを呼び出すファクトリーまたはコンテキストクラスの役割を担います。これにより、新しい戦略の追加が容易になります。

2. **指標構成ロジックの分離 (SRP)**

   - `random_gene_generator.py` から指標の構成ロジックを `IndicatorCompositionService` のような新しいクラスに分離します。
   - このサービスは「トレンド指標を 1 つ以上含める」「MA クロス戦略を試みる」といったルールベースの構成を担当し、`RandomGeneGenerator` はそれを呼び出すだけにします。

3. **依存性注入 (DI) の採用 (メンテナンス性)**
   - `GeneGeneratorFactory` や `SmartGeneGenerator` が、依存する他のジェネレーターをコンストラクタで受け取るように変更します（依存性注入）。これにより、クラス間の結合度が下がり、単体テストが容易になります。

---

## 4. サービスと永続化 (`services` ディレクトリ)

### 課題

- **巨大なメソッド**: `experiment_persistence_service.py` の `save_experiment_result` メソッドが、最良戦略の保存、詳細バックテストの実行、その他戦略の保存、パレート最適解の保存など、多数の責務を負っており、非常に長大です。
- **複数の責務**: `auto_strategy_service.py` の `start_strategy_generation` メソッドが、設定検証、DB 操作、バックグラウンドタスク登録など、複数の異なる関心事を扱っています。

### 提案

1. **永続化ロジックの分割 (SRP/メンテナンス性)**

   - `save_experiment_result` メソッドを、責務ごとにプライベートメソッドに分割します。
     - `_save_best_strategy(...)`
     - `_run_and_save_detailed_backtest(...)`
     - `_save_pareto_front_strategies(...)`
     - `_save_other_population_strategies(...)`
   - これにより、各メソッドの責務が明確になり、コードの可読性とメンテナンス性が大幅に向上します。

2. **サービスメソッドの責務明確化 (SRP)**
   - `start_strategy_generation` 内の各処理（設定検証、DB 作成など）を、それぞれ小さなプライベートメソッドに切り出します。これにより、メインのメソッドは処理の流れを追うだけのシンプルなものになります。

---

以上の提案を実行することで、`auto_strategy` パッケージ全体がより堅牢で、拡張しやすく、メンテナンスしやすい構造になると考えられます。
