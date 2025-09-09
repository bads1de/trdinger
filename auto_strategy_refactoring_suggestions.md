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

---

以上の提案を実行することで、`auto_strategy` パッケージ全体がより堅牢で、拡張しやすく、メンテナンスしやすい構造になると考えられます。
