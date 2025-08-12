# auto_strategy パッケージのコード重複と統合に関する提案

このドキュメントは、`auto_strategy`パッケージ内のコードベースを分析し、コードの重複排除、責務の集約、および全体的な保守性と可読性の向上のためのリファクタリング案をまとめたものです。

## 1. 計算ロジックの集約 (`calculators` vs `generators`)

現在、TP/SL（テイクプロフィット/ストップロス）やポジションサイジングに関する計算ロジックが、`calculators`と`generators`の複数のモジュールに分散しています。これを一つのサービスに集約することで、見通しが良くなり、管理が容易になります。

### 課題

-   **TP/SL計算の分散**:
    -   `calculators/tpsl_calculator.py`: 基本的なTP/SL価格を計算します。
    -   `calculators/risk_reward_calculator.py`: リスクリワード比に基づいたTPを計算します。
    -   `generators/statistical_tpsl_generator.py`: 統計データに基づいてTP/SLを生成します。
    -   `generators/volatility_based_generator.py`: ATRなどのボラティリティに基づいてTP/SLを生成します。
    これらはすべてTP/SLを決定するための異なる戦略であり、ロジックが分散しているため全体像の把握が困難です。

-   **ポジションサイジング計算の曖昧さ**:
    -   `calculators/position_sizing_calculator.py`: `PositionSizingCalculatorService`が`PositionSizingGene`に基づいて計算を行っています。
    -   `models/gene_position_sizing.py`: `PositionSizingGene`クラス内にも`calculate_position_size`メソッドが存在し、役割分担がやや曖昧です。

### 提案

-   **`TPSLCalculatorService`の導入**:
    -   `TPSLGene`の`method`（`FIXED_PERCENTAGE`, `RISK_REWARD_RATIO`, `VOLATILITY_BASED`など）に応じて、適切な計算ロジックを呼び出す統一的なサービス `TPSLCalculatorService` を作成します。
    -   現在`calculators`と`generators`に分散しているTP/SL関連の計算ロジックを、この新しいサービスにすべて集約します。
    -   これにより、TP/SLの計算方法を追加・変更する際に、修正箇所が1つのサービスに限定され、保守性が向上します。

-   **`PositionSizingCalculatorService`へのロジック集約**:
    -   `PositionSizingGene`クラスからは計算ロジックを削除し、純粋なデータクラス（Data Class）としての役割に徹させます。
    -   すべてのポジションサイズ計算ロジックを`PositionSizingCalculatorService`に完全に移管し、責務を明確化します。

## 2. 遺伝的演算子の整理 (`operators` vs `engines`)

遺伝的アルゴリズムの核心である交叉・突然変異のロジックが、複数の場所に重複して存在しています。

### 課題

-   `operators/genetic_operators.py`と`engines/evolution_operators.py`の両方に、`crossover_strategy_genes`と`mutate_strategy_gene`という類似した機能を持つ関数が定義されています。
-   `ga_engine.py`は`operators/genetic_operators.py`から関数をインポートしており、`engines/evolution_operators.py`は現在使用されていないか、古いコードの残骸である可能性が高いです。

### 提案

-   `engines/evolution_operators.py`を削除し、遺伝的演算子のロジックを`operators/genetic_operators.py`に一本化します。
-   これにより、コードの重複が排除され、GAのコアロジックがどこで定義されているかが明確になります。

## 3. 指標名解決ロジックの改善 (`IndicatorNameResolver`)

`core/indicator_name_resolver.py`は、条件評価時に文字列から指標の値を取得する重要な役割を担っていますが、現在の実装には保守性の課題があります。

### 課題

-   `try_resolve_value`メソッド内に、`MACD_0`, `BB_1`, `STOCH_0`といった特定の指標名とその出力インデックスに関するハードコードされたマッピングが多数存在します。
-   この実装では、新しい複数出力の指標を追加したり、既存の指標の出力形式を変更したりする際に、このリゾルバーを毎回手動で修正する必要があり、エラーが発生しやすくなります。

### 提案

-   **指標レジストリの拡張**:
    -   `app/services/indicators/config/indicator_config.py`内の`indicator_registry`に、各指標のデフォルト出力名やエイリアス（例: `MACD`のデフォルトは`MACD_0`）といったメタ情報を追加します。
-   **`IndicatorNameResolver`の汎用化**:
    -   `IndicatorNameResolver`が、この拡張された指標レジストリを参照して動的に名前解決を行うようにリファクタリングします。
    -   これにより、ハードコードされたロジックが不要になり、新しい指標の追加が`indicator_registry`への登録だけで完結するようになります。

## 4. エンコード/デコード層の簡素化 (`models/gene_encoding.py`)

遺伝子のエンコード・デコードに関するモジュール構造がやや複雑です。

### 課題

-   `models/gene_encoder.py` (Encoderクラス) と `models/gene_decoder.py` (Decoderクラス) に実際のロジックが実装されています。
-   `models/gene_encoding.py` (GeneEncoderクラス) が上記2つのクラスをラップするファサードとして機能していますが、クラス名が紛らわしく、冗長に見える可能性があります。

### 提案

-   **クラス名の明確化**:
    -   ファサードクラスの名前を`GeneEncodingFacade`などに変更し、役割を明確にします。
-   **(代替案) ロジックの統合**:
    -   あるいは、エンコードとデコードのロジックを`gene_encoding.py`に統合し、モジュール数を減らして構造を簡素化することも考えられます。

## 5. 定数管理の一元化 (`constants.py`)

### 課題

現在、`constants.py`に演算子（`OPERATORS`）やデータソース（`DATA_SOURCES`）のリストが定義されています。

しかし、`models/gene_validation.py`の中にある`GeneValidator`クラスでも、`_get_valid_operators`や`_get_valid_data_sources`といったメソッド内で、実質的に同じ内容のリストが再定義されています。

これはコードの重複であり、将来的に演算子やデータソースの種類を追加・変更する際に、複数箇所を修正する必要がでてきます。その結果、修正漏れや不整合が発生するリスクがあります。

### 提案

`GeneValidator`クラス内のメソッドでリストを再定義するのではなく、`constants.py`から直接定数をインポートして使用するように変更することを提案します。

```python
# models/gene_validation.py の修正案

from ..constants import OPERATORS, DATA_SOURCES # ← インポートを追加

class GeneValidator:
    def __init__(self):
        # メソッド呼び出しの代わりに、インポートした定数を直接利用する
        self.valid_indicator_types = self._get_valid_indicator_types()
        self.valid_operators = OPERATORS
        self.valid_data_sources = DATA_SOURCES

    # ... _get_valid_operators と _get_valid_data_sources メソッドは不要になるので削除 ...
```

これにより、関連する定数が`constants.py`に集約され、一元管理が可能となり、保守性が向上します。

## まとめ

上記のリファクタリング提案を実装することで、`auto_strategy`パッケージはよりクリーンで、責務が明確になり、将来的な機能追加や変更が容易になります。特に、計算ロジックの集約、指標名解決の汎用化、そして定数の一元管理は、コードの保守性と拡張性を大幅に向上させる上で効果的です。
