# リファクタリング計画：auto_strategy と indicators

## 目的

本ドキュメントは、`backend/app/core/services/auto_strategy` および `backend/app/core/services/indicators` ディレクトリ内のコードベースにおける重複、冗長性、および責務の不明確な箇所を特定し、それらに対するリファクタリング案を提示することを目的とします。これにより、コードの可読性、保守性、拡張性を向上させます。

## 概要

以下の主要なリファクタリング領域を特定しました。

1.  **冗長なデータ変換の排除**
2.  **`IndicatorCalculator` の計算ロジックの合理化**
3.  **`IndicatorCalculator` 内のパラメータハンドリングの洗練** (セクション 2 に統合)
4.  **OI/FR データソースのハードコードされたフォールバック値の見直し**
5.  **指標インスタンス作成とパラメータバリデーションの調和**

---

## 2. `IndicatorCalculator` の計算ロジックの合理化

### 現状の課題

`IndicatorCalculator` クラスは、指標の計算ロジックにおいて冗長性や複雑さを抱えています。特に、`_setup_indicator_config` での新旧設定の混在、`calculate_indicator` 内の `_calculate_from_config` と `_calculate_from_adapter` という二つの計算パス、そして `_calculate_from_adapter` 内でのパラメータの手動再解釈が主な課題です。

- `backend/app/core/services/auto_strategy/factories/indicator_calculator.py`:
  - `_setup_indicator_config` メソッドは、`indicator_registry` からの `IndicatorConfig` インスタンスと、ハードコードされたレガシー設定の両方を `self.indicator_config` に格納しています。これにより、設定管理が複雑になっています。
  - `calculate_indicator` メソッドは、`indicator_type` が `self.indicator_config` に存在するかどうかで処理を分岐させ、`_calculate_from_config` または `_calculate_from_adapter` を呼び出しています。
  - `_calculate_from_adapter` メソッドは、特定の指標タイプ（SMA, EMA, RSI など）に対してパラメータを手動で再解釈しており、これは `IndicatorConfig` の `parameters` 定義との重複や不整合を生む可能性があります。このメソッドは冗長であり、`IndicatorConfig` を中心とした設計に反しています。
  - `_prepare_parameters_for_indicator` メソッドは、新しい JSON 形式の設定とレガシー形式の設定の両方を扱っていますが、`IndicatorConfig` の `parameters` 定義に完全に依存するように強化する必要があります。
  - `_generate_indicator_name` というメソッドは存在しません。
  - `_handle_complex_result` は `config["indicator_config"].generate_json_name()` を使用しており、これは適切です。しかし、`macd_handler` と `bb_handler` のロジックはハードコードされており、`IndicatorConfig` の `result_handler` をより汎用的に活用できる可能性があります。

これにより、以下の問題が発生しています。

- **ロジックの複雑性**: 2 つの計算パスが存在するため、コードの理解とデバッグが困難です。
- **設定の一貫性の欠如**: JSON 形式への移行が中途半端なため、新旧の設定が混在し、将来的な拡張が難しくなっています。
- **冗長なパラメータハンドリング**: `_calculate_from_adapter` でパラメータを再度解析している部分が冗長です。

### 提案

`IndicatorCalculator` の計算ロジックを合理化し、`IndicatorConfig` を中心とした単一の、より明確なフローに統合することを提案します。これにより、JSON 形式の指標設定を完全に活用し、レガシー互換性をより適切に管理できます。

**変更案の概要:**

1.  **`_setup_indicator_config` の合理化**:
    - `self.indicator_config` には `indicator_registry` から取得した `IndicatorConfig` インスタンスのみを格納するように変更します。レガシー設定のフォールバックは `indicator_registry` または `IndicatorConfig` 自体で処理されるべきです。
    - `_get_legacy_config` メソッドは削除します。
2.  **`_setup_indicator_adapters` の役割の明確化**:
    - このメソッドは、`IndicatorConfig` 内の `adapter_function` フィールドへの参照を提供する役割に限定します。`indicator_registry` から `IndicatorConfig` を取得し、その `adapter_function` を直接使用するように変更します。
3.  **単一の計算フローへの統合**:
    - `calculate_indicator` メソッド内で、直接 `self.indicator_config` を参照し、対応する `adapter_function` と `IndicatorConfig` から取得したパラメータ情報を使用して計算を実行します。
    - `_calculate_from_adapter` メソッドを完全に削除します。
    - `_calculate_from_config` のロジックを `calculate_indicator` に統合し、`_prepare_data_for_indicator` と `_prepare_parameters_for_indicator` を活用して統一されたデータとパラメータの渡し方を実現します。
4.  **`_prepare_parameters_for_indicator` の強化**:
    - このメソッドを強化し、`IndicatorConfig` の `parameters` 定義に完全に依存してパラメータを準備するようにします。これにより、手動でのパラメータ解析が不要になります。
5.  **`_generate_indicator_name` への言及の削除**:
    - `_generate_indicator_name` というメソッドは存在しないため、コード内のその言及を削除します。
6.  **`_handle_complex_result` の改善**:
    - `IndicatorConfig` で定義された `result_handler` に基づいて、計算結果から適切な値を抽出するロジックを保持します。必要に応じて、`result_handler` の種類を拡張し、より汎用的な処理を可能にします。

**期待される効果:**

- **コードの簡素化**: 計算ロジックが単一の明確なパスに統合され、コードベースが大幅に簡素化されます。
- **保守性の向上**: 指標の計算ロジックが一元化されるため、バグ修正や機能追加が容易になります。
- **JSON 形式の完全活用**: `IndicatorConfig` に定義された JSON 形式の情報を最大限に活用し、設定とロジックの一貫性が向上します。
- **パフォーマンスの可能性**: 不要な条件分岐や再解析が減ることで、わずかながらパフォーマンスの改善が期待できます。

---

## 3. OI/FR データソースのハードコードされたフォールバック値の見直し

### 現状の課題

`random_gene_generator.py` の `_generate_threshold_value` メソッドにおいて、Funding Rate と Open Interest の閾値がハードコードされており、`GAConfig` の `threshold_ranges` を十分に活用できていません。これにより、閾値の管理が一元化されておらず、設定変更の際に複数のファイルを修正する必要が生じる可能性があります。

- `backend/app/core/services/auto_strategy/generators/random_gene_generator.py`:
  - `_generate_threshold_value` メソッド内で、`FundingRate` と `OpenInterest` の閾値が直接数値で定義されています。

これにより、以下の問題が発生しています。

- **設定の一貫性の欠如**: 閾値の定義が `GAConfig` と `random_gene_generator.py` の間で分散しています。
- **保守性の低下**: 閾値の変更が必要になった場合、コードを直接修正する必要があり、エラーのリスクが高まります。

### 提案

Funding Rate と Open Interest の閾値生成ロジックを `GAConfig` の `threshold_ranges` に完全に依存するように変更し、ハードコードされた値を排除することを提案します。

**変更案の概要:**

1.  **`random_gene_generator.py` の修正**:
    - `_generate_threshold_value` メソッド内で、`FundingRate` と `OpenInterest` の閾値を生成する際に、`self.threshold_ranges.get("funding_rate", [デフォルト値])` および `self.threshold_ranges.get("open_interest", [デフォルト値])` を使用するように変更します。これにより、`GAConfig` で定義された範囲が優先的に使用されるようになります。
    - `GAConfig` にこれらのデフォルト範囲が定義されていない場合のフォールバックとして、適切なデフォルト値を設定します。

**期待される効果:**

- **設定の一元化**: 全ての閾値が `GAConfig` を通じて管理されるため、一貫性が保証されます。
- **保守性の向上**: 閾値の変更が容易になり、コードの修正なしで設定を調整できるようになります。
- **柔軟性の向上**: `GAConfig` を介して、より多様な閾値の範囲を動的に設定できるようになります。

---

## 4. 指標インスタンス作成とパラメータバリデーションの調和

### 現状の課題

当初、`backend/app/core/services/indicators/factories/indicator_factory.py` に `IndicatorFactory` クラスが存在し、`backend/app/core/services/auto_strategy/factories/indicator_initializer.py` との間で指標インスタンス作成とパラメータバリデーションロジックが重複していると推測していました。

しかし、その後の調査で `IndicatorFactory` というクラスは現在のコードベースには存在しないことが判明しました。

現在のところ、指標のインスタンス作成と初期化は主に `backend/app/core/services/auto_strategy/factories/indicator_initializer.py` および `backend/app/core/services/auto_strategy/factories/indicator_calculator.py` で行われているようです。具体的には以下の課題が見られます。

- `indicator_initializer.py` の `initialize_indicator` メソッドが、`IndicatorCalculator` を呼び出して指標の計算と結果取得を行っています。
- `_create_indicator_instance` のような直接的なインスタンス生成ロジックは `indicator_initializer.py` にはありませんが、`calculate_indicator_only` メソッドが `IndicatorCalculator` を直接利用しています。
- パラメータのバリデーションは、`IndicatorConfig` に定義があるにも関わらず、呼び出し側で一貫して適用されているか不明確です。

これにより、以下の問題が発生しています。

- **ロジックの分散**: 指標インスタンスの作成とパラメータバリデーションに関する責任が、`IndicatorInitializer` と `IndicatorCalculator` の間で明確に分かれていない可能性があります。
- **バリデーションの不透明性**: パラメータがどこで、どのようにバリデーションされているかが一貫していません。
- **保守性の低下**: 指標の追加や変更時に、複数のファイルを修正する必要があり、エラーのリスクが高まります。
- **責務の不明確さ**: どのモジュールが指標のインスタンス作成とパラメータバリデーションの主要な責務を持つべきかが不明確です。

### 提案

指標のインスタンス作成とパラメータバリデーションのロジックを、指標の「初期化」と「計算」という役割分担に基づき、より明確に責任を割り当てることを提案します。`IndicatorInitializer` を指標のセットアップと `backtesting.py` 戦略インスタンスへの統合に特化させ、`IndicatorCalculator` は純粋に指標の計算ロジックに集中させます。

**変更案の概要:**

1.  **`IndicatorInitializer` を指標のセットアップと統合に特化**:
    - `initialize_indicator` メソッド内で、`IndicatorConfig` を参照し、その情報に基づいて必要なデータ準備とパラメータのバリデーションを行います。
    - パラメータのバリデーションには、「1. パラメータ生成とバリデーションの一元化」で提案した `ParameterService` を利用することを検討します。
    - バリデーション済みのデータとパラメータを `IndicatorCalculator` に渡し、計算結果を受け取ります。
    - 計算結果を `backtesting.py` の戦略インスタンスに適切に統合する責務を担います。
2.  **`IndicatorCalculator` は純粋な計算ロジックに集中**:
    - 「4. `IndicatorCalculator` の計算ロジックの合理化」の提案に基づき、`IndicatorConfig` を唯一の情報源として計算を実行します。
    - パラメータのバリデーションは `IndicatorInitializer` が実施し、`IndicatorCalculator` にはバリデーション済みのパラメータが渡されることを前提とします。
3.  **パラメータバリデーションの一元化と強制**:
    - 全ての指標インスタンス作成（または計算開始前）において、`ParameterService` を介してパラメータのバリデーションを強制します。これにより、不正なパラメータが指標計算に渡されることを防ぎます。

**期待される効果:**

- **単一責任の原則**: 各モジュールの責務が明確になり、コードの品質が向上します。
- **一貫したバリデーション**: パラメータのバリデーションが強制され、不正なデータによるエラーを防ぎます。
- **保守性の向上**: 指標の追加や変更が容易になり、エラー発生のリスクが減少します。
- **可読性の向上**: コードのフローが明確になり、各モジュールの役割がより鮮明になります。

---

## 結論

本リファクタリング計画は、`auto_strategy` および `indicators` ディレクトリ内のコードの重複と複雑さを解消し、システムの全体的な健全性を向上させることを目指しています。提案された変更は、各モジュールの責務を明確にし、コードの一貫性を高め、将来的な機能拡張や保守を容易にするでしょう。

これらのリファクタリングは、一歩ずつ慎重に進める必要があります。特に、既存の機能への影響を最小限に抑えるため、単体テストと結合テストを徹底しながら実施することが重要です。TDD の原則に基づき、リファクタリングの各ステップでテストが確実にパスすることを確認します。

この計画が、より堅牢で保守しやすい取引戦略バックテストシステムの構築に貢献することを期待します。

---
