# 自動戦略生成(`auto_strategy`)モジュール リファクタリング提案

## 1. はじめに

`backend/app/services/auto_strategy/`ディレクトリ全体のコードを拝見しました。遺伝的アルゴリズム（GA）を用いた戦略生成のコアロジックが非常によく構造化されており、多くの機能が実装されていることがわかります。

このドキュメントでは、現状のコードベースのさらなる可読性、保守性、拡張性の向上を目的としたリファクタリングの提案をまとめます。

## 2. 全体的な課題と提案

いくつかのモジュールで共通して見られる課題と、それに対する全体的な改善案です。

### 2.1. ユーティリティモジュールの乱立

**課題:**
`utils`ディレクトリ内に `auto_strategy_utils.py`, `common_utils.py`, `gene_utils.py`, `error_handling.py` など、複数のユーティリティファイルが存在し、責務の境界がやや曖昧になっています。特に `gene_utils.py` の機能は他のモジュールに吸収されつつあり、ファイル自体が冗長になっています。

**提案:**
- **`gene_utils.py`の廃止:** このファイルの機能は`models/gene_serialization.py`や`config/constants.py`に完全に移譲し、ファイルを削除します。
- **`auto_strategy_utils.py`への集約:** `auto_strategy`モジュール固有のユーティリティ関数（遺伝子操作、パラメータ正規化など）は`auto_strategy_utils.py`に集約します。
- **`common_utils.py`の役割見直し:** プロジェクト全体で汎用的に使える関数（安全な型変換など）のみを`common_utils.py`に残し、`app/utils/`ディレクトリへの移動を検討します。

### 2.2. 依存性注入（DI）の促進

**課題:**
`MLOrchestrator`や`AutoStrategyService`などのサービスクラスが、内部で他のサービスクラスを直接インスタンス化しています。これにより、コンポーネント間の結合度が高くなり、単体テストが困難になっています。

**提案:**
FastAPIの依存性注入システム（`Depends`）を積極的に活用し、サービスクラスが必要とする他のサービスやリポジトリをコンストラクタやメソッドの引数として受け取るように変更します。これにより、モックを使ったテストが容易になり、コンポーネントの再利用性も向上します。

```python
# 修正前 (例: MLOrchestrator)
class MLOrchestrator:
    def __init__(self):
        self.ml_training_service = MLTrainingService() # 直接インスタンス化

# 修正後
class MLOrchestrator:
    def __init__(self, ml_training_service: MLTrainingService = Depends()):
        self.ml_training_service = ml_training_service
```

## 3. 主要なリファクタリング項目

各モジュールにおける具体的な改善点です。

### 3.1. `models` パッケージ

#### `gene_serialization.py`
- **責務の分離:** `GeneSerializer`クラスが現在、辞書/JSONへのシリアライズと、GA用の数値リストへのエンコード/デコードという2つの異なる責務を担っています。これらを`GeneSerializer`と`GeneEncoder`のように、責務ごとにクラスを分離することを推奨します。
- **デコード処理の単純化:** `from_list`メソッド内で`SmartConditionGenerator`を呼び出して条件を再生成するロジックは、エンコードとデコードの対称性を崩しています。デコード処理は、エンコードされた情報を忠実に復元することに専念させ、複雑な再生成ロジックは避けるべきです。

### 3.2. `calculators` パッケージ

#### `indicator_calculator.py`
- **冗長な指標登録処理の削除:** `init_indicator`メソッド内で、`__dict__`, `setattr`, `indicators`辞書, クラス変数への設定と、4つの方法で指標を登録していますが、これは過剰です。`backtesting.py`の仕様を確認し、最も確実な1つの方法（通常は`setattr`で十分な場合が多い）に絞ることで、コードが大幅に簡潔になります。
- **フォールバック処理の改善:** `calculate_indicator`内のSMA/EMA/WMAのフォールバック計算は、特定の指標に限定されています。このロジックは`TechnicalIndicatorService`側に移譲するか、より汎用的なフォールバック機構として設計し直すことで、拡張性が向上します。

### 3.3. `generators` パッケージ

#### `random_gene_generator.py`
- **複雑なロジックの単純化:** `_generate_random_indicators`メソッド内の「成立性の底上げ」や「カバレッジ向上」のためのロジックが非常に複雑化しています。これらの補助ロジックは、それぞれ独立した責務を持つ小さなプライベートメソッドに分割することで、可読性と保守性が大幅に向上します。

#### `strategy_factory.py`
- **`next`メソッドの可読性向上:** 売買ロジックを担う`next`メソッドが長大になっています。ポジションサイズの計算、TP/SL価格の計算、注文実行などの各ステップを、それぞれ別のプライベートメソッドに切り出すことで、メインのロジックの流れが追いやすくなります。

### 3.4. `config` パッケージ

#### `shared_constants.py`
- **定数の動的生成:** `VALID_INDICATOR_TYPES`のような定数リストが手動で管理されています。`get_all_indicator_ids`のように`TechnicalIndicatorService`から情報を取得して動的に生成することで、指標の追加・削除に強くなり、保守の手間が省けます。

## 4. まとめ

ここに挙げた提案は、コードの品質をさらに一段階引き上げるためのものです。特に、以下の点を重視しました。

- **責務の明確化:** 各クラス・モジュールが単一の責任を持つようにする。
- **依存関係の疎結合化:** DIを活用し、テストしやすく柔軟なコンポーネント構造を目指す。
- **コードの簡潔化:** 重複や冗長なロジックを排除し、可読性を高める。

これらのリファクタリングを段階的に進めることで、今後の機能追加や仕様変更にも迅速かつ安全に対応できる、より堅牢なコードベースを構築できると確信しています。
