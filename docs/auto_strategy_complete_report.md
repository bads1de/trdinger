# auto-strategy 機能 統合修正レポート

## 概要

本レポートは、auto-strategy 機能で発生した一連の問題（取引回数 0 問題、マージン不足エラー、指標値の固定化問題）の調査、原因分析、修正内容、そして最終的な解決策を統合的に記録したものです。

---

## 1. 問題の時系列と修正の変遷

### 1.1 初期問題：取引回数 0 問題

- **主要症状**: 戦略のバックテスト実行時に取引回数が常に 0 になる。
- **根本原因**: `IndicatorInitializer`における指標初期化プロセスの失敗。具体的には、指標関数が`backtesting.py`の期待する配列ではなくスカラー値を返していたため、互換性の問題が発生していました。
- **修正内容**:
  - `IndicatorInitializer`を修正し、指標関数がデータ長に応じた適切な配列を返すように変更。
  - データが不足する場合のパディング処理を追加。
  - `ConditionEvaluator`に詳細なデバッグログを追加し、条件評価プロセスを可視化。具体的には、従来の`print`文によるデバッグ出力から、より堅牢な`logging`モジュールを使用したログ出力に移行しました。
- **結果**: システム的な指標初期化エラーは解消されました。しかし、取引回数 0 の問題は依然として発生しており、その原因が「厳しすぎる戦略条件」と「短すぎるテスト期間」という、設定上の問題であることが判明しました。

### 1.2 次の問題：マージン不足エラー

- **エラー**: `Broker canceled the relative-sized order due to insufficient margin`
- **原因**:
  1. `self.buy()`が引数なしで呼び出され、`backtesting.py`のデフォルトである全資産(100%)での注文が試行されていた。
  2. `position_size: 0.001`という設定が小さすぎた。
- **修正内容**:
  - ポジションサイズを計算するロジックを実装。具体的には、`backend/app/core/strategies/risk_management/calculators.py`内の`RiskCalculator`クラスに`calculate_optimal_position_size`メソッドが実装され、これを利用して注文サイズが決定されるようになりました。
  - `self.buy(size=position_size)`のように、取得したポジションサイズを明示的に指定して注文するように修正。
  - デバッグ用に固定金額での注文（`self.buy(size=100)`）も試行し、エラーを回避。
- **結果**: マージン不足エラーは回避され、買い注文の実行が確認されました。しかし、新たに「注文は実行されるが取引履歴に記録されない」という問題が浮上しました。

### 1.3 根本原因の特定：`backtesting.py`の誤解と指標値の固定化

詳細な調査の結果、一連の問題の根本原因が **`backtesting.py` の `I` メソッドの設計思想の誤解** にあることが確定しました。

- **失敗の核心**: 指標を**事前計算**し、その**静的な結果**を返すラッパー関数を `I` メソッドに渡していたこと。
- **本来あるべき姿**: `I` メソッドには**指標計算ロジック（関数）そのもの**を渡し、計算を `backtesting.py` に委譲する必要がある。

この誤解が原因で、以下の問題が連鎖的に発生していました。

1.  **指標値の固定化**: `backtesting.py`が各バーで指標を動的に再計算しないため、同じ値が返され続ける。
2.  **取引記録が 0 になる**: イグジット条件が正しく評価されず、エントリーから決済までの一連の取引が完了しないため、取引として記録されない。

---

## 2. 最終的な解決策：動的計算への移行

現在の「事前計算」アプローチを完全に破棄し、**指標の計算ロジックを `backtesting.py` に委譲する**設計に移行します。

### `IndicatorInitializer`の修正案

`auto-strategy`機能の中核となる指標初期化ロジック（ここでは概念的に`IndicatorInitializer`クラスとして示します）を以下のように改修し、`I`メソッドに計算関数そのものを渡すようにします。

**注:** 以下のコードは、特定のファイル（例: `indicator_initializer.py`）に存在するものではなく、関連する複数のファクトリークラスやサービスに実装されているロジックを概念的に表現したものです。

```python
# 指標初期化ロジックの修正案
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from backtesting.lib import _Indicator
from app.core.services.indicators.config.indicator_definitions import indicator_registry
from app.core.utils.data_utils import convert_to_series

logger = logging.getLogger(__name__)

class IndicatorInitializer:

    def initialize_indicator(
        self, indicator_gene: IndicatorGene, data, strategy_instance
    ) -> Optional[List[str]]:
        """
        単一指標をbacktesting.pyに正しく登録する（動的計算版）
        """
        try:
            indicator_type = indicator_gene.type
            parameters = indicator_gene.parameters

            indicator_config = indicator_registry.get_indicator_config(indicator_type)
            if not indicator_config:
                logger.error(f"指標設定が見つかりません: {indicator_type}")
                return None

            adapter_function = indicator_config.adapter_function
            required_data_keys = indicator_config.required_data

            input_data = [getattr(strategy_instance.data, key.capitalize()) for key in required_data_keys]

            param_values = [parameters[p.name] for p in indicator_config.parameters]

            indicator_result = strategy_instance.I(
                adapter_function, *input_data, *param_values
            )

            json_indicator_name = indicator_registry.generate_json_name(indicator_type)

            if isinstance(indicator_result, tuple):
                output_names = []
                for i, res in enumerate(indicator_result):
                    name = f"{json_indicator_name}_{i}"
                    strategy_instance.indicators[name] = res
                    output_names.append(name)
                logger.info(f"複数値指標を登録: {output_names}")
                return output_names
            elif isinstance(indicator_result, _Indicator):
                strategy_instance.indicators[json_indicator_name] = indicator_result
                logger.info(f"単一値指標を登録: {json_indicator_name}")
                return [json_indicator_name]
            else:
                logger.error(f"不明なIメソッド返り値タイプ: {type(indicator_result)}")
                return None

        except Exception as e:
            logger.error(f"指標初期化エラー ({indicator_gene.type}): {e}", exc_info=True)
            return None
```

### 期待される効果

1.  **指標値の固定化問題の完全解決**: `backtesting.py` が各バーで指標を動的に計算するため、値は正しく更新されます。
2.  **取引記録の正常化**: イグジット条件が正しく評価されるようになり、エントリーから決済までの一連の取引が完了するため、取引回数が正確に記録されます。
3.  **コードの簡素化**: 事前計算のための複雑なロジックが不要になり、コードベースがシンプルになります。

---

## 3. まとめと今後の推奨事項

### 技術的な学び

1.  **外部ライブラリとの連携**: `backtesting.py`のような外部ライブラリを使用する際は、その設計思想を深く理解することが不可欠です。
2.  **デバッグの重要性**: 詳細なログ出力は、問題の根本原因を特定する上で極めて重要です。
3.  **段階的な問題解決**: 複雑な問題は、一つずつ切り分けて解決していくアプローチが有効です。

### 次のステップ

1.  上記修正案に基づき、`indicator_initializer.py` をリファクタリングする。
2.  `MACD` のような複数値を返す指標の結果を、条件評価 (`ConditionEvaluator`) で正しく扱えるように名前解決のルールを確定・実装する。
3.  単体テストおよび結合テストを実施し、複数の指標（RSI, MACD, SMA など）で戦略が正常に動作することを確認する。

これらの修正を適用することで、auto-strategy 機能は本来の性能を発揮し、実用的なレベルに到達できると考えられます。
