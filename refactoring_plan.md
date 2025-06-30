# リファクタリング計画

## 目的

コードの重複を排除し、モジュール間の責務を明確にすることで、可読性、保守性、拡張性を向上させる。

## 計画

### 3. `BacktestDataService` のデータ検証メソッドの整理

- **現状:** `backtest_data_service.py` の `_validate_dataframe` と `_validate_extended_dataframe` が類似した検証ロジックを持っている。
- **改善案:**
  - 共通の検証ロジックを抽出したプライベートメソッドを作成し、各メソッドがその共通ロジックを呼び出すようにする。

### 4. `StrategyGene` の `parameters` プロパティの責務の明確化

- **現状:** `auto_strategy/models/strategy_gene.py` の `parameters` プロパティが、`backtesting.py` の `Strategy` クラスに渡すためのパラメータを抽出するロジックを含んでいる。
- **改善案:**
  - この変換ロジックを `auto_strategy/utils/gene_converter.py` に移動させ、`StrategyGene` は純粋なデータモデルとしての責務に集中させる。

### 5. データ取得・保存 (`fetch_and_save_..._data`) パターンの共通化

- **現状:** `funding_rate_service.py`, `market_data_service.py`, `open_interest_service.py` の `fetch_and_save_..._data` メソッドが共通のパターンを持っている。
- **改善案:**
  - `bybit_service.py` に、データ取得関数、データ変換関数、リポジトリ、および関連するパラメータを引数として受け取る汎用的な `_fetch_and_save_data` メソッドを実装し、各サービスがそれを呼び出すようにする。

## 今後の進め方

- 上記の計画に基づき、各項目を個別のタスクとして実施する。
- 各タスクの完了後には、関連するテストを実行し、機能が損なわれていないことを確認する。
- 必要に応じて、新しいテストを追加し、リファクタリングの安全性を確保する。
