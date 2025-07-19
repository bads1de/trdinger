# リファクタリング候補

## 概要

このドキュメントは、コードベース全体を分析し、将来の拡張性、保守性、可読性を向上させるためのリファクタリング候補をまとめたものです。

---

## バックエンド

### 1. 設定管理の完全統一 (High Priority)

#### 現状の問題点

`backend/app/config` ディレクトリ内に、複数の設定管理ファイル (`unified_config.py`, `settings.py`, `market_config.py`) が混在しています。

- `unified_config.py` が最新の統一的な設定管理システムとして設計されている一方で、`settings.py` や `market_config.py` といったレガシーな設定ファイルが後方互換性のために残存しており、設定の参照元が分散し、混乱を招く可能性があります。
- `market_config.py` のコメントには、`unified_config` への移行が推奨されていることが明記されています。

#### 解決案

- **`unified_config.py` への一本化**: アプリケーション全体の設定を `unified_config.py` に完全に集約します。
- **レガシーファイルの廃止**: `settings.py` と `market_config.py` を参照している箇所を全て `unified_config` を参照するように書き換え、古いファイルを削除します。
- **`__init__.py` の整理**: 後方互換性のためのエクスポートを削除し、`unified_config` のみを提供するようにします。

これにより、設定管理の責務が単一のモジュールに集約され、コードの保守性と見通しが大幅に向上します。

### 2. 依存関係の正常化と責務の分離 (High Priority)

#### 現状の問題点

`backend/app/api/fear_greed.py` 内で、トップレベルの `data_collector` モジュールが直接インポートされています。

```python
# backend/app/api/fear_greed.py
from data_collector.external_market_collector import ExternalMarketDataCollector
```

これは、アプリケーションのコアロジック (`app` ディレクトリ) が、外部のデータ収集スクリプトに直接依存していることを意味し、レイヤー化アーキテクチャの原則に反しています。このような依存関係は、循環参照や密結合を引き起こし、将来的な変更を困難にします。

また、`fear_greed.py` のエンドポイント内で、`ExternalMarketDataCollector` を直接インスタンス化しており、ビジネスロジックがAPI層に漏れ出しています。

#### 解決案

- **サービス層の利用**: `fear_greed.py` は、`data_collector` を直接呼び出す代わりに、`app/core/services/data_collection/orchestration/fear_greed_orchestration_service.py` を利用すべきです。オーケストレーションサービスが、データ収集の詳細なロジックをカプセル化します。
- **依存関係の逆転**: `data_collector` が `app` のコンポーネント（リポジトリやサービス）を利用する必要がある場合は、依存性注入（Dependency Injection）などのテクニックを用いて、依存関係の方向を `app` -> `data_collector` ではなく、`data_collector` -> `app` となるように修正します。
- **リポジトリの直接呼び出しの禁止**: `backtest.py` など、他のAPIエンドポイントでも見られるリポジトリの直接呼び出しを禁止し、必ずサービス層を経由するように統一します。

### 3. エラーハンドリングの統一と強化 (Medium Priority)

#### 現状の問題点

`backend/app/api` 内の各エンドポイントで、`try...except`ブロックによる個別のエラーハンドリングが実装されています。`UnifiedErrorHandler` が存在するものの、その利用方法が一貫しておらず、一部のエンドポイントでは直接 `HTTPException` を送出しています。

#### 解決案

- **`UnifiedErrorHandler.safe_execute_async` の徹底**: 全てのAPIエンドポイントでこのメソッドを利用するように統一します。これにより、`try...except` ブロックを削減し、エラーハンドリングロジックを一元管理できます。
- **サービス層でのエラーハンドリング**: 各サービス層のメソッドも、必要に応じて `@safe_ml_operation` デコレータや `unified_operation_context` を活用し、エラーハンドリングとロギングを強化します。

### 4. `scripts` ディレクトリの再編成 (Medium Priority)

#### 現状の問題点

`backend/scripts` ディレクトリには、現在 `utils/db_utils.py` という、データベース関連のユーティリティ関数のみが存在しています。過去には多数のスクリプトが存在していた可能性がありますが、現状ではディレクトリ名とその役割が一致していません。

#### 解決案

- **ユーティリティの移動**: `db_utils.py` の役割を考慮し、`backend/database/utils.py` のような、データベース関連のユーティリティであることが明確にわかる場所へ移動します。
- **ディレクトリの削除**: `scripts` ディレクトリが空になる場合は削除し、プロジェクト構造をシンプルに保ちます。
- **スクリプトの再評価**: もし過去のスクリプトがまだ必要なのであれば、それらを復元し、役割に応じてAPIエンドポイント化するか、あるいはコマンドラインツールとして再設計・整備することを検討します。

---

## フロントエンド

### 1. データ取得用カスタムフックの共通化 (High Priority)

#### 現状の問題点

`frontend/hooks` ディレクトリ内に、特定のAPIエンドポイントからデータを取得するためのカスタムフックが多数存在します（例: `useOhlcvData.ts`, `useFundingRateData.ts`など）。

これらのフックは、内部で汎用的な `useApiCall` フックを呼び出していますが、データ状態の管理（`data`, `limit`など）、データ取得関数の定義、`useEffect` による初期ロードといった点で、多くのコードが重複しています。

#### 解決案

- **共通フックの作成**: データ取得ロジックを抽象化し、より高レベルな共通カスタムフック（例: `useDataFetching.ts`）を作成します。このフックは、APIエンドポイントのURL、パラメータ、成功時のデータ整形ロジックなどを引数として受け取るようにします。
- **既存フックのリファクタリング**: `useOhlcvData` などの既存フックを、この新しい共通フックを利用して書き換えます。これにより、個別のフックは数行のコードで実装できるようになり、コードの重複が大幅に削減されます。

### 2. 共通コンポーネントの再利用促進 (Medium Priority)

#### 現状の問題点

- **ボタンコンポーネント**: `frontend/components/button` ディレクトリに、`AllDataCollectionButton.tsx` や `FearGreedCollectionButton.tsx` など、特定のデータ収集機能に特化したボタンが多数存在します。これらのコンポーネントは、内部で `ApiButton` をラップしており、ロジックの多くが重複しています。
- **テーブルコンポーネント**: 同様に、`frontend/components/table` にも、データソースごとに特化したテーブルコンポーネント（`FundingRateDataTable.tsx` など）が存在します。これらは汎用的な `DataTable` コンポーネントをラップしており、冗長な構造になっています。

#### 解決案

- **ボタンの汎用化**: 特化ボタンを廃止し、`ApiButton` または `DataCollectionButton` のような、より汎用的なコンポーネントに統一します。APIエンドポイントや確認メッセージなどを `props` として渡すことで、コンポーネントの再利用性を高めます。
- **テーブルの統合**: `frontend/app/data/page.tsx` で使用されている `DataTableContainer.tsx` のように、タブで表示するテーブルを切り替えるコンポーネントにロジックを集約します。これにより、データソースごとのテーブルコンポーネントファイルを削除し、コンポーネント構造をシンプルにできます。
