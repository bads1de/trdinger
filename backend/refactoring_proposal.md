# リファクタリング提案書

## 概要

本提案書は、`backend`ディレクトリ内のコードベースにおけるリファクタリングの機会を特定し、その改善策を提示することを目的としています。コードの可読性、保守性、テスト容易性、パフォーマンスの向上を目指します。

## 1. `app/api` ディレクトリ

API 層は、リクエストの受け付け、ビジネスロジックの呼び出し、レスポンスの整形に専念すべきです。

### 1.1. `app/api/dependencies.py`

- **問題点**: 依存性注入のファクトリ関数が、サービス（例: `AutoStrategyService`, `AutoMLFeatureGenerationService`）の初期化ロジックを直接含んでいます。サービスが複雑になったり、複数の依存関係を持つようになった場合に、このファイルが肥大化し、依存関係の管理が困難になる可能性があります。
- **提案**:
  - **DI コンテナの導入検討**: `FastAPI`の依存性注入は強力ですが、より複雑な依存関係の解決には`python-dependency-injector`のような DI コンテナの導入を検討します。これにより、依存関係の定義と解決を一元化し、テスト容易性を向上させます。
  - **サービス初期化の集約**: 各サービスの`__init__`メソッドに初期化ロジックを集約し、`dependencies.py`ではシンプルなインスタンス化を行うようにします。

### 1.2. `app/api/*.py` (各 API エンドポイントファイル)

- **問題点**: 各 API エンドポイントで`app.utils.unified_error_handler.UnifiedErrorHandler.safe_execute_async`を直接呼び出しています。これにより、API 層がエラーハンドリングの詳細に依存し、関心の分離に反しています。
- **提案**:
  - **グローバルエラーハンドリングの強化**: `app/main.py`に定義されているグローバル例外ハンドラを強化し、API 層でのエラーハンドリングの重複を排除します。カスタム例外を定義し、それらをグローバルハンドラで捕捉して適切な HTTP レスポンスを返すようにします。
  - **デコレータによるエラーハンドリング**: 必要に応じて、API エンドポイントに適用できるエラーハンドリング用のデコレータを作成し、コードの重複を削減します。

## 2. `app/services` ディレクトリ

サービス層はビジネスロジックをカプセル化し、外部システム（DB、外部 API など）との連携を抽象化すべきです。

### 2.1. サービス初期化時の DB リポジトリの直接初期化

- **問題点**: 以下のサービスが、自身の初期化時に DB リポジトリを直接初期化しています。
  - `app/services/auto_strategy/services/auto_strategy_service.py`
  - `app/services/auto_strategy/services/ml_orchestrator.py`
  - `app/services/backtest/backtest_service.py`
  - `app/services/data_collection/orchestration/*.py` (各オーケストレーションサービス)
- **提案**:
  - **依存性注入の徹底**: DB セッションやリポジトリは、サービスのコンストラクタを通じて依存性注入されるべきです。これにより、サービスのテストが容易になり、DB 実装の変更に対する柔軟性が向上します。
  - **`app/api/dependencies.py`の活用**: `app/api/dependencies.py`でリポジトリのインスタンスを生成し、それをサービスに注入するファクトリ関数を定義します。

### 2.2. `app/services/auto_strategy/services/indicator_service.py`

- **問題点**: `IndicatorCalculator`が`TechnicalIndicatorService`と`MLOrchestrator`を直接初期化しています。
- **提案**:
  - **依存性注入**: `IndicatorCalculator`のコンストラクタで`TechnicalIndicatorService`と`MLOrchestrator`のインスタンスを受け取るように変更します。

## 3. `app/utils` ディレクトリ

ユーティリティ層は、特定のビジネスロジックを持たない汎用的な機能を提供すべきです。

### 3.2. 例外クラスの重複定義

- **問題点**: `app/utils/unified_error_handler.py`と`app/services/ml/exceptions.py`の両方で、`MLDataError`、`MLValidationError`などの ML 関連の例外クラスが重複して定義されています。
- **提案**:
  - **例外クラス定義の一元化**: `app/services/ml/exceptions.py`に ML 関連のすべての例外クラスを定義し、`app/utils/unified_error_handler.py`はそれらの例外クラスを利用する形に変更します。これにより、例外の管理と拡張が容易になります。

## 4. `database` ディレクトリ

データベース層は、データ永続化のロジックに専念すべきです。

### 4.1. `database/connection.py`

- **問題点**: `SessionLocal`がグローバル変数として定義されており、テスト時にモック化が難しい場合があります。
- **提案**:
  - **DI 可能なセッション管理**: `get_db`関数を`FastAPI`の依存性注入システムで利用できるようにし、テスト時にはモックの DB セッションを注入できるようにします。

### 4.2. `database/models.py`

- **問題点**: `ToDictMixin`が各モデルに適用されており、モデルが自身のシリアライズロジックを持つことになっています。これはリポジトリ層でデータ変換を行うべきであり、モデルが持つべき責務ではありません。
- **提案**:
  - **リポジトリでのデータ変換**: 各リポジトリに、モデルオブジェクトを辞書や Pydantic モデルに変換するメソッド（例: `to_dict`, `to_pydantic_model`）を実装し、`ToDictMixin`を削除します。これにより、モデルは純粋なデータ構造として機能し、リポジトリがデータ変換の責務を負うようになります。

### 4.3. `database/repositories/*.py`

- **問題点**: 各リポジトリが`app.utils.data_conversion.DataSanitizer`（旧`DataValidator`）を直接インポートし、データ検証を行っています。リポジトリはデータの永続化に専念すべきであり、データ検証はサービス層や API 層で行われるべきです。
- **提案**:
  - **サービス層でのデータ検証**: リポジトリが受け取るデータは既に検証済みであることを前提とします。データ検証のロジックは、サービス層（例: `app/services/data_collection/orchestration`）に移動します。

## 5. 全体的な構造

### 5.1. 循環参照の解消

- **問題点**: 複数のファイル間で循環参照が発生している可能性があります（例: `app.services.auto_strategy.services.ml_orchestrator`と`app.services.ml.ml_training_service`）。これはコードの理解を難しくし、テストを複雑にします。
- **提案**:
  - **依存関係の整理**: モジュール間の依存関係を整理し、単方向の依存関係を確立します。必要に応じて、共通のインターフェースを定義したり、遅延インポートを活用したりして循環参照を解消します。

### 5.2. ログレベルの管理

- **問題点**: ログレベルが複数の場所で設定されている可能性があります（例: `app/main.py`の`setup_logging`、`app/services/auto_strategy/core/ga_engine.py`の`initialize_ga_engine`）。これにより、ログ設定の一貫性が失われ、デバッグが困難になる可能性があります。
- **提案**:
  - **ログ設定の一元化**: ログ設定を`app/config`ディレクトリに一元化し、アプリケーション全体で同じ設定が適用されるようにします。`logging.basicConfig`や`logging.config.dictConfig`を使用して、設定ファイルからログ設定を読み込むようにします。
