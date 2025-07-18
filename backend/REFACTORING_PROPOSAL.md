# バックエンド リファクタリング提案書 Phase 2

## 概要

Phase 1 のリファクタリング完了後、コードベースの深層分析を実施した結果、さらなる改善の余地が発見されました。本提案書では、重複コードの統合、設定管理の統一、エラーハンドリングの一元化など、コードベースの品質向上とメンテナンス性の向上を目的とした第 2 フェーズのリファクタリング案を提示します。

---

## 発見された問題点

### 1. エラーハンドリングの重複

#### 重複している機能

- **タイムアウト処理**: 複数箇所で実装
- **ログ出力パターン**: 類似したログ形式
- **エラーレスポンス生成**: 同様の構造
- **安全な関数実行**: `safe_execute`系の重複 (`APIErrorHandler.handle_api_exception` と `MLErrorHandler.safe_execute` の類似)
- **エラーハンドリングクラスの重複**: `app/core/utils/api_utils.py` の `APIErrorHandler` と `app/core/utils/ml_error_handler.py` の `MLErrorHandler` が類似の責務を持つ。
- **`MLCommonErrorHandler` の不在**: 提案書で言及されている `MLCommonErrorHandler` は、現在のコードベースでは `app/core/services/ml/common/error_handler.py` に存在しない。これは `MLErrorHandler` に統合されたか、過去の遺物である可能性が高い。

#### 影響

- エラーハンドリングの一貫性が欠如
- 同じようなコードの重複によるメンテナンス負荷
- 新しいエラータイプ追加時の対応箇所の分散

### 2. ディレクトリ構造の改善点

#### 現状の問題

- **設定ファイルの配置**: 論理的でない分散 (`app/config/settings.py` と `app/config/market_config.py` に設定が分散している)
- **共通ユーティリティ**: 機能別の整理不足 (例: `MarketDataConfig` 内のバリデーションロジックが設定クラスに混在)
- **テストファイル**: 一部でインデントエラー

### 3. API エンドポイントの重複

#### 現状の問題

- `data_collection.py`, `funding_rates.py`, `open_interest.py` など、データソースごとに類似した API エンドポイント（収集、ステータス確認、リセット）が多数存在し、コードの重複とメンテナンス性の低下を招いています。

### 4. ビジネスロジックの分散

#### 現状の問題

- API ルーター内に、データ収集やバックテスト実行などのビジネスロジックが直接記述されている箇所があり、責務の分離が不十分です。

### 5. データベースアクセスの重複パターン

#### 現状の問題

- 各リポジトリクラス（`ohlcv_repository.py`など）に、`get_latest_timestamp` のような類似のメソッドが個別に実装されており、冗長性が生じています。

---

## 提案するリファクタリング

### 1. 統一設定管理システムの構築

#### 目標

設定ファイルを統一し、階層的で管理しやすい設定システムを構築する。

#### 実装案

```python
# app/config/unified_config.py
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List

# 各設定カテゴリのPydanticモデルを定義
class AppConfig(BaseSettings):
    app_name: str = Field(default="Trdinger Trading API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], alias="CORS_ORIGINS")

class DatabaseConfig(BaseSettings):
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="trdinger", alias="DB_NAME")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")

    @property
    def complete_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

class LoggingConfig(BaseSettings):
    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="LOG_FORMAT",
    )
    log_file: str = Field(default="market_data.log", alias="LOG_FILE")
    log_max_bytes: int = Field(default=10485760, alias="LOG_MAX_BYTES")
    log_backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT")

class MarketConfig(BaseSettings):
    # Settingsから移動
    market_data_sandbox: bool = Field(default=False, alias="MARKET_DATA_SANDBOX")
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")
    max_cache_size: int = Field(default=1000, alias="MAX_CACHE_SIZE")

    # MarketDataConfigから移動
    supported_exchanges: List[str] = Field(default=["bybit"], alias="SUPPORTED_EXCHANGES")
    supported_symbols: List[str] = Field(default=["BTC/USDT:USDT"], alias="SUPPORTED_SYMBOLS")
    supported_timeframes: List[str] = Field(default=["15m", "30m", "1h", "4h", "1d"], alias="SUPPORTED_TIMEFRAMES")
    default_exchange: str = Field(default="bybit", alias="DEFAULT_EXCHANGE")
    default_symbol: str = Field(default="BTC/USDT:USDT", alias="DEFAULT_SYMBOL")
    default_timeframe: str = Field(default="1h", alias="DEFAULT_TIMEFRAME")
    default_limit: int = Field(default=100, alias="DEFAULT_LIMIT")
    min_limit: int = Field(default=1, alias="MIN_LIMIT")
    max_limit: int = Field(default=1000, alias="MAX_LIMIT")
    bybit_config: dict = Field(default={
        "sandbox": False,
        "enableRateLimit": True,
        "timeout": 30000,
    }, alias="BYBIT_CONFIG")
    symbol_mapping: dict = Field(default={
        "BTCUSDT": "BTC/USDT:USDT",
        "BTC-USDT": "BTC/USDT:USDT",
        "BTC/USDT": "BTC/USDT:USDT",
        "BTC/USDT:USDT": "BTC/USDT:USDT",
        "BTCUSDT_PERP": "BTC/USDT:USDT",
    }, alias="SYMBOL_MAPPING")

class SecurityConfig(BaseSettings):
    secret_key: str = Field(default="your-secret-key-here", alias="SECRET_KEY")

class MLConfig(BaseSettings):
    # ML関連の設定をここに統合
    ga_fallback_symbol: str = Field(default="BTC/USDT", alias="GA_FALLBACK_SYMBOL")
    ga_fallback_timeframe: str = Field(default="1d", alias="GA_FALLBACK_TIMEFRAME")
    ga_fallback_start_date: str = Field(default="2024-01-01", alias="GA_FALLBACK_START_DATE")
    ga_fallback_end_date: str = Field(default="2024-04-09", alias="GA_FALLBACK_END_DATE")
    ga_fallback_initial_capital: float = Field(default=100000.0, alias="GA_FALLBACK_INITIAL_CAPITAL")
    ga_fallback_commission_rate: float = Field(default=0.001, alias="GA_FALLBACK_COMMISSION_RATE")
    # ml_config.pyの内容もここに統合される

class UnifiedConfig(BaseSettings):
    """
    アプリケーション全体の統一設定クラス
    """
    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    class Config:
        env_nested_delimiter = '__' # ネストされた環境変数をサポート (例: APP__NAME)
        extra = "ignore" # 未知のフィールドを無視

# 設定のシングルトンインスタンス
unified_settings = UnifiedConfig()
```

#### 移行計画

1.  **新しい統一設定クラスの作成**
2.  **既存設定の段階的移行**
3.  **環境変数マッピングの統一**
4.  **設定バリデーションの統合**

### 2. エラーハンドリングの統一

#### 目標

分散したエラーハンドリングを統一し、一貫性のあるエラー処理システムを構築する。

#### 実装案

```python
# app/core/utils/unified_error_handler.py
from fastapi import HTTPException
from typing import Any, Callable, Dict, Optional

class UnifiedErrorHandler:
    """
    統一エラーハンドリングクラス

    API関連エラーとML関連エラーを一元的に処理し、
    安全な関数実行機能を提供する。
    """

    @staticmethod
    def handle_api_error(
        error: Exception,
        context: str = "",
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        message: str = "予期しないエラーが発生しました",
    ) -> HTTPException:
        """
        API関連のエラーを処理し、HTTPExceptionを発生させる。
        APIErrorHandlerの各handle_*_errorメソッドを統合する。
        """
        # 既存のAPIErrorHandlerのロジックをここに統合
        # 例: バリデーションエラー、データベースエラー、外部APIエラー、一般的なエラー
        # ログ出力もここで行う
        detail_message = f"{context} - {message}: {error}"
        # logger.error(detail_message) # 適切なロガーをインポートして使用
        return HTTPException(
            status_code=status_code,
            detail={
                "success": False,
                "message": message,
                "error_code": error_code,
                "details": str(error),
            },
        )

    @staticmethod
    def handle_ml_error(
        error: Exception,
        context: str = "",
        default_return_value: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        ML関連のエラーを処理し、デフォルト値またはエラー情報を含む辞書を返す。
        MLErrorHandlerのhandle_data_error, handle_prediction_error, handle_model_error, handle_timeout_errorを統合する。
        """
        # 既存のMLErrorHandlerのロジックをここに統合
        # 例: データエラー、予測エラー、モデルエラー、タイムアウトエラー
        # ログ出力もここで行う
        # logger.error(f"MLエラー in {context}: {error}") # 適切なロガーをインポートして使用
        return {
            "success": False,
            "error": str(error),
            "context": context,
            "message": f"ML処理中にエラーが発生しました: {error}",
            "default_value_returned": default_return_value is not None,
        }

    @staticmethod
    def safe_execute(
        func: Callable[..., Any],
        default_value: Optional[Any] = None,
        error_message: str = "処理中にエラーが発生しました",
        log_level: str = "error",
        is_api_call: bool = False,
        api_status_code: int = 500,
        api_error_code: str = "INTERNAL_ERROR",
    ) -> Any:
        """
        関数を安全に実行し、例外を捕捉して適切に処理する。
        APIErrorHandler.handle_api_exception と MLErrorHandler.safe_execute の共通部分を統合する。
        """
        try:
            return func()
        except HTTPException as e:
            if is_api_call:
                raise e  # 既にHTTPExceptionの場合はそのまま再raise
            else:
                # MLコンテキストでHTTPExceptionが発生した場合の処理
                # logger.error(f"ML処理中にAPI例外が発生: {e.detail}")
                return default_value
        except Exception as e:
            if is_api_call:
                # API関連のエラーとして処理
                raise UnifiedErrorHandler.handle_api_error(
                    e,
                    context=error_message,
                    status_code=api_status_code,
                    error_code=api_error_code,
                    message=error_message,
                )
            else:
                # ML関連のエラーとして処理
                # log_func = getattr(logger, log_level, logger.error)
                # log_func(f"{error_message}: {e}")
                return UnifiedErrorHandler.handle_ml_error(
                    e, context=error_message, default_return_value=default_value
                )
```

#### 統合対象

- `APIErrorHandler` → `UnifiedErrorHandler.handle_api_error`
- `MLErrorHandler` → `UnifiedErrorHandler.handle_ml_error`
- `MLCommonErrorHandler` → `UnifiedErrorHandler.handle_ml_error`

### 3. 共通データベースユーティリティの作成

#### 目標

スクリプトファイルで重複しているデータベース操作を共通化する。

#### 実装案

```python
# app/core/utils/db_utils.py
class DatabaseUtils:
    @staticmethod
    def get_connection_with_retry() -> Session

    @staticmethod
    def validate_data_integrity(repository: BaseRepository) -> ValidationResult

    @staticmethod
    def get_statistics(repository: BaseRepository) -> Dict[str, Any]
```

### 4. ディレクトリ構造の最適化

#### 提案する新構造

```
app/
├── config/
│   ├── __init__.py
│   ├── unified_config.py    # 統一設定
│   └── environment.py       # 環境別設定
├── core/
│   ├── utils/
│   │   ├── unified_error_handler.py  # 統一エラーハンドラー
│   │   ├── db_utils.py              # DB共通ユーティリティ
│   │   └── common_validators.py     # 共通バリデーター
│   └── services/
└── api/
```

### 5. 汎用データ収集 API エンドポイントの導入

#### 目的

データソース名をパラメータとして受け取る汎用的なエンドポイントを導入し、API の数を削減し、コードの再利用性を高めます。

#### 実装案

```
# app/api/data.py (新規作成)
POST /api/data/collect/{source_name}
GET /api/data/status/{source_name}
DELETE /api/data/reset/{source_name}
```

#### 影響

フロントエンドの API 呼び出し箇所の修正が必要になりますが、バックエンドのメンテナンス性が大幅に向上します。

### 6. サービス層の導入によるビジネスロジックの集約

#### 目的

API ルーターからビジネスロジックをサービス層に分離し、コードの見通しとテスト容易性を向上させます。

#### 実装案

- `app/services/data_management_service.py` を新規作成し、データ収集・リセット・ステータス確認ロジックを集約します。
- 既存の `BacktestService` なども、より純粋なビジネスロジックに専念できるようにリファクタリングします。

### 7. リポジトリ層のさらなる共通化

#### 目的

`DatabaseQueryHelper` を活用し、定型的なクエリを `BaseRepository` に集約することで、コードの重複を削減します。

#### 実装案

- `BaseRepository` に `get_latest_timestamp`, `get_oldest_timestamp`, `get_record_count` などの汎用メソッドを実装します。
- 各具象リポジトリクラスは、これらの汎用メソッドを呼び出すように修正します。

---

## 実装計画

### 優先度

- **高**:
  1. 統一設定管理システム
  2. エラーハンドリング統一
- **中**:
  1. データベースユーティリティ統合
  2. ディレクトリ構造最適化
  3. API エンドポイントの汎用化
  4. サービス層の導入
- **低**:
  1. データベースリポジトリ層の共通化
  2. テストファイルの修正
  3. ドキュメント更新

### 次のステップ

1.  **Phase 2a の実装**: 統一設定管理とエラーハンドリング統合
2.  **Phase 2b の実装**: API エンドポイントの汎用化とサービス層の導入
3.  **Phase 2c の実装**: データベースユーティリティとリポジトリ層の共通化
4.  **動作確認**: 各段階での包括的なテストの実行
5.  **継続的改善**: 定期的なコードベース分析

---

## 期待される効果

### 短期的効果

- **設定管理の簡素化**: 一箇所での設定変更
- **エラーハンドリングの一貫性**: 統一されたエラー処理
- **コード重複の削減**: DRY 原則の徹底

### 長期的効果

- **メンテナンス性の向上**: 変更箇所の明確化
- **新機能開発の効率化**: 統一されたパターンの活用
- **品質の向上**: 一貫性のあるコードベース

---

## リスク評価

### リスクレベル

- **低リスク**:
  - **設定統合**: 段階的移行により影響を最小化
  - **エラーハンドリング統合**: 既存インターフェースの維持
- **中リスク**:
  - **ディレクトリ構造変更**: インポートパスの更新が必要
  - **API エンドポイントの汎用化**: フロントエンドの修正が必要

### 対策

- **段階的実装**: 小さな変更を積み重ね
- **後方互換性の維持**: 既存コードの動作保証
- **包括的テスト**: 各段階でのテスト実行

---

**提案日**: 2025 年 1 月 18 日  
**提案者**: Augment Agent  
**対象**: Phase 1 完了後のさらなる品質向上
