# バックエンドコードのリファクタリング提案

このドキュメントは、`backend/` ディレクトリ以下の Python コードベースにおける重複、共通ロジック、および改善の余地がある構造を特定し、リファクタリングの提案をまとめたものです。

## 1. 共通のエラーハンドリングと API レスポンス形式の抽象化

### 問題点

API エンドポイントの多くで、`APIErrorHandler.handle_api_exception` を使用して例外を処理し、`APIResponseHelper.api_response` を使用してレスポンスを生成しています。これは適切なパターンですが、各エンドポイントで以下のような冗長なラッパー関数が定義されています。

```python
async def _func():
    # ... 実際のロジック ...
    return {"success": True, "result": result} # または APIResponseHelper.api_response(...)

return await APIErrorHandler.handle_api_exception(_func)
```

このパターンは、コードの重複と可読性の低下を招いています。

### 改善提案

共通のラッパーロジックを抽象化するデコレータを作成し、API エンドポイント関数に適用することで、コードの重複を減らし、可読性を向上させます。

**ファイル**: `backend/app/core/utils/api_utils.py`

**変更内容**:
`APIErrorHandler.handle_api_exception` のロジックをラップするデコレータを導入します。

```python
# backend/app/core/utils/api_utils.py に追加
from functools import wraps

def api_endpoint_wrapper(func):
    """
    APIエンドポイント関数をラップし、共通のエラーハンドリングとレスポンス形式を適用するデコレータ。
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        async def _inner_func():
            return await func(*args, **kwargs)
        return await APIErrorHandler.handle_api_exception(_inner_func)
    return wrapper

# 使用例 (backend/app/api/backtest.py など)
# @router.post("/run", response_model=BacktestResponse)
# @api_endpoint_wrapper
# async def run_backtest(request: BacktestRequest, db: Session = Depends(get_db)):
#     backtest_service = BacktestService()
#     config = _create_base_config(request)
#     result = backtest_service.run_backtest(config)
#     saved_result = _save_backtest_result(result, db)
#     # デコレータがAPIResponseHelper.api_responseを適用するため、ここでは生の辞書を返す
#     return {"success": True, "result": saved_result}
```

これにより、各 API エンドポイントのコードがより簡潔になります。

## 2. データ収集 API における共通タスク管理ロジックの抽出

### 問題点

`backend/app/api/data_collection.py` 内の `collect_bulk_historical_data` と `collect_all_data_bulk` 関数は、シンボルと時間軸の組み合わせをループし、データが存在するかどうかを確認し、存在しない場合にバックグラウンドタスクを追加するという、非常に類似したロジックを含んでいます。

- **ファイルパスと行番号**:
  - `backend/app/api/data_collection.py`: 181-293 (`collect_bulk_historical_data`)
  - `backend/app/api/data_collection.py`: 444-579 (`collect_all_data_bulk`)

### 改善提案

この共通のタスク管理ロジックを、独立したヘルパー関数またはクラスに抽出します。このヘルパー関数は、シンボルと時間軸のリスト、および各組み合わせに対して実行する具体的な収集関数（OHLCV のみ、または全データ）を受け取るようにします。

**ファイル**: `backend/app/api/data_collection.py` (または `backend/app/core/services/data_collection_manager.py` のような新しいファイル)

**変更内容**:
共通のタスク処理ロジックを `_process_collection_tasks` のような関数に抽出します。

```python
# backend/app/api/data_collection.py に追加 (例)
async def _process_collection_tasks(
    background_tasks: BackgroundTasks,
    db: Session,
    symbols: List[str],
    timeframes: List[str],
    collection_func: Callable, # _collect_historical_background または _collect_all_data_background
    logger_prefix: str
) -> Dict[str, Any]:
    """
    データ収集タスクの共通処理ロジック。
    """
    total_combinations = len(symbols) * len(timeframes)
    started_at = datetime.now(timezone.utc).isoformat()

    repository = OHLCVRepository(db) # OHLCVRepositoryは例。必要に応じて汎用化。

    tasks_to_execute = []
    skipped_tasks = []
    failed_tasks = []

    logger.info(f"{logger_prefix}開始: {total_combinations}組み合わせを確認")

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                normalized_symbol = MarketDataConfig.normalize_symbol(symbol)
                data_exists = repository.get_data_count(normalized_symbol, timeframe) > 0

                if data_exists:
                    skipped_tasks.append({
                        "symbol": normalized_symbol,
                        "original_symbol": symbol,
                        "timeframe": timeframe,
                        "reason": "data_exists",
                    })
                    logger.debug(f"スキップ: {normalized_symbol} {timeframe} - データが既に存在")
                else:
                    background_tasks.add_task(
                        collection_func,
                        normalized_symbol,
                        timeframe,
                        db,
                    )
                    tasks_to_execute.append({
                        "symbol": normalized_symbol,
                        "original_symbol": symbol,
                        "timeframe": timeframe,
                    })
                    logger.info(f"タスク追加: {normalized_symbol} {timeframe}")

            except Exception as task_error:
                logger.warning(f"タスク処理エラー {symbol} {timeframe}: {task_error}")
                failed_tasks.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "error": str(task_error),
                })
                continue

    actual_tasks = len(tasks_to_execute)
    skipped_count = len(skipped_tasks)
    failed_count = len(failed_tasks)

    logger.info(f"{logger_prefix}タスク分析完了:")
    logger.info(f"  - 総組み合わせ数: {total_combinations}")
    logger.info(f"  - 実行タスク数: {actual_tasks}")
    logger.info(f"  - スキップ数: {skipped_count} (既存データ)")
    logger.info(f"  - 失敗数: {failed_count}")

    return {
        "total_combinations": total_combinations,
        "actual_tasks": actual_tasks,
        "skipped_tasks": skipped_count,
        "failed_tasks": failed_count,
        "started_at": started_at,
        "symbols": symbols,
        "timeframes": timeframes,
        "task_details": {
            "executing": tasks_to_execute,
            "skipped": skipped_tasks,
            "failed": failed_tasks,
        },
    }

# collect_bulk_historical_data と collect_all_data_bulk からこの関数を呼び出す
# 例:
# @router.post("/bulk-historical")
# async def collect_bulk_historical_data(...):
#     # ... データベース初期化確認 ...
#     symbols = ["BTC/USDT:USDT"] # または MarketDataConfig.SUPPORTED_SYMBOLS
#     timeframes = ["15m", "30m", "1h", "4h", "1d"] # または MarketDataConfig.SUPPORTED_TIMEFRAMES
#     result_data = await _process_collection_tasks(
#         background_tasks, db, symbols, timeframes, _collect_historical_background, "一括データ収集"
#     )
#     return APIResponseHelper.api_response(
#         success=True,
#         message=f"一括データ収集を開始しました（{result_data['actual_tasks']}タスク実行、{result_data['skipped_tasks']}タスクスキップ）",
#         status="started",
#         data=result_data,
#     )
```

## 3. リポジトリ層の共通処理の汎用化

### 問題点

各リポジトリ（例: `BacktestResultRepository`, `FundingRateRepository`, `OpenInterestRepository`, `OHLCVRepository`）で、データの取得、フィルタリング、ソート、ページネーションのロジックが個別に実装されている場合、重複が発生し、一貫性のない実装になる可能性があります。

### 改善提案

`backend/database/repositories/base_repository.py` に、共通のデータ取得、フィルタリング、ソート、ページネーションを処理する汎用的な基底クラスを導入します。各具体的なリポジトリクラスは、この基底クラスを継承し、それぞれのエンティティに特化したクエリやビジネスロジックを追加するようにします。

**ファイル**: `backend/database/repositories/base_repository.py`

**変更内容**:
`BaseRepository` クラスに、`get_all`, `get_by_id`, `count` などの汎用的なメソッドを実装します。

```python
# backend/database/repositories/base_repository.py (例)
from sqlalchemy.orm import Session
from sqlalchemy import desc

class BaseRepository:
    def __init__(self, db: Session, model):
        self.db = db
        self.model = model

    def get_all(self, limit: int = 100, offset: int = 0, order_by: str = "created_at", **filters):
        query = self.db.query(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.filter(getattr(self.model, key) == value)

        if hasattr(self.model, order_by):
            query = query.order_by(desc(getattr(self.model, order_by)))

        return query.offset(offset).limit(limit).all()

    def get_by_id(self, item_id: int):
        return self.db.query(self.model).filter(self.model.id == item_id).first()

    def count(self, **filters):
        query = self.db.query(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.filter(getattr(self.model, key) == value)
        return query.count()

    # ... その他の共通メソッド (save, delete など)

# 各リポジトリはこれを継承
# class BacktestResultRepository(BaseRepository):
#     def __init__(self, db: Session):
#         super().__init__(db, BacktestResult)
#     # ... BacktestResult に特化したメソッド ...
```

## 4. 日付文字列の変換ロジックの統一

### 問題点

`start_date` や `end_date` の ISO 形式文字列を `datetime` オブジェクトに変換するロジック (`datetime.fromisoformat(date_str.replace("Z", "+00:00"))`) が複数の API ファイルで繰り返されています。

- **ファイルパスと行番号**:
  - `backend/app/api/funding_rates.py`: 62-65
  - `backend/app/api/open_interest.py`: 60-63
  - `backend/app/api/market_data.py`: 75-78 (既に `DateTimeHelper.parse_iso_datetime` を使用)

### 改善提案

`backend/app/core/utils/api_utils.py` に既に存在する `DateTimeHelper.parse_iso_datetime` を、他のファイルでも利用するように統一します。

**ファイル**:

- `backend/app/api/funding_rates.py`
- `backend/app/api/open_interest.py`

**変更内容**:
`datetime.fromisoformat(date_str.replace("Z", "+00:00"))` の代わりに `DateTimeHelper.parse_iso_datetime(date_str)` を使用します。

```python
# backend/app/api/funding_rates.py (変更例)
# from app.core.utils.api_utils import DateTimeHelper # 既にインポートされているはず

# ...
# if start_date:
#     start_time = DateTimeHelper.parse_iso_datetime(start_date)
# if end_date:
#     end_time = DateTimeHelper.parse_iso_datetime(end_date)
# ...

# backend/app/api/open_interest.py (変更例)
# from app.core.utils.api_utils import DateTimeHelper # 既にインポートされているはず

# ...
# if start_date:
#     start_time = DateTimeHelper.parse_iso_datetime(start_date)
# if end_date:
#     end_time = DateTimeHelper.parse_iso_datetime(end_date)
# ...
```

## 5. API エンドポイントにおけるデータ取得ロジックの共通化

### 問題点

`backend/app/api/funding_rates.py`, `backend/app/api/open_interest.py`, `backend/app/api/market_data.py` の各エンドポイントでは、データベースからデータを取得し、API レスポンス形式に変換するロジックが非常に似ています。特に、日付パラメータの変換、シンボルの正規化、リポジトリからのデータ取得、そして結果の API レスポンス形式への変換（`APIResponseHelper.api_response`の使用）が共通しています。

- **ファイルパスと行番号**:
  - `backend/app/api/funding_rates.py`: 23-112 (`get_funding_rates`)
  - `backend/app/api/open_interest.py`: 27-102 (`get_open_interest_data`)
  - `backend/app/api/market_data.py`: 29-107 (`get_ohlcv_data`)

### 改善提案

これらの共通ロジックを抽象化し、汎用的なデータ取得ヘルパー関数またはクラスを作成します。このヘルパーは、リポジトリ、サービス、データ変換ロジックを引数として受け取り、共通の処理フローを実行します。

**ファイル**: `backend/app/core/utils/api_data_fetcher.py` (新規作成)

**変更内容**:
共通のデータ取得ロジックをカプセル化する関数を定義します。

```python
# backend/app/core/utils/api_data_fetcher.py (新規作成)
import logging
from datetime import datetime
from typing import Optional, Callable, List, Any, Dict

from sqlalchemy.orm import Session

from app.core.utils.api_utils import APIResponseHelper, DateTimeHelper
from app.core.services.base_bybit_service import BaseBybitService # シンボル正規化のため

logger = logging.getLogger(__name__)

async def fetch_and_respond_data(
    symbol: str,
    db: Session,
    repository_class: Any, # 例: FundingRateRepository
    service_class: Any, # 例: BybitFundingRateService (シンボル正規化用)
    data_converter_class: Any, # 例: FundingRateDataConverter
    get_data_method_name: str, # 例: "get_funding_rate_data"
    response_key: str, # 例: "funding_rates"
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None,
    limit: Optional[int] = None,
    log_prefix: str = "データ",
    additional_filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    APIエンドポイント向けの汎用データ取得・レスポンス生成関数。
    """
    logger.info(f"{log_prefix}取得リクエスト: symbol={symbol}, limit={limit}")

    repository = repository_class(db)

    start_time = DateTimeHelper.parse_iso_datetime(start_date_str) if start_date_str else None
    end_time = DateTimeHelper.parse_iso_datetime(end_date_str) if end_date_str else None

    # シンボルの正規化
    service = service_class()
    normalized_symbol = service.normalize_symbol(symbol)

    # リポジトリからデータを取得
    get_data_method = getattr(repository, get_data_method_name)

    # OHLCVの場合の特殊処理 (get_latest_ohlcv_data)
    if get_data_method_name == "get_latest_ohlcv_data":
        records = get_data_method(
            symbol=normalized_symbol,
            timeframe=additional_filters.get("timeframe"),
            limit=limit,
        )
    else:
        # その他のデータ取得メソッド
        method_args = {
            "symbol": normalized_symbol,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit,
        }
        if additional_filters:
            method_args.update(additional_filters)
        records = get_data_method(**method_args)

    # レスポンス形式に変換
    api_data = data_converter_class.db_to_api_format(records)

    logger.info(f"{log_prefix}取得成功: {len(api_data)}件")

    return APIResponseHelper.api_response(
        success=True,
        data={
            "symbol": normalized_symbol,
            "count": len(api_data),
            response_key: api_data,
            **({"timeframe": additional_filters["timeframe"]} if "timeframe" in additional_filters else {})
        },
        message=f"{len(api_data)}件の{log_prefix}データを取得しました",
    )

# 各APIエンドポイントでの使用例:
# @router.get("/funding-rates")
# async def get_funding_rates(...):
#     async def _get_funding_rates_data():
#         return await fetch_and_respond_data(
#             symbol=symbol,
#             db=db,
#             repository_class=FundingRateRepository,
#             service_class=BybitFundingRateService,
#             data_converter_class=FundingRateDataConverter,
#             get_data_method_name="get_funding_rate_data",
#             response_key="funding_rates",
#             start_date_str=start_date,
#             end_date_str=end_date,
#             limit=limit,
#             log_prefix="ファンディングレート",
#         )
#     return await APIErrorHandler.handle_api_exception(_get_funding_rates_data, message="ファンディングレートデータ取得エラー")
```

## 6. API エンドポイントにおけるデータ収集ロジックの共通化

### 問題点

`backend/app/api/funding_rates.py`, `backend/app/api/open_interest.py`, `backend/app/api/data_collection.py` の各エンドポイントでは、外部 API からデータを収集し、データベースに保存するロジックが重複しています。特に、`ensure_db_initialized()`の呼び出し、サービスとリポジトリのインスタンス化、`fetch_and_save_xxx_data`の呼び出し、そして結果の API レスポンス形式への変換が共通しています。一括収集 (`bulk-collect`) のロジックも非常に似ています。

- **ファイルパスと行番号**:
  - `backend/app/api/funding_rates.py`: 115-173 (`collect_funding_rate_data`), 223-304 (`bulk_collect_funding_rates`)
  - `backend/app/api/open_interest.py`: 104-160 (`collect_open_interest_data`), 200-280 (`bulk_collect_open_interest`)
  - `backend/app/api/data_collection.py`: 24-95 (`collect_historical_data`), 164-298 (`collect_bulk_historical_data`), 444-580 (`collect_all_data_bulk`)

### 改善提案

共通のデータ収集ロジックを抽象化し、汎用的なデータ収集ヘルパー関数またはクラスを作成します。このヘルパーは、サービス、リポジトリ、収集メソッドを引数として受け取り、共通の処理フローを実行します。一括収集についても同様に共通化します。

**ファイル**: `backend/app/core/utils/api_data_collector.py` (新規作成)

**変更内容**:
共通のデータ収集ロジックをカプセル化する関数を定義します。

```python
# backend/app/core/utils/api_data_collector.py (新規作成)
import logging
from typing import Optional, Any, Dict, List
from sqlalchemy.orm import Session
from fastapi import HTTPException, BackgroundTasks
import asyncio

from database.connection import ensure_db_initialized
from app.core.utils.api_utils import APIResponseHelper
from app.config.market_config import MarketDataConfig # シンボル正規化用

logger = logging.getLogger(__name__)

async def collect_and_respond_data(
    symbol: str,
    db: Session,
    service_class: Any, # 例: BybitFundingRateService
    repository_class: Any, # 例: FundingRateRepository
    fetch_and_save_method_name: str, # 例: "fetch_and_save_funding_rate_data"
    log_prefix: str = "データ",
    limit: Optional[int] = None,
    fetch_all: bool = False,
    additional_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    APIエンドポイント向けの汎用データ収集・レスポンス生成関数。
    """
    logger.info(f"{log_prefix}収集開始: symbol={symbol}, fetch_all={fetch_all}")

    if not ensure_db_initialized():
        logger.error("データベースの初期化に失敗しました")
        raise HTTPException(status_code=500, detail="データベースの初期化に失敗しました")

    service = service_class()
    repository = repository_class(db)

    fetch_and_save_method = getattr(service, fetch_and_save_method_name)

    method_args = {
        "symbol": symbol,
        "repository": repository,
        "fetch_all": fetch_all,
    }
    if limit is not None:
        method_args["limit"] = limit
    if additional_args:
        method_args.update(additional_args)

    result = await fetch_and_save_method(**method_args)

    logger.info(f"{log_prefix}収集完了: {result}")

    return APIResponseHelper.api_response(
        data=result,
        message=f"{result['saved_count']}件の{log_prefix}データを保存しました",
        success=True,
    )

async def bulk_collect_and_respond_data(
    db: Session,
    service_class: Any,
    repository_class: Any,
    fetch_and_save_method_name: str,
    symbols_to_collect: List[str],
    log_prefix: str = "データ",
    additional_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    APIエンドポイント向けの汎用一括データ収集・レスポンス生成関数。
    """
    logger.info(f"{log_prefix}一括収集開始: {len(symbols_to_collect)}シンボル")

    if not ensure_db_initialized():
        logger.error("データベースの初期化に失敗しました")
        raise HTTPException(status_code=500, detail="データベースの初期化に失敗しました")

    service = service_class()
    repository = repository_class(db)

    results = []
    total_saved = 0
    successful_symbols = 0
    failed_symbols = []

    for symbol in symbols_to_collect:
        try:
            method_args = {
                "symbol": symbol,
                "repository": repository,
                "fetch_all": True,
            }
            if additional_args:
                method_args.update(additional_args)

            result = await getattr(service, fetch_and_save_method_name)(**method_args)
            results.append(result)
            total_saved += result["saved_count"]
            successful_symbols += 1
            logger.info(f"✅ {symbol}: {result['saved_count']}件保存")
            await asyncio.sleep(0.1) # APIレート制限考慮

        except Exception as e:
            logger.error(f"❌ {symbol} 収集エラー: {e}")
            failed_symbols.append({"symbol": symbol, "error": str(e)})

    logger.info(f"{log_prefix}一括収集完了: {successful_symbols}/{len(symbols_to_collect)}成功")

    return APIResponseHelper.api_response(
        data={
            "total_symbols": len(symbols_to_collect),
            "successful_symbols": successful_symbols,
            "failed_symbols": len(failed_symbols),
            "total_saved_records": total_saved,
            "results": results,
            "failures": failed_symbols,
        },
        success=True,
        message=f"{successful_symbols}/{len(symbols_to_collect)}シンボルで合計{total_saved}件の{log_prefix}データを保存しました",
    )

# 各APIエンドポイントでの使用例:
# @router.post("/funding-rates/collect")
# async def collect_funding_rate_data(...):
#     return await APIErrorHandler.handle_api_exception(
#         lambda: collect_and_respond_data(
#             symbol=symbol,
#             db=db,
#             service_class=BybitFundingRateService,
#             repository_class=FundingRateRepository,
#             fetch_and_save_method_name="fetch_and_save_funding_rate_data",
#             log_prefix="ファンディングレート",
#             limit=limit,
#             fetch_all=fetch_all,
#         ),
#         message="ファンディングレートデータ収集エラー",
#     )

# @router.post("/funding-rates/bulk-collect")
# async def bulk_collect_funding_rates(...):
#     symbols = ["BTC/USDT:USDT"] # または MarketDataConfig.SUPPORTED_SYMBOLS
#     return await APIErrorHandler.handle_api_exception(
#         lambda: bulk_collect_and_respond_data(
#             db=db,
#             service_class=BybitFundingRateService,
#             repository_class=FundingRateRepository,
#             fetch_and_save_method_name="fetch_and_save_funding_rate_data",
#             symbols_to_collect=symbols,
#             log_prefix="ファンディングレート",
#         ),
#         message="ファンディングレート一括収集エラー",
#     )
```

## 7. リポジトリ層における CRUD 操作の共通化の深化

### 問題点

`backend/database/repositories/funding_rate_repository.py`, `backend/database/repositories/open_interest_repository.py`, `backend/database/repositories/ohlcv_repository.py` の各リポジトリは `BaseRepository` を継承していますが、`insert_xxx_data`、`get_xxx_data`、`get_latest_xxx_timestamp`、`clear_all_xxx_data`、`clear_xxx_data_by_symbol` などのメソッドで、`BaseRepository` の汎用メソッドを呼び出すだけの薄いラッパーが多く存在します。これにより、各リポジトリクラスが冗長になっています。

- **ファイルパスと行番号**:
  - `backend/database/repositories/funding_rate_repository.py`: 24-194
  - `backend/database/repositories/open_interest_repository.py`: 23-156
  - `backend/database/repositories/ohlcv_repository.py`: 24-361

### 改善提案

`BaseRepository` をさらに強化し、一般的なデータ操作メソッドを直接提供するようにします。各具体的なリポジトリは、`BaseRepository` のメソッドを直接使用するか、必要に応じてオーバーライドまたは拡張する形にします。これにより、各リポジトリのコード量を削減し、よりクリーンな設計を実現します。

**ファイル**: `backend/database/repositories/base_repository.py`

**変更内容**:
`BaseRepository` に、`insert_records`, `get_records`, `get_latest_timestamp_by_filter`, `get_record_count_by_filter`, `clear_all`, `clear_by_filter` などの汎用メソッドを追加または改善します。

```python
# backend/database/repositories/base_repository.py (改善例)
from typing import List, Optional, Type, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
import logging

from app.core.utils.database_utils import DatabaseInsertHelper, DatabaseQueryHelper

logger = logging.getLogger(__name__)

class BaseRepository:
    """リポジトリの基底クラス"""

    def __init__(self, db: Session, model_class: Type[Any]):
        self.db = db
        self.model_class = model_class

    def insert_records(self, records: List[Dict[str, Any]], conflict_columns: List[str]) -> int:
        """
        レコードを一括挿入（重複処理付き）
        """
        if not records:
            logger.warning(f"挿入する{self.model_class.__name__}データがありません")
            return 0
        try:
            inserted_count = self.bulk_insert_with_conflict_handling(records, conflict_columns)
            logger.info(f"{self.model_class.__name__}データを {inserted_count} 件挿入しました")
            return inserted_count
        except Exception as e:
            logger.error(f"{self.model_class.__name__}データ挿入エラー: {e}")
            raise

    def get_records(
        self,
        filters: Optional[Dict[str, Any]] = None,
        time_range_column: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        order_by_column: Optional[str] = None,
        order_asc: bool = True,
        limit: Optional[int] = None,
    ) -> List[Any]:
        """
        フィルター条件に基づいてレコードを取得
        """
        try:
            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=self.model_class,
                filters=filters,
                time_range_column=time_range_column,
                start_time=start_time,
                end_time=end_time,
                order_by_column=order_by_column,
                order_asc=order_asc,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"{self.model_class.__name__}データ取得エラー: {e}")
            raise

    def get_latest_timestamp_by_filter(self, timestamp_column: str, filters: Optional[Dict[str, Any]] = None) -> Optional[datetime]:
        """
        フィルター条件に基づいて最新タイムスタンプを取得
        """
        return self.get_latest_timestamp(timestamp_column, filters)

    def get_record_count_by_filter(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        フィルター条件に基づいてレコード数を取得
        """
        return self.get_record_count(filters)

    def clear_all(self) -> int:
        """
        全てのレコードを削除
        """
        try:
            deleted_count = self._delete_all_records()
            logger.info(f"全ての{self.model_class.__name__}データを削除しました: {deleted_count}件")
            return deleted_count
        except Exception as e:
            self._handle_delete_error(e, f"{self.model_class.__name__}データ全削除")

    def clear_by_filter(self, filter_column: str, filter_value: Any) -> int:
        """
        指定されたカラムと値に基づいてレコードを削除
        """
        try:
            deleted_count = self._delete_records_by_filter(filter_column, filter_value)
            logger.info(f"'{filter_column}={filter_value}' の{self.model_class.__name__}データを削除しました: {deleted_count}件")
            return deleted_count
        except Exception as e:
            self._handle_delete_error(e, f"{self.model_class.__name__}データ削除", filter_column=filter_column, filter_value=filter_value)

    # 既存の bulk_insert_with_conflict_handling, get_latest_timestamp, get_oldest_timestamp, get_record_count, get_date_range, get_available_symbols, _delete_all_records, _delete_records_by_filter, _handle_delete_error はそのまま残すか、必要に応じてプライベートメソッドとして再定義する。
```

**各リポジトリクラスの変更例**:

```python
# backend/database/repositories/funding_rate_repository.py (変更例)
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import pandas as pd
import logging

from .base_repository import BaseRepository
from database.models import FundingRateData

logger = logging.getLogger(__name__)

class FundingRateRepository(BaseRepository):
    def __init__(self, db: Session):
        super().__init__(db, FundingRateData)

    def insert_funding_rate_data(self, funding_rate_records: List[dict]) -> int:
        return self.insert_records(funding_rate_records, ["symbol", "funding_timestamp"])

    def get_funding_rate_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[FundingRateData]:
        return self.get_records(
            filters={"symbol": symbol},
            time_range_column="funding_timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="funding_timestamp",
            order_asc=True,
            limit=limit,
        )

    def get_latest_funding_timestamp(self, symbol: str) -> Optional[datetime]:
        return self.get_latest_timestamp_by_filter("funding_timestamp", {"symbol": symbol})

    def get_funding_rate_count(self, symbol: str) -> int:
        return self.get_record_count_by_filter({"symbol": symbol})

    def clear_all_funding_rate_data(self) -> int:
        return self.clear_all()

    def clear_funding_rate_data_by_symbol(self, symbol: str) -> int:
        return self.clear_by_filter("symbol", symbol)

    # get_funding_rate_dataframe はそのまま残す
```

## 8. データ変換ロジックの共通化の深化

### 問題点

`backend/app/core/utils/data_converter.py` 内の `OHLCVDataConverter`, `FundingRateDataConverter`, `OpenInterestDataConverter` はそれぞれ `ccxt_to_db_format` と `db_to_api_format` メソッドを持っています。これらのメソッドは、`datetime`オブジェクトの変換（ミリ秒タイムスタンプから`datetime`、ISO 文字列から`datetime`など）や`float`へのキャストなど、共通のロジックを多く含んでいます。

- **ファイルパスと行番号**:
  - `backend/app/core/utils/data_converter.py`: 12-81 (`OHLCVDataConverter`)
  - `backend/app/core/utils/data_converter.py`: 84-178 (`FundingRateDataConverter`)
  - `backend/app/core/utils/data_converter.py`: 181-271 (`OpenInterestDataConverter`)

### 改善提案

共通の変換ロジックを抽象化し、汎用的な基底データコンバータークラスを作成します。特に、タイムスタンプの変換ロジックは共通のヘルパーメソッドとして抽出できます。

**ファイル**: `backend/app/core/utils/data_converter.py`

**変更内容**:
`BaseDataConverter` クラスを導入し、共通のタイムスタンプ変換メソッドなどを定義します。各コンバータークラスはこれを継承します。

```python
# backend/app/core/utils/data_converter.py (改善例)
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class BaseDataConverter:
    """データ変換の基底クラス"""

    @staticmethod
    def _convert_timestamp_to_datetime(timestamp: Any) -> Optional[datetime]:
        """
        様々な形式のタイムスタンプをUTCのdatetimeオブジェクトに変換するヘルパー。
        """
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"無効なISO形式タイムスタンプ: {timestamp}")
                return None
        elif isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            except (ValueError, TypeError):
                logger.warning(f"無効なUNIXタイムスタンプ: {timestamp}")
                return None
        elif isinstance(timestamp, datetime):
            return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        return None

    @staticmethod
    def _convert_datetime_to_timestamp_ms(dt: datetime) -> int:
        """
        datetimeオブジェクトをミリ秒のUNIXタイムスタンプに変換するヘルパー。
        """
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _safe_float_conversion(value: Any, default: float = 0.0) -> float:
        """
        値を安全にfloatに変換するヘルパー。変換できない場合はデフォルト値を返す。
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

class OHLCVDataConverter(BaseDataConverter):
    """OHLCV データ変換の共通ヘルパークラス"""

    @staticmethod
    def ccxt_to_db_format(
        ohlcv_data: List[List], symbol: str, timeframe: str
    ) -> List[Dict[str, Any]]:
        db_records = []
        for candle in ohlcv_data:
            timestamp_ms, open_price, high, low, close, volume = candle
            timestamp = OHLCVDataConverter._convert_timestamp_to_datetime(timestamp_ms)
            if timestamp is None:
                logger.warning(f"OHLCVデータ変換スキップ: 無効なタイムスタンプ {timestamp_ms}")
                continue
            db_record = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "open": OHLCVDataConverter._safe_float_conversion(open_price),
                "high": OHLCVDataConverter._safe_float_conversion(high),
                "low": OHLCVDataConverter._safe_float_conversion(low),
                "close": OHLCVDataConverter._safe_float_conversion(close),
                "volume": OHLCVDataConverter._safe_float_conversion(volume),
            }
            db_records.append(db_record)
        return db_records

    @staticmethod
    def db_to_api_format(ohlcv_records: List[Any]) -> List[List]:
        api_data = []
        for record in ohlcv_records:
            api_data.append(
                [
                    OHLCVDataConverter._convert_datetime_to_timestamp_ms(record.timestamp),
                    record.open,
                    record.high,
                    record.low,
                    record.close,
                    record.volume,
                ]
            )
        return api_data

class FundingRateDataConverter(BaseDataConverter):
    """ファンディングレートデータ変換の共通ヘルパークラス"""

    @staticmethod
    def ccxt_to_db_format(
        funding_rate_data: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        db_records = []
        for rate_data in funding_rate_data:
            funding_timestamp = FundingRateDataConverter._convert_timestamp_to_datetime(rate_data.get("datetime"))
            if funding_timestamp is None:
                logger.warning(f"Funding Rateデータ変換スキップ: 無効なタイムスタンプ {rate_data.get('datetime')}")
                continue
            db_record = {
                "symbol": symbol,
                "funding_rate": FundingRateDataConverter._safe_float_conversion(rate_data.get("fundingRate", 0)),
                "funding_timestamp": funding_timestamp,
                "timestamp": datetime.now(timezone.utc),
                "next_funding_timestamp": FundingRateDataConverter._convert_timestamp_to_datetime(rate_data.get("nextFundingDatetime")),
                "mark_price": FundingRateDataConverter._safe_float_conversion(rate_data.get("markPrice")),
                "index_price": FundingRateDataConverter._safe_float_conversion(rate_data.get("indexPrice")),
            }
            db_records.append(db_record)
        return db_records

    @staticmethod
    def db_to_api_format(funding_rate_records: List[Any]) -> List[Dict[str, Any]]:
        api_data = []
        for record in funding_rate_records:
            api_data.append(
                {
                    "symbol": record.symbol,
                    "funding_rate": record.funding_rate,
                    "funding_timestamp": record.funding_timestamp.isoformat(),
                    "timestamp": record.timestamp.isoformat(),
                    "next_funding_timestamp": (
                        record.next_funding_timestamp.isoformat()
                        if record.next_funding_timestamp
                        else None
                    ),
                    "mark_price": record.mark_price,
                    "index_price": record.index_price,
                }
            )
        return api_data

class OpenInterestDataConverter(BaseDataConverter):
    """オープンインタレストデータ変換の共通ヘルパークラス"""

    @staticmethod
    def ccxt_to_db_format(
        open_interest_data: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        db_records = []
        logger.info(f"オープンインタレストデータ変換開始: {len(open_interest_data)}件")
        for oi_data in open_interest_data:
            data_timestamp = OpenInterestDataConverter._convert_timestamp_to_datetime(oi_data.get("timestamp"))
            if data_timestamp is None:
                logger.warning(f"Open Interestデータ変換スキップ: 無効なタイムスタンプ {oi_data.get('timestamp')}")
                continue

            open_interest_value = oi_data.get("openInterestValue")
            if open_interest_value is None and "info" in oi_data:
                info_data = oi_data["info"]
                if "openInterest" in info_data:
                    open_interest_value = OpenInterestDataConverter._safe_float_conversion(info_data["openInterest"])

            if open_interest_value is None:
                logger.warning(f"オープンインタレスト値が取得できませんでした: {oi_data}")
                continue

            db_record = {
                "symbol": symbol,
                "open_interest_value": open_interest_value,
                "data_timestamp": data_timestamp,
                "timestamp": datetime.now(timezone.utc),
            }
            db_records.append(db_record)
        return db_records

    @staticmethod
    def db_to_api_format(open_interest_records: List[Any]) -> List[Dict[str, Any]]:
        api_data = []
        for record in open_interest_records:
            api_data.append(
                {
                    "symbol": record.symbol,
                    "open_interest": record.open_interest_value, # 修正: open_interest_value を使用
                    "data_timestamp": record.data_timestamp.isoformat(),
                    "timestamp": record.timestamp.isoformat(),
                }
            )
        return api_data

# DataValidator はそのまま残す
```

## 9. `BaseBybitService`における`_fetch_paginated_data`の汎用性向上

### 問題点

`backend/app/core/services/base_bybit_service.py` の `_fetch_paginated_data` メソッドはページネーション処理を共通化していますが、`latest_existing_timestamp`による差分更新ロジックが特定のデータ型（`timestamp`キーを持つ辞書）に依存しています。また、`page_data[-1]["timestamp"]` のような直接アクセスは、`page_data`が空の場合にエラーを引き起こす可能性があります。

- **ファイルパスと行番号**:
  - `backend/app/core/services/base_bybit_service.py`: 145-247 (`_fetch_paginated_data`)

### 改善提案

`_fetch_paginated_data` をより汎用的にし、データ内のタイムスタンプキーを引数で指定できるようにします。また、`page_data`が空の場合のハンドリングを強化し、より堅牢なロジックにします。

**ファイル**: `backend/app/core/services/base_bybit_service.py`

**変更内容**:
`_fetch_paginated_data` メソッドを修正し、`timestamp_key` 引数を追加し、空の`page_data`のハンドリングを改善します。

```python
# backend/app/core/services/base_bybit_service.py (改善例)
# ... (既存のインポートとクラス定義)

class BaseBybitService(ABC):
    # ... (既存のメソッド)

    async def _fetch_paginated_data(
        self,
        fetch_func: Callable,
        symbol: str,
        page_limit: int = 200,
        max_pages: int = 50,
        latest_existing_timestamp: Optional[int] = None,
        timestamp_key: str = "timestamp", # 新しい引数
        **fetch_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        ページネーション処理で全期間データを取得
        """
        logger.info(f"全期間データ取得開始: {symbol}")

        all_data = []
        page_count = 0
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

        while page_count < max_pages:
            page_count += 1

            try:
                params = fetch_kwargs.get("params", {})
                params["end"] = end_time

                page_data = await self._handle_ccxt_errors(
                    f"ページデータ取得 (page={page_count})",
                    fetch_func,
                    symbol,
                    None,  # since
                    page_limit,
                    params,
                )

                if not page_data:
                    logger.info(f"ページ {page_count}: データなし。取得終了")
                    break

                # 取得したデータの最後のタイムスタンプを次のリクエストのend_timeに設定
                # page_dataが空でないことを確認済み
                end_time = page_data[-1][timestamp_key] - 1 # 最後のデータは次のリクエストのendになるため、重複を避けるために-1

                logger.info(
                    f"ページ {page_count}: {len(page_data)}件取得 "
                    f"(累計: {len(all_data) + len(page_data)}件)"
                )

                # 重複チェック（タイムスタンプベース）
                existing_timestamps = {item[timestamp_key] for item in all_data}
                new_items = [
                    item
                    for item in page_data
                    if item[timestamp_key] not in existing_timestamps
                ]

                # 差分更新: 既存データより古いデータのみ追加
                if latest_existing_timestamp:
                    new_items = [
                        item
                        for item in new_items
                        if item[timestamp_key] < latest_existing_timestamp
                    ]

                    if not new_items:
                        logger.info(
                            f"ページ {page_count}: 既存データに到達。差分更新完了"
                        )
                        break

                all_data.extend(new_items)
                logger.info(
                    f"ページ {page_count}: 新規データ {len(new_items)}件追加 "
                    f"(累計: {len(all_data)}件)"
                )

                if len(page_data) < page_limit: # ページリミットより少ない場合は最後のページ
                    logger.info(f"ページ {page_count}: 最後のページに到達")
                    break

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"ページ {page_count} 取得エラー: {e}")
                continue

        all_data.sort(key=lambda x: x[timestamp_key])
        logger.info(f"全期間データ取得完了: {len(all_data)}件 ({page_count}ページ)")
        return all_data

    # ... (残りのメソッド)
```

## 10. `data_collection.py`におけるバックグラウンドタスクの共通化

### 問題点

`backend/app/api/data_collection.py` 内の `_collect_all_data_background` と `_collect_historical_background` は、それぞれ OHLCV、Funding Rate、Open Interest の収集ロジックを含んでいますが、これらは`HistoricalDataService`、`BybitFundingRateService`、`BybitOpenInterestService`のメソッドを呼び出しているだけです。これらのバックグラウンドタスクは、より抽象化された形で共通化できる可能性があります。特に、エラーハンドリングと DB セッションのクローズ処理が各関数で繰り返されています。

- **ファイルパスと行番号**:
  - `backend/app/api/data_collection.py`: 582-673 (`_collect_all_data_background`)
  - `backend/app/api/data_collection.py`: 676-697 (`_collect_historical_background`)

### 改善提案

バックグラウンドタスクの実行とエラーハンドリング、DB セッションの管理を共通化するヘルパー関数を作成します。これにより、各収集タスクのロジックが簡潔になり、DRY 原則に準拠します。

**ファイル**: `backend/app/api/data_collection.py`

**変更内容**:
`_run_collection_task_in_background` のような共通ヘルパー関数を定義し、各バックグラウンド収集関数からそれを呼び出すようにします。

```python
# backend/app/api/data_collection.py (改善例)
# ... (既存のインポート)

async def _run_collection_task_in_background(
    task_func: Callable, symbol: str, timeframe: str, db: Session
):
    """
    バックグラウンド収集タスクの共通実行ロジックとエラーハンドリング。
    """
    try:
        result = await task_func(symbol, timeframe, db)
        if result["success"]:
            logger.info(
                f"バックグラウンド収集完了: {symbol} {timeframe} - {result['saved_count']}件保存"
            )
        else:
            logger.error(
                f"バックグラウンド収集失敗: {symbol} {timeframe} - {result.get('message')}",
                exc_info=True,
            )
    except Exception:
        logger.error(f"バックグラウンド収集エラー: {symbol} {timeframe}", exc_info=True)
    finally:
        db.close()

async def _collect_all_data_background(symbol: str, timeframe: str, db: Session):
    """バックグラウンドでの全データ収集（OHLCV・FR・OI・TI）"""
    async def _task_logic(sym: str, tf: str, session: Session):
        logger.info(f"全データ収集開始: {sym} {tf}")

        # OHLCVデータ収集
        historical_service = HistoricalDataService()
        ohlcv_repository = OHLCVRepository(session)
        ohlcv_result = await historical_service.collect_historical_data(sym, tf, ohlcv_repository)
        if not ohlcv_result["success"]:
            return ohlcv_result # 失敗した場合はここで終了

        # Funding Rate収集
        from app.core.services.funding_rate_service import BybitFundingRateService
        from database.repositories.funding_rate_repository import FundingRateRepository
        funding_service = BybitFundingRateService()
        funding_repository = FundingRateRepository(session)
        funding_result = await funding_service.fetch_and_save_funding_rate_data(
            symbol=sym, repository=funding_repository, fetch_all=True
        )
        if not funding_result["success"]:
            return funding_result # 失敗した場合はここで終了

        # Open Interest収集
        from app.core.services.open_interest_service import BybitOpenInterestService
        from database.repositories.open_interest_repository import OpenInterestRepository
        oi_service = BybitOpenInterestService()
        oi_repository = OpenInterestRepository(session)
        oi_result = await oi_service.fetch_and_save_open_interest_data(
            symbol=sym, repository=oi_repository, fetch_all=True, interval=tf
        )
        if not oi_result["success"]:
            return oi_result # 失敗した場合はここで終了

        return {"success": True, "saved_count": ohlcv_result["saved_count"] + funding_result["saved_count"] + oi_result["saved_count"]}

    await _run_collection_task_in_background(_task_logic, symbol, timeframe, db)

async def _collect_historical_background(symbol: str, timeframe: str, db: Session):
    """バックグラウンドでの履歴データ収集"""
    async def _task_logic(sym: str, tf: str, session: Session):
        service = HistoricalDataService()
        repository = OHLCVRepository(session)
        return await service.collect_historical_data(sym, tf, repository)

    await _run_collection_task_in_background(_task_logic, symbol, timeframe, db)
```
