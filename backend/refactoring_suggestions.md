# バックエンドコードのリファクタリング案

このドキュメントでは、`backend`ディレクトリ内のコードにおける重複や冗長な箇所を指摘し、具体的なリファクタリング案を提案します。リファクタリングにより、コードの保守性、可読性、再利用性を向上させることを目的とします。

## 1. API 層 (`app/api/`) のリファクタリング

### 1.1. API レスポンス生成の共通化

**現状の問題点:**
各 API エンドポイントで、`api_response`や`error_response`のようなカスタム関数や、FastAPI の`JSONResponse`を直接使用しており、レスポンス形式に一貫性がありません。また、成功・失敗時のロジックが各所に散在しています。

**リファクタリング案:**
`app.utils.response.py` にあるユーティリティを拡張し、FastAPI のレスポンスを返す共通のラッパー関数を作成します。これにより、レスポンス形式を統一し、エラーハンドリングを簡素化できます。

**実装例 (`app/utils/response.py`):**

```python
from fastapi.responses import JSONResponse
from fastapi import status

def success_response(data: Any, message: str = "Success", status_code: int = status.HTTP_200_OK) -> JSONResponse:
    """成功時の共通レスポンスを生成"""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
    )

def client_error_response(message: str, status_code: int = status.HTTP_400_BAD_REQUEST, details: Any = None) -> JSONResponse:
    """クライアントエラー時の共通レスポンスを生成"""
    content = {
        "success": False,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }
    if details:
        content["details"] = details
    return JSONResponse(status_code=status_code, content=content)

def server_error_response(message: str = "Internal Server Error", details: Any = None) -> JSONResponse:
    """サーバーエラー時の共通レスポンスを生成"""
    # ... 同様に実装 ...
```

### 1.2. エンドポイント内のロジック共通化

**現状の問題点:**
多くのエンドポイントで `async def _internal_logic(): ...` のような内部関数を定義し、`UnifiedErrorHandler.safe_execute_async` でラップするパターンが繰り返されています。これは定型的なコードの重複です。

**リファクタリング案:**
デコレーターを作成して、このパターンを抽象化します。

**実装例（デコレーター）:**

```python
from functools import wraps
from app.utils.unified_error_handler import UnifiedErrorHandler

def api_endpoint_wrapper(message: str = "処理中にエラーが発生しました"):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def execute_logic():
                return await func(*args, **kwargs)
            return await UnifiedErrorHandler.safe_execute_async(execute_logic, message=message)
        return wrapper
    return decorator

# 使用例
@router.get("/status")
@api_endpoint_wrapper(message="データステータスの取得に失敗しました")
async def get_data_status(db: Session = Depends(get_db)):
    service = DataManagementOrchestrationService()
    return await service.get_data_status(db_session=db)
```

## 2. サービス層 (`app/services/`) のリファクタリング

### 2.1. データ収集オーケストレーションサービスの統合

**現状の問題点:**
`data_collection/orchestration` 内に、データソースごとにオーケストレーションサービスが複数存在します (`DataCollectionOrchestrationService`, `FearGreedOrchestrationService` など)。責務は似ていますが、クラスが分かれているため見通しが悪くなっています。

**リファクタリング案:**
単一の `DataOrchestrationService` に統合し、各データソースの収集・管理ロジックをメソッドとして提供します。

**実装例:**

```python
class DataOrchestrationService:
    def __init__(self, db: Session):
        self.ohlcv_repo = OHLCVRepository(db)
        self.fr_repo = FundingRateRepository(db)
        # ... 他のリポジトリも初期化 ...

    async def collect_ohlcv_data(self, ...):
        # ... OHLCV収集ロジック ...

    async def collect_funding_rate_data(self, ...):
        # ... FR収集ロジック ...
```

### 2.2. ML モデルラッパーの評価ロジック共通化

**現状の問題点:**
`app/services/ml/models` 内の各モデルラッパー（`lightgbm.py`, `xgboost.py`など）の `_train_model_impl` メソッド内で、評価指標を計算するコードが重複しています。

**リファクタリング案:**
`BaseEnsemble` または `BaseMLTrainer` に評価ロジックを計算する共通メソッドを実装し、各モデルラッパーから呼び出すようにします。`EnhancedMetricsCalculator` を活用して、評価指標の計算を一元化します。

**実装例 (`BaseMLTrainer`):**

```python
from app.services.ml.evaluation.enhanced_metrics import EnhancedMetricsCalculator, MetricsConfig

class BaseMLTrainer:
    # ...
    def _evaluate_model(self, y_true, y_pred, y_pred_proba) -> Dict[str, Any]:
        config = MetricsConfig(...)
        calculator = EnhancedMetricsCalculator(config)
        return calculator.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)

    def _train_model_impl(self, ...):
        # ... 学習ロジック ...
        test_metrics = self._evaluate_model(y_test, y_pred_test, y_pred_proba_test)
        # ...
```

**実装例:**

```python
from sqlalchemy.inspection import inspect

class SerializationMixin:
    def to_dict(self):
        """オブジェクトを辞書に変換する共通メソッド"""
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

class OHLCVData(Base, SerializationMixin):
    # ... カラム定義 ...
    # to_dict() は不要になる
```

## まとめ

上記のリファクタリング案を適用することで、以下の効果が期待できます。

- **保守性の向上:** コードの重複が削減され、ロジックの変更が容易になります。
- **可読性の向上:** 各クラス・モジュールの責務が明確になり、コードが理解しやすくなります。
- **再利用性の向上:** 共通化されたコンポーネント（レスポンス生成、エラーハンドリング、評価指標計算など）は、他の部分でも再利用可能です。
- **一貫性の確保:** API レスポンスやエラー処理の形式が統一され、フロントエンドとの連携がスムーズになります。
