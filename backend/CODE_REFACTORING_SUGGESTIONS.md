# バックエンドコード改善提案書

## 1. はじめに

このドキュメントは、`backend` ディレクトリ全体のコードベースを分析し、コードの品質、保守性、再利用性を向上させるための具体的な改善案を提案するものです。主に、コードの重複、冗長なロジック、設計原則の観点から分析を行いました。

## 2. 全体的な評価

現在のバックエンドは、API、サービス、リポジトリの各層に責務が分離されており、多くの箇所で設計原則が適用されています。特に、`BaseRepository` や `BybitService` のような基底クラスの導入は、コードの共通化に貢献しています。

しかし、プロジェクトの成長に伴い、いくつかの領域で重複や冗長性が生じています。本提案は、これらの点を解消し、よりクリーンで効率的なコードベースを実現することを目的とします。

---

## 3. 改善提案一覧

### 提案1: API層における依存性注入の簡素化

- **現状の課題**:
  `app/api/dependencies.py` には、単純にクラスをインスタンス化して返すだけのファクトリ関数（例: `get_data_collection_orchestration_service`）が多数存在します。これは冗長であり、コードの可読性をわずかに低下させます。

- **具体的な修正案**:
  FastAPI の `Depends` は、直接クラスを指定することでインスタンスを生成できます。単純なファクシミリ関数を削除し、API ルーターで直接クラスを依存性として注入します。

  **修正前 (`app/api/data_collection.py`)**:
  ```python
  from app.api.dependencies import get_data_collection_orchestration_service

  @router.post("/historical")
  async def collect_historical_data(
      # ...
      orchestration_service: DataCollectionOrchestrationService = Depends(
          get_data_collection_orchestration_service
      ),
  ) -> Dict:
      # ...
  ```

  **修正後 (`app/api/data_collection.py`)**:
  ```python
  from app.services.data_collection.orchestration.data_collection_orchestration_service import DataCollectionOrchestrationService

  @router.post("/historical")
  async def collect_historical_data(
      # ...
      orchestration_service: DataCollectionOrchestrationService = Depends(DataCollectionOrchestrationService),
  ) -> Dict:
      # ...
  ```
  これにより、`dependencies.py` 内の多くの単純なゲッター関数を削除できます。

- **期待される効果**:
  - `dependencies.py` ファイルがスリムになり、管理が容易になる。
  - 依存関係がより明確になり、コードの可読性が向上する。

---

### 提案2: データ収集サービスのロジック共通化

- **現状の課題**:
  `funding_rate_service.py` と `open_interest_service.py` には、データの取得、ページネーション、保存に関する類似したロジックが個別に実装されています。これはコードの重複であり、将来的な仕様変更時のメンテナンスコストを増大させます。

- **具体的な修正案**:
  `BybitService` 基底クラスに、データ収集と保存のための汎用的なメソッド（例: `_fetch_and_save_data`, `_fetch_incremental_data`）を実装します。各サブクラス（`BybitFundingRateService` など）は、`data_config.py` で定義されたデータソース固有の設定をこの汎用メソッドに渡すだけにします。

  **修正案 (`app/services/data_collection/bybit/bybit_service.py`)**:
  ```python
  # BybitServiceに汎用メソッドを追加
  async def fetch_and_save_data(
      self,
      symbol: str,
      config: DataServiceConfig, # データソース固有の設定
      limit: Optional[int] = None,
      fetch_all: bool = False,
      **kwargs,
  ) -> dict:
      # ... 共通のデータ取得・保存ロジック ...
      fetch_history_method = getattr(self.exchange, config.fetch_history_method_name)
      # ...
  ```

  **修正案 (`app/services/data_collection/bybit/funding_rate_service.py`)**:
  ```python
  # サブクラスは設定を渡すだけ
  class BybitFundingRateService(BybitService):
      def __init__(self):
          super().__init__()
          self.config = get_funding_rate_config()

      async def fetch_and_save_funding_rate_data(self, ...):
          return await self.fetch_and_save_data(
              symbol=symbol,
              config=self.config,
              # ...
          )
  ```

- **期待される効果**:
  - データ収集ロジックの重複を排除し、コード量を削減。
  - 新しいデータソースの追加が容易になる。
  - エラーハンドリングとページネーションロジックを一元管理できる。

---

### 提案3: リポジトリ層における `BaseRepository` の活用徹底

- **現状の課題**:
  `backtest_result_repository.py` など一部のリポジトリでは、`BaseRepository` で提供されている汎用的なクエリヘルパー (`get_filtered_records` など) を利用せず、独自のクエリを実装している箇所があります。

- **具体的な修正案**:
  すべてのリポジトリクラスで、`BaseRepository` の汎用メソッドを最大限に活用するようにリファクタリングします。

  **修正前 (`database/repositories/backtest_result_repository.py`)**:
  ```python
  def get_backtest_results(self, ...):
      query = self.db.query(BacktestResult)
      if symbol:
          query = query.filter(BacktestResult.symbol == symbol)
      # ...
      return query.all()
  ```

  **修正後 (`database/repositories/backtest_result_repository.py`)**:
  ```python
  def get_backtest_results(self, ...):
      filters = {}
      if symbol:
          filters["symbol"] = symbol
      if strategy_name:
          filters["strategy_name"] = strategy_name
      
      return self.get_filtered_data(
          filters=filters,
          order_by_column="created_at",
          order_asc=False,
          limit=limit,
          offset=offset # BaseRepositoryにoffsetの追加が必要
      )
  ```

- **期待される効果**:
  - 各リポジトリのコードが簡潔になり、可読性と保守性が向上する。
  - クエリロジックが一元化され、バグの発生リスクが低減する。

---

### 提案4: MLモデルラッパーの評価ロジック共通化

- **現状の課題**:
  `app/services/ml/models/` 以下の各モデルラッパークラス（`adaboost_wrapper.py`, `randomforest_wrapper.py` など）の `_train_model_impl` メソッド内で、`accuracy_score`, `f1_score` などの評価指標計算ロジックが重複して実装されています。

- **具体的な修正案**:
  `app/utils/metrics_calculator.py` にある `calculate_detailed_metrics` 関数を、`BaseMLTrainer` から呼び出すように変更します。各モデルラッパーは、学習と予測値の生成に専念し、評価処理は基底クラスに委譲します。

  **修正案 (`app/services/ml/base_ml_trainer.py`)**:
  ```python
  from app.utils.metrics_calculator import calculate_detailed_metrics

  class BaseMLTrainer:
      # ...
      def _evaluate_model(self, y_true, y_pred, y_proba):
          # ...
          return calculate_detailed_metrics(y_true, y_pred, y_proba)
      
      def _train_model_impl(self, ...):
          # ... (学習処理)
          y_pred = self.model.predict(X_test)
          y_proba = self.model.predict_proba(X_test)
          
          # 評価は共通メソッドを呼び出す
          metrics = self._evaluate_model(y_test, y_pred, y_proba)
          # ...
          return metrics
  ```

- **期待される効果**:
  - 評価指標の計算ロジックが一元化され、コードの重複が排除される。
  - 新しい評価指標の追加や変更が容易になる。
  - 各モデルラッパーの責務が明確になる。

---

## 4. 結論

本提案で示した改善を実施することで、バックエンドのコードベースはより堅牢で、保守しやすく、拡張性の高いものになります。特に、ロジックの共通化と責務の明確化は、将来的な機能追加や仕様変更に迅速に対応するための強固な基盤となります。
