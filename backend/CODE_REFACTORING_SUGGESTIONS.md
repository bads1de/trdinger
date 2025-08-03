# バックエンドコード改善提案書

## 1. はじめに

このドキュメントは、`backend` ディレクトリ全体のコードベースを分析し、コードの品質、保守性、再利用性を向上させるための具体的な改善案を提案するものです。主に、コードの重複、冗長なロジック、設計原則の観点から分析を行いました。

## 2. 全体的な評価

現在のバックエンドは、API、サービス、リポジトリの各層に責務が分離されており、多くの箇所で設計原則が適用されています。特に、`BaseRepository` や `BybitService` のような基底クラスの導入は、コードの共通化に貢献しています。

しかし、プロジェクトの成長に伴い、いくつかの領域で重複や冗長性が生じています。本提案は、これらの点を解消し、よりクリーンで効率的なコードベースを実現することを目的とします。

---

## 3. 改善提案一覧

---

### 提案 2: データ収集サービスのロジック共通化

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

### 提案 3: リポジトリ層における `BaseRepository` の活用徹底

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

### 提案 4: ML モデルラッパーの評価ロジック共通化

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

### 提案 5: ML モデルのデフォルトハイパーパラメータ共通化

- **現状の課題**:
  `app/services/ml/models/` 配下の各ラッパークラス (`randomforest_wrapper.py`, `ridge_wrapper.py` など) では、`default_params` という辞書を個別に持っています。同様のキーが繰り返し定義されており、ハイパーパラメータを調整する際に複数ファイルを修正する必要が生じます。

- **具体的な修正案**:
  1. `app/services/ml/configs/model_default_params.py` を新規作成し、アルゴリズム名をキーとしたデフォルトパラメータのマッピングを保持します。
  2. 各モデルラッパーでは `from ...configs.model_default_params import MODEL_DEFAULT_PARAMS` をインポートし、`self.default_params = MODEL_DEFAULT_PARAMS["randomforest"]` のように取得します。
  3. ハイパーパラメータのオーバーライドが必要な場合は、`training_params` 引数で上書きする方針を統一します。

  ```python
  # app/services/ml/configs/model_default_params.py
  MODEL_DEFAULT_PARAMS = {
      "randomforest": {
          "n_estimators": 200,
          "max_depth": 12,
          "min_samples_split": 5,
          "min_samples_leaf": 2,
          "max_features": "sqrt",
          "class_weight": "balanced",
          "random_state": 42,
          "n_jobs": -1,
      },
      "ridge": {
          "alpha": 1.0,
          "random_state": 42,
      },
      # ... ほかのモデル
  }
  ```

- **期待される効果**:
  - ハイパーパラメータ定義の一元化により、重複を排除し保守性が向上。
  - チューニングや A/B テスト時に一箇所の変更で全モデルへ反映可能。
  - モデル追加時に既存コードをコピー＆ペーストする必要がなくなり、実装ミスを削減。

---

### 提案 6: 例外ハンドリングとロギングの共通デコレータ化

- **現状の課題**:
  多くのサービス・リポジトリ・モデルクラスで、`try / except Exception as e` のブロックと `logger.error()` を毎回記述しています。同様のエラーメッセージ組み立てロジックが散在し、コードが冗長になっています。

- **具体的な修正案**:
  1. `app/utils/decorators.py` に `@log_exceptions` デコレータを実装し、関数名・スタックトレースを自動で出力する汎用ロギング処理をまとめます。
  2. 既存の `try / except` ブロックを削減し、メソッドや関数にデコレータを付与して例外ハンドリングを一元化します。

  ```python
  # app/utils/decorators.py
  import functools
  import logging
  import traceback

  logger = logging.getLogger(__name__)

  def log_exceptions(default_return=None):
      """例外を捕捉してログ出力し、任意の値を返すデコレータ"""
      def decorator(func):
          @functools.wraps(func)
          def wrapper(*args, **kwargs):
              try:
                  return func(*args, **kwargs)
              except Exception as exc:
                  logger.error(
                      "%s failed: %s\n%s",
                      func.__qualname__,
                      exc,
                      traceback.format_exc(),
                  )
                  return default_return
          return wrapper
      return decorator
  ```

- **期待される効果**:
  - 重複する `try / except` と `logger.error` 記述を排除し、コードベースを約 5〜10% 削減。
  - エラーログのフォーマットを統一し、運用時のモニタリング効率を向上。
  - 新しいメソッドへの適用もデコレータを付けるだけで済み、開発速度が向上。

---

### 提案 7: ロギング設定の集中管理

- **現状の課題**:
  各テストファイルや一部のスクリプト内で `logging.basicConfig()` を個別に呼び出しています。ログフォーマットが統一されておらず、同じ設定を繰り返し記述しているため冗長です。

- **具体的な修正案**:
  1. `app/logging_config.py` を作成し、`dictConfig` ベースでハンドラ・フォーマッタ・ロガーを定義します。
  2. `main.py` 起動時に `logging.config.dictConfig(LOGGING_CONFIG)` を一度だけ実行。以降のモジュールは `import logging` し `logger = logging.getLogger(__name__)` とするだけで統一フォーマットが適用されます。
  3. テストでは `conftest.py` で同じ設定を読み込むことで、テスト出力も同フォーマットに統一します。

  ```python
  # app/logging_config.py
  LOGGING_CONFIG = {
      "version": 1,
      "disable_existing_loggers": False,
      "formatters": {
          "standard": {
              "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
          }
      },
      "handlers": {
          "console": {
              "class": "logging.StreamHandler",
              "formatter": "standard",
          }
      },
      "root": {
          "handlers": ["console"],
          "level": "INFO",
      },
  }
  ```

- **期待される効果**:
  - ログフォーマット・ログレベルがプロジェクト全体で統一され、解析・モニタリングが容易に。
  - 重複する `basicConfig()` 呼び出しを削減し、設定変更時の影響範囲を最小化。

---

## 4. 結論

本提案で示した改善を実施することで、バックエンドのコードベースはより堅牢で、保守しやすく、拡張性の高いものになります。特に、ロジックの共通化と責務の明確化は、将来的な機能追加や仕様変更に迅速に対応するための強固な基盤となります。
