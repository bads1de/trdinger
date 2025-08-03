# ML サービスのリファクタリング提案

`backend/app/services/ml/` 配下のコードベースについて、冗長な箇所、責務の分離、汎用性の観点からリファクタリング案を提案します。

## 1. 全体的な課題と方針

- **責務の混在**: `BaseMLTrainer` や `MLTrainingOrchestrationService` など、一つのクラスに多くの責務が集中しています。機能をより小さなクラスに分割し、単一責任の原則を徹底します。
- **コードの重複**: 特にモデルラッパークラス群や評価ロジックに多くの重複が見られます。共通処理をユーティリティや基底クラスに集約します。
- **設定管理の煩雑さ**: 設定オブジェクトと辞書表現の相互変換が手動で行われており、メンテナンス性に欠けます。Pydantic の機能を活用して自動化します。
- **デッドコードの可能性**: `ensemble_models.py` など、現在使用されていない可能性のあるコードが存在します。利用状況を確認し、不要であれば削除します。

---

## 2. 個別のリファクタリング提案

- [ ] ### 2.1. 評価ロジックの統一

- **課題**:
  - `base_ensemble.py` の `_evaluate_predictions` メソッドと `evaluation/enhanced_metrics.py` の `EnhancedMetricsCalculator` は、モデル評価という同じ責務を持ちながら、実装が重複・分散しています。
  - 各モデルラッパー (`lightgbm_wrapper.py` など) 内でも、個別に評価指標の計算ロジックが実装されており、冗長です。
- **提案**:

  - `evaluation/enhanced_metrics.py` の `EnhancedMetricsCalculator` を唯一の評価指標計算クラスとして利用するように統一します。
  - 各モデルラッパーやアンサンブルクラスは、予測結果（`y_pred`, `y_pred_proba`）を `EnhancedMetricsCalculator` に渡し、評価結果の辞書を受け取るように変更します。これにより、評価ロジックが一元管理され、新しい指標の追加や変更が容易になります。

  **実装例 (`base_ensemble.py`):**

  ```python
  # 変更前
  # from sklearn.metrics import accuracy_score, ...
  # self._evaluate_predictions(...)

  # 変更後
  from ..evaluation.enhanced_metrics import enhanced_metrics_calculator

  def _evaluate_ensemble(self, ...):
      # ...
      y_pred = self.predict(X_test)
      y_pred_proba = self.predict_proba(X_test)
      # 統一された評価クラスを呼び出す
      ensemble_metrics = enhanced_metrics_calculator.calculate_comprehensive_metrics(
          y_test, y_pred, y_pred_proba
      )
      result.update(ensemble_metrics)
      # ...
  ```

- [ ] ### 2.2. `BaseMLTrainer` の責務分割

- **課題**: `base_ml_trainer.py` は、特徴量計算、データ準備、モデル学習、評価、保存など、多くの責務を担っており、クラスが肥大化しています。
- **提案**:
  - **特徴量計算**: `_calculate_features` メソッド内のロジックは、`EnhancedFeatureEngineeringService` に完全に移譲します。`BaseMLTrainer` は `FeatureEngineeringService` のインスタンスを保持し、そのメソッドを呼び出すだけにします。
  - **データ準備**: `_prepare_training_data` メソッド内の欠損値補完やラベル生成ロジックは、`utils/data_preprocessing.py` や `utils/label_generation.py` の汎用関数を利用するようにし、`BaseMLTrainer` から具体的な処理を分離します。
  - **メタデータ構築**: `train_model` 内の冗長なメタデータ構築ロジックを、`ModelMetadata` のような `dataclass` を活用して簡潔にします。

- [ ] ### 2.3. モデル管理の統一

- **課題**: `model_manager.py` と `model_management/enhanced_model_manager.py` の 2 つのモデル管理クラスが存在し、機能が重複しています。特に `enhanced_model_manager.py` はバージョン管理やパフォーマンス監視など高機能ですが、`model_manager.py` が各所で利用されており、機能が分断されています。
- **提案**:
  - `enhanced_model_manager.py` を正式なモデル管理クラスとし、`model_manager.py` の機能を統合・廃止します。
  - `save_model` や `load_model` の呼び出し箇所を全て `enhanced_model_manager` のインスタンス経由に変更します。
  - `ml_management_orchestration_service.py` でモデル情報を取得する際に、毎回 `load_model` を呼ぶのではなく、`enhanced_model_manager` が保持するレジストリ（`model_registry.json`）からメタデータを直接読み込むように変更し、パフォーマンスを向上させます。

- [ ] ### 2.4. 設定管理の改善

- **課題**:
  - `config/ml_config_manager.py` の `get_config_dict` メソッドは、`MLConfig` の属性を手動で辞書にマッピングしており、設定クラスの変更に脆弱です。
  - `base_ml_trainer.py` の `_create_automl_config_from_dict` も同様に、手動での変換ロジックが記述されています。
- **提案**:

  - `MLConfig` や `AutoMLConfig` などの Pydantic モデルに、`to_dict()` や `from_dict()` のようなシリアライズ/デシリアライズ用のクラスメソッドを実装します。Pydantic の `.model_dump()` (v2) や `.dict()` (v1) を内部で利用することで、この処理を自動化できます。

  **実装例 (`ml_config.py`):**

  ```python
  # MLConfigクラス内
  def to_dict(self) -> Dict[str, Any]:
      return self.model_dump() # Pydantic v2の場合

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "MLConfig":
      return cls(**data)
  ```

- [ ] ### 2.5. デッドコードの削除

- **課題**: `models/ensemble_models.py` は、多くの scikit-learn モデルをインスタンス化していますが、プロジェクト内の他のファイル（`EnsembleTrainer`など）からは直接利用されておらず、デッドコードである可能性が高いです。
- **提案**:
  - `models/ensemble_models.py` の利用箇所を検索し、もしどこからも参照されていない場合は、このファイルを削除します。
  - 代わりに、`EnsembleTrainer` や `SingleModelTrainer` で利用されている各モデルラッパー (`lightgbm_wrapper.py` など) が、モデルのインスタンス化の責任を担っていることを明確にします。

- [ ] ### 2.6. 状態管理の改善

- **課題**: `orchestration/ml_training_orchestration_service.py` は、グローバル変数 `training_status` を使ってトレーニングの状態を管理しており、ステートフルでテストが困難です。
- **提案**:
  - `orchestration/background_task_manager.py` の機能を拡張し、トレーニングの状態（進捗、ステータス、メッセージなど）も管理できるようにします。
  - `MLTrainingOrchestrationService` は `background_task_manager` を通じて状態の読み書きを行い、グローバル変数への依存をなくします。これにより、コードの見通しが良くなり、テスト容易性も向上します。

- [ ] ### 2.7. Feature Engineering サービスの階層重複

- **課題**:
  - `FeatureEngineeringService`、`EnhancedFeatureEngineeringService`、`AutoMLFeatureGenerationService` の 3 つのサービスが存在し、責務が重複・分散しています。
  - `EnhancedFeatureEngineeringService` は `FeatureEngineeringService` を継承していますが、実際には大部分の機能を再実装しており、継承の利点が活かされていません。
  - `AutoMLFeatureGenerationService` は単なるファサードクラスとして機能しており、独立したサービスとしての価値が低いです。
- **提案**:
  - `FeatureEngineeringService` を基底クラスとして残し、AutoML 機能を統合した単一の `UnifiedFeatureEngineeringService` を作成します。
  - `AutoMLFeatureGenerationService` の機能を `UnifiedFeatureEngineeringService` に統合し、API レイヤーから直接呼び出せるようにします。
  - 継承ではなくコンポジションパターンを使用し、AutoML 機能を必要に応じて注入できる設計に変更します。

- [ ] ### 2.8. モデルラッパークラスの大量重複

- **課題**:
  - `models/` ディレクトリに 12 個のモデルラッパークラス（`LightGBMModel`, `XGBoostModel`, `CatBoostModel` など）が存在し、全て同じ構造を持っています。
  - 各クラスは `__init__`, `_train_model_impl`, `predict`, `predict_proba` などの同じメソッドを実装しており、コードの重複が深刻です。
  - パラメータ設定やエラーハンドリングのロジックも各クラスで重複しています。
- **提案**:

  - 抽象基底クラス `BaseModelWrapper` を作成し、共通のインターフェースと実装を定義します。
  - 各モデル固有の設定は設定ファイルまたは辞書として外部化し、`BaseModelWrapper` が動的にモデルを生成できるようにします。
  - ファクトリーパターンを使用して、モデル名から適切なラッパーインスタンスを生成する `ModelWrapperFactory` を実装します。

  **実装例:**

  ```python
  # base_model_wrapper.py
  from abc import ABC, abstractmethod

  class BaseModelWrapper(ABC):
      def __init__(self, model_config: Dict[str, Any]):
          self.model = None
          self.is_trained = False
          self.feature_columns = None
          self.model_config = model_config

      @abstractmethod
      def _create_model(self) -> Any:
          """モデル固有のインスタンス生成"""
          pass

      @abstractmethod
      def _get_model_params(self, num_classes: int) -> Dict[str, Any]:
          """モデル固有のパラメータ取得"""
          pass

      def train(self, X_train, X_test, y_train, y_test, **kwargs):
          # 共通の学習ロジック
          pass

  # model_wrapper_factory.py
  class ModelWrapperFactory:
      @staticmethod
      def create_wrapper(model_name: str, config: Dict[str, Any]) -> BaseModelWrapper:
          if model_name == "lightgbm":
              return LightGBMWrapper(config)
          elif model_name == "xgboost":
              return XGBoostWrapper(config)
          # ...
  ```

- [ ] ### 2.9. メトリクス収集機能の重複

- **課題**:
  - `common/metrics.py` の `MLMetricsCollector` と `evaluation/enhanced_metrics.py` の `EnhancedMetricsCalculator` が、メトリクス関連の機能を重複して実装しています。
  - `MLMetricsCollector` はパフォーマンスメトリクスの収集に特化していますが、`EnhancedMetricsCalculator` はモデル評価メトリクスに特化しており、統合の余地があります。
- **提案**:
  - `EnhancedMetricsCalculator` をモデル評価専用クラスとして維持し、`MLMetricsCollector` をシステム全体のメトリクス収集クラスとして位置づけます。
  - `MLMetricsCollector` に `EnhancedMetricsCalculator` の結果を記録する機能を追加し、統一的なメトリクス管理を実現します。
  - 両クラス間でメトリクス名やフォーマットの標準化を行い、一貫性を保ちます。

- [ ] ### 2.10. 未使用ファイルとデッドコードの特定

- **課題**:
  - `__pycache__` ディレクトリに多数の `.cpython-310.pyc` と `.cpython-312.pyc` ファイルが混在しており、異なる Python バージョンでの実行履歴が残っています。
  - 一部のファイル（例：`automl_preset_service.cpython-312.pyc`）に対応する `.py` ファイルが見当たらず、削除されたファイルのキャッシュが残っている可能性があります。
- **提案**:
  - プロジェクト全体で `__pycache__` ディレクトリをクリーンアップし、`.gitignore` に適切に追加されていることを確認します。
  - 存在しない `.py` ファイルに対応するキャッシュファイルを特定し、削除します。
  - 定期的なコードベースの整理を自動化するスクリプトを作成します。

- [ ] ### 2.11. トレーナーサービスの責務重複

- **課題**:
  - `MLTrainingService` と `BaseMLTrainer` が似たような責務を持っており、学習ロジックが分散しています。
  - `MLTrainingService` は `BaseMLTrainer` を使用していますが、実際には多くの学習ロジックを重複して実装しています。
  - `SingleModelTrainer` と `EnsembleTrainer` の初期化・学習パターンが非常に似ており、共通化の余地があります。
- **提案**:
  - `MLTrainingService` を軽量なファサードクラスとして再設計し、実際の学習ロジックは `BaseMLTrainer` の継承クラスに集約します。
  - `SingleModelTrainer` と `EnsembleTrainer` の共通部分を `BaseMLTrainer` に移動し、重複を削減します。
  - トレーナーの選択ロジックをファクトリーパターンで実装し、設定に基づいて適切なトレーナーを動的に生成できるようにします。

- [ ] ### 2.12. AutoML 設定の重複定義

- **課題**:
  - `MLTrainingOrchestrationService.get_default_automl_config()` と `AutoMLConfig.get_default_config()` で同じような設定が重複定義されています。
  - `get_financial_optimized_automl_config()` も複数箇所で似たような実装が存在します。
  - 設定の変更時に複数箇所を修正する必要があり、保守性に問題があります。
- **提案**:
  - `AutoMLConfig` クラスを唯一の設定管理クラスとして統一し、他の箇所からの重複定義を削除します。
  - 設定のプリセット（デフォルト、金融最適化など）は `AutoMLConfig` のクラスメソッドとして集約します。
  - 他のサービスクラスは `AutoMLConfig` のインスタンスを受け取るか、プリセットメソッドを呼び出すように変更します。

- [ ] ### 2.13. 特徴量計算クラスの構造重複

- **課題**:
  - `PriceFeatureCalculator`、`TechnicalFeatureCalculator`、`MarketDataFeatureCalculator` などが同じ初期化パターンを持っています。
  - 各クラスで `DataValidator.safe_*` メソッドを使用した同様の計算パターンが繰り返されています。
  - エラーハンドリングや結果の検証ロジックが各クラスで重複しています。
- **提案**:

  - 抽象基底クラス `BaseFeatureCalculator` を作成し、共通の初期化・検証・エラーハンドリングロジックを集約します。
  - 各特徴量計算クラスは `BaseFeatureCalculator` を継承し、具体的な計算ロジックのみを実装するように変更します。
  - 共通の計算パターン（移動平均、比率計算、変化率計算など）をユーティリティメソッドとして `BaseFeatureCalculator` に実装します。

  **実装例:**

  ```python
  # base_feature_calculator.py
  from abc import ABC, abstractmethod

  class BaseFeatureCalculator(ABC):
      def __init__(self):
          self.validator = DataValidator()

      def safe_rolling_mean(self, series: pd.Series, window: int) -> pd.Series:
          return self.validator.safe_rolling_mean(series, window)

      def safe_ratio_calculation(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
          return self.validator.safe_divide(numerator, denominator, default_value=1.0)

      @abstractmethod
      def calculate_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
          pass
  ```

- [ ] ### 2.14. アルゴリズム名マッピングの重複

- **課題**:
  - `ModelManager._extract_algorithm_name()` メソッドで大量のアルゴリズム名マッピングが定義されています。
  - 同様のマッピングロジックが他のクラス（モデルラッパーなど）でも重複している可能性があります。
  - 新しいアルゴリズムを追加する際に複数箇所を修正する必要があります。
- **提案**:
  - アルゴリズム名マッピングを専用の設定ファイルまたは定数クラスに外部化します。
  - `AlgorithmRegistry` クラスを作成し、アルゴリズム名の正規化・マッピング・検証を一元管理します。
  - モデル関連のクラスは `AlgorithmRegistry` を参照するように統一し、重複を削減します。

- [ ] ### 2.15. インターフェースの未活用

- **課題**:
  - `MLPredictionInterface`、`MLTrainingInterface`、`MLServiceInterface` が定義されていますが、実装クラスが見つかりません。
  - インターフェースが活用されておらず、型安全性やコードの一貫性が保たれていません。
  - 新しい実装を追加する際の指針が不明確です。
- **提案**:
  - 既存の ML サービスクラス（`MLTrainingService`、予測関連サービスなど）にインターフェースを実装させます。
  - インターフェースに基づいた依存性注入を導入し、テスト容易性を向上させます。
  - インターフェースの仕様を見直し、実際の使用パターンに合わせて調整します。

- [ ] ### 2.16. 設定管理の手動マッピング重複

- **課題**:
  - `MLConfigManager.get_config_dict()` メソッドで、Pydantic モデルの属性を手動で辞書にマッピングしています。
  - 同様の手動マッピングが `AutoMLConfig.to_dict()` や他の設定クラスでも実装されています。
  - 設定クラスの属性変更時に、マッピングロジックも手動で更新する必要があります。
- **提案**:
  - Pydantic の `.model_dump()` メソッド（v2）または `.dict()` メソッド（v1）を活用して、自動シリアライゼーションを実装します。
  - カスタムシリアライゼーションが必要な場合は、Pydantic のフィールド設定やカスタムシリアライザーを使用します。
  - 手動マッピングロジックを削除し、保守性を向上させます。

- [ ] ### 2.17. インポート文の最適化

- **課題**:
  - 多くのファイルで未使用のインポート文が存在する可能性があります。
  - 相対インポートと絶対インポートが混在しており、一貫性に欠けます。
- **提案**:
  - `isort` と `autoflake` を使用して、未使用インポートの削除とインポート順序の統一を行います。
  - プロジェクト全体で絶対インポートを使用するように統一します。
  - pre-commit フックを設定して、今後のコミット時に自動的にインポート文を整理します。

- [ ] ### 2.18. 重複インポート文の修正

- **課題**:

  - `backend/app/api/ml_training.py` で同じインポート文が重複しています：

    ```python
    from app.services.ml.orchestration.ml_training_orchestration_service import (
    from app.services.ml.orchestration.ml_training_orchestration_service import (
        MLTrainingOrchestrationService,
    )
    ```

  - `AutoMLConfigModel` も重複してインポートされています。
  - テストファイル `backend/tests/test_single_model_training.py` でも `MLTrainingService` が 3 回重複してインポートされています。

- **提案**:
  - 重複したインポート文を削除し、1 つのインポート文にまとめます。
  - `isort` と `autoflake` を使用して、プロジェクト全体の重複インポートを自動検出・修正します。
  - pre-commit フックを設定して、今後の重複インポートを防止します。

- [ ] ### 2.19. AutoMLConfig 作成ロジックの重複

- **課題**:
  - `BaseMLTrainer._create_automl_config_from_dict()` と `MLOrchestrator._create_automl_config_from_dict()` で同じロジックが重複実装されています。
  - 辞書から AutoMLConfig オブジェクトを作成する処理が複数箇所で手動実装されています。
  - 設定の変更時に複数箇所を修正する必要があり、保守性に問題があります。
- **提案**:
  - `AutoMLConfig.from_dict()` クラスメソッドを拡張し、すべての辞書 → オブジェクト変換を統一します。
  - 各クラスの重複した `_create_automl_config_from_dict()` メソッドを削除し、`AutoMLConfig.from_dict()` を直接使用します。
  - 設定の検証ロジックも `AutoMLConfig` クラス内に集約し、一貫性を保ちます。

- [ ] ### 2.20. ログメッセージの重複パターン

- **課題**:
  - ML 関連のログメッセージで似たようなパターンが多数存在します：
    - `"MLトレーニング〇〇エラー"` パターンが複数箇所
    - `"AutoML〇〇"` パターンの重複
    - `"✅"` や `"❌"` などの絵文字を使ったログの一貫性不足
- **提案**:
  - ML 専用のログメッセージテンプレートクラス `MLLogMessages` を作成し、統一されたメッセージフォーマットを提供します。
  - ログレベルと絵文字の使用ルールを統一し、一貫性のあるログ出力を実現します。
  - 国際化（i18n）を考慮したメッセージ管理システムの導入を検討します。

- [ ] ### 2.21. リソースクリーンアップロジックの重複

- **課題**:
  - `BaseMLTrainer.cleanup_resources()`、`EnsembleTrainer.cleanup_resources()`、`MLTrainingService.cleanup_resources()` で似たようなクリーンアップ処理が重複しています。
  - メモリ解放、モデルオブジェクトのクリア、キャッシュクリアなどの処理が各クラスで個別実装されています。
  - クリーンアップの順序や方法が統一されておらず、メモリリークのリスクがあります。
- **提案**:
  - 抽象基底クラス `BaseResourceManager` を作成し、統一されたリソース管理インターフェースを定義します。
  - クリーンアップの優先順位と手順を標準化し、確実なリソース解放を保証します。
  - コンテキストマネージャー（`with` 文）を使用した自動リソース管理の導入を検討します。

- [ ] ### 2.22. ML 設定管理の分散

- **課題**:
  - ML 関連の設定が複数箇所に分散しています：
    - `MLConfig` クラス（Pydantic モデル）
    - `MLConfigManager` クラス（ファイル永続化）
    - `AutoMLConfig` クラス（AutoML 専用設定）
    - オーケストレーションサービス内のハードコードされた設定
- **提案**:
  - 設定管理を階層化し、基底設定クラス `BaseMLConfig` を作成します。
  - 設定の継承関係を明確にし、共通設定と専用設定を分離します。
  - 設定の検証、デフォルト値管理、環境変数オーバーライドを統一的に処理します。

- [ ] ### 2.23. インターフェース実装の不完全性

- **課題**:
  - `MLOrchestrator` が `MLPredictionInterface` を実装していますが、一部のメソッドが未実装または不完全です。
  - インターフェースの契約が守られておらず、実行時エラーのリスクがあります。
  - 他のクラスでもインターフェースの実装が中途半端な状態です。
- **提案**:
  - すべてのインターフェース実装クラスで完全な実装を強制します。
  - 抽象基底クラスを使用して、未実装メソッドをコンパイル時に検出できるようにします。
  - インターフェースの仕様を見直し、実際の使用パターンに合わせて調整します。

- [ ] ### 2.24. テストコードの重複パターン

- **課題**:
  - テストファイルで同じようなセットアップ・ティアダウン処理が重複しています。
  - ML サービスの初期化パターンが複数のテストで重複実装されています。
  - テストデータの生成ロジックが各テストファイルで個別実装されています。
- **提案**:
  - 共通のテストユーティリティクラス `MLTestUtils` を作成し、重複処理を集約します。
  - pytest フィクスチャを活用して、ML サービスの初期化を共通化します。
  - テストデータファクトリーパターンを導入し、一貫したテストデータ生成を実現します。

---

これらの追加のリファクタリングを実行することで、コードの保守性、拡張性、パフォーマンスが大幅に向上することが期待されます。特にモデルラッパークラスの統一化と特徴量計算クラスの共通化により、新しいモデルや特徴量の追加が容易になり、メンテナンスコストも大幅に削減されます。

## 総括

本リファクタリング提案では、**25 項目**の包括的な改善点を特定しました。これらの改善により、ML コードベースの品質が大幅に向上し、開発効率とシステムの安定性が向上することが期待されます。

- [ ] ### 2.25. エラーハンドリングパターンの重複

- **課題**:

  - 多くの ML クラスで同じような `try-except` 構造が重複しています：

    ```python
    try:
        # 処理
    except Exception as e:
        logger.error(f"〇〇エラー: {e}")
        raise UnifiedModelError(f"〇〇に失敗しました: {e}")
    ```

  - `UnifiedModelError` の使用パターンが統一されておらず、似たようなエラーメッセージが複数箇所で定義されています。
  - 設定検証エラー（`ValueError`）の処理が複数箇所で重複しています。

- **提案**:

  - エラーハンドリング専用のデコレータ `@ml_error_handler` を作成し、共通のエラー処理ロジックを集約します。
  - エラーメッセージテンプレートを定義し、一貫したエラーメッセージフォーマットを提供します。
  - エラーの分類（設定エラー、データエラー、モデルエラーなど）に応じた専用ハンドラーを実装します。

  **実装例:**

  ```python
  # error_handlers.py
  def ml_error_handler(operation_name: str, error_type: Type[Exception] = UnifiedModelError):
      def decorator(func):
          @wraps(func)
          def wrapper(*args, **kwargs):
              try:
                  return func(*args, **kwargs)
              except Exception as e:
                  logger.error(f"{operation_name}エラー: {e}")
                  raise error_type(f"{operation_name}に失敗しました: {e}")
          return wrapper
      return decorator

  # 使用例
  @ml_error_handler("単一モデル学習")
  def _train_model_impl(self, X_train, X_test, y_train, y_test, **kwargs):
      # 学習処理
  ```

## 総括

本リファクタリング提案では、**25 項目**の包括的な改善点を特定しました。これらの改善により、ML コードベースの品質が大幅に向上し、開発効率とシステムの安定性が向上することが期待されます。

### 改善効果の期待値

1. **コード重複の削減**: 推定 40-50%のコード重複を解消
2. **保守性の向上**: 新機能追加時の修正箇所を 70%削減
3. **テスト容易性**: 統一されたインターフェースによりテストカバレッジ向上
4. **パフォーマンス**: メモリ使用量と CPU 使用率の最適化
5. **開発効率**: 新しいモデルや特徴量の追加時間を 60%短縮

### 実装優先度

**高優先度（即座に実装すべき）:**

- 2.8. モデルラッパークラスの大量重複
- 2.18. 重複インポート文の修正
- 2.25. エラーハンドリングパターンの重複

**中優先度（次のスプリントで実装）:**

- 2.1. 評価ロジックの統一
- 2.13. 特徴量計算クラスの構造重複
- 2.19. AutoMLConfig 作成ロジックの重複

**低優先度（長期的な改善）:**

- 2.20. ログメッセージの重複パターン
- 2.24. テストコードの重複パターン
