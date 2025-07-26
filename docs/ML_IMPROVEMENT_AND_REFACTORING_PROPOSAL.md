# ML システム改善・強化提案 および リファクタリング提案

## 1. 総評

`backend/app/core/services/ml/`以下の実装は、特徴量エンジニアリングからモデル学習、アンサンブル、AutoML、最適化、モデル管理、オーケストレーションに至るまで、非常に高度で包括的な ML パイプラインが構築されており、素晴らしい完成度です。特に、責務分離を意識したクラス設計や、エラーハンドリング、ロギングの仕組みは堅牢なシステムの基盤となっています。

今後の改善は、以下の 3 つのテーマに集約されると考えられます。

1. **依存関係の疎結合化**: クラス間の依存をより緩やかにし、テスト容易性とコンポーネントの交換可能性を高める。
2. **金融時系列データ特性への特化**: `train_test_split`など、一般的な機械学習の手法から、金融時系列データ特有の課題（リーク、非定常性）にさらに適した手法へ移行する。
3. **運用とスケーラビリティの強化**: 本番環境での安定稼働や、将来的なシステム拡張を見据えた状態管理・プロセス管理を導入する。

---

## 2. 具体的な改善・強化点

### 【アーキテクチャと設計】

#### 提案 1: 依存性注入 (Dependency Injection) の導入によるテスト容易性の向上

- **現状**: `BaseMLTrainer`や`MLTrainingOrchestrationService`内で、`SessionLocal`や各種リポジトリ、サービスが直接インポート・インスタンス化されています。これにより、クラスが特定のデータベース実装やサービスクラスと密結合になっています。
- **改善案**: FastAPI の DI システムなどを活用し、依存するオブジェクトをコンストラクタやメソッドの引数として外部から注入する（「制御の反転」）ことを推奨します。
- **対象ファイル**:
  - `base_ml_trainer.py`: `_get_fear_greed_data`内で`SessionLocal`と`FearGreedIndexRepository`を直接使用。
  - `orchestration/ml_training_orchestration_service.py`: `get_data_service`内で各リポジトリを直接インスタンス化。
- **メリット**:
  - ユニットテスト時に、本物のデータベースの代わりにモックオブジェクトを簡単に注入できるようになります。
  - クラスの責務がより明確になり、再利用性が向上します。

### 【特徴量エンジニアリング】

#### 提案 3: AutoML のメモリ管理の最適化

- **現状**: `autofeat_calculator.py`などで`gc.collect()`が明示的に呼ばれています。これはメモリリークの兆候を緩和する対症療法である可能性があります。
- **改善案**: `memory-profiler`などのツールを使用して、メモリ使用量の多い箇所やオブジェクトの循環参照がないかを特定し、根本原因を解決することを推奨します。コンテキストマネージャ (`@contextmanager`) を使ったリソース管理は非常に良いアプローチなので、その中で根本的なクリーンアップを目指します。
- **対象ファイル**: `feature_engineering/automl_features/autofeat_calculator.py`, `performance_optimizer.py`
- **メリット**: ガベージコレクションへの過度な依存を減らし、より安定的で予測可能なメモリ使用量を実現します。

#### 提案 4: 特徴量の冗長性管理の強化

- **現状**: `AdvancedFeatureSelector`で相関の高い特徴量を除去する仕組みがありますが、手動生成された特徴量と AutoML によって生成された特徴量の間での冗長性は考慮されていない可能性があります。
- **改善案**: `EnhancedFeatureEngineeringService`の最終段階で、すべての特徴量（手動＋ AutoML）を対象として`AdvancedFeatureSelector`を実行し、全体から見た冗長な特徴量（例：`manual_RSI`と`TSF_rsi_...`）を削除するステップを追加します。
- **対象ファイル**: `feature_engineering/enhanced_feature_engineering_service.py`, `feature_engineering/automl_features/feature_selector.py`
- **メリット**: モデルの学習効率が向上し、過学習のリスクが低減します。

### 【運用とオーケストレーション】

#### 提案 8: バックグラウンドタスクの確実な停止メカニズムの導入

- **現状**: `stop_training`は状態フラグを更新するのみで、実際に計算負荷の高い学習プロセスを強制的に停止する仕組みがありません。
- **改善案**: `multiprocessing.Process`などを使って学習タスクを別プロセスで実行し、そのプロセス ID を状態管理ストア（提案 7 の Redis など）に保存します。停止リクエストがあった際には、そのプロセス ID に対して停止シグナルを送ることで、確実にリソースを解放できます。
- **対象ファイル**: `orchestration/ml_training_orchestration_service.py`
- **メリット**: 不要な計算リソースの消費を防ぎ、システムの応答性と安定性を高めます。

---

## 3. ML Services Refactoring Proposal

### 3.1. グローバルな状態管理 (`ml_training_orchestration_service.py`)

- **課題**: ML のトレーニング状態が、単一のグローバル変数 `training_status` によって管理されています。これは、複数のトレーニングリクエストを同時に処理できず、サーバーを複数インスタンスでスケールアウトした場合に状態が共有されないという深刻な問題を抱えています。
- **提案**:
  - **状態管理の外部化**: トレーニングジョブの状態を、グローバル変数ではなく、データベース（例: `ml_jobs` テーブル）またはインメモリキャッシュ（Redis など）で管理するように変更します。
  - **ジョブ ID の導入**: 各トレーニングリクエストに対して一意のジョブ ID を発行し、クライアントはその ID を使って進捗や結果を非同期に問い合わせるアーキテクチャに変更します。
  - **`BackgroundTaskManager` の活用**: `background_task_manager.py` を活用し、タスクごとのリソースと状態をより堅牢に管理します。

### 3.2. 複雑な特徴量生成パイプライン (`EnhancedFeatureEngineeringService`)

- **課題**: `calculate_enhanced_features` メソッドが、手動特徴量、TSFresh、Featuretools、AutoFeat の処理を逐次的に呼び出す長大なものになっており、可読性と拡張性が低下しています。
- **提案**:

  - **パイプラインパターンの導入**: 各特徴量生成ステップを独立した「ステージ」としてカプセル化します。
  - **動的パイプライン構築**: `automl_config` の設定に基づき、実行するステージを動的に組み合わせるパイプラインを構築します。これにより、新しい特徴量生成手法の追加や順序変更が容易になります。

  ```mermaid
  graph TD
      A[Input Data] --> B(Manual Features Stage);
      B --> C{TSFresh Enabled?};
      C -- Yes --> D[TSFresh Stage];
      C -- No --> E;
      D --> E{Featuretools Enabled?};
      E -- Yes --> F[Featuretools Stage];
      E -- No --> G;
      F --> G{AutoFeat Enabled?};
      G -- Yes --> H[AutoFeat Stage];
      G -- No --> I[Output Features];
      H --> I;
  end
  ```

### 3.3. 設定管理の一貫性の欠如 (`config/ml_config.py`)

- **課題**: ML 関連の設定が `dataclass` で定義されており、プロジェクトの他の部分で使用されている `pydantic-settings` と一貫性がありません。これにより、環境変数からの設定オーバーライドが困難になっています。
- **提案**:
  - **`pydantic-settings` への統一**: `DataProcessingConfig`, `ModelConfig` などの全ての ML 設定クラスを `pydantic_settings.BaseSettings` を継承するように変更します。これにより、`.env` ファイルや環境変数からの設定読み込みが容易になり、プロジェクト全体での設定管理方法が標準化されます。

### 3.4. 責務の曖昧さ (`MLTrainingService` と `EnsembleTrainer`)

- **課題**: `MLTrainingService` がハイパーパラメータ最適化 (`_train_with_optimization`) のような複雑なロジックを保持しており、`EnsembleTrainer` との責務分担がやや曖昧です。
- **提案**:
  - **`MLTrainingService` の責務を限定**: `MLTrainingService` の役割は、API からのリクエストを解釈し、適切なトレーナー（`EnsembleTrainer`）を初期化することに限定します。
  - **トレーナーへの責務移譲**: ハイパーパラメータ最適化のロジックは、`EnsembleTrainer` 自身、または専用の `OptimizationCoordinator` のようなクラスに移動します。これにより、トレーナーが自身の学習プロセス（通常学習と最適化学習）を完全に管理できるようになります。

## 4. 期待される効果

- **堅牢性とスケーラビリティの向上**: グローバルな状態管理を排除することで、システムの堅牢性が向上し、将来的なスケールアウトに対応可能になります。
- **保守性と拡張性の向上**: パイプラインパターンや責務の明確化により、新しい特徴量生成手法やモデルの追加、既存ロジックの変更が容易になります。
- **設定管理の統一**: プロジェクト全体で設定管理の方法が統一され、開発体験が向上します。
- **コードの可読性向上**: 各モジュールの関心事が明確になり、コードが理解しやすくなります。
