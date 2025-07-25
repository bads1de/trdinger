# ML Services Refactoring Proposal

## 1. はじめに

`backend/app/core/services/ml` パッケージは、本アプリケーションの機械学習機能の中核を担っています。現状、機能は実装されていますが、将来的な拡張性、保守性、そして堅牢性の観点からいくつかの改善点が見られます。この文書では、MLサービス層のリファクタリング案を提案します。

## 2. 課題と提案

### 2.1. グローバルな状態管理 (`ml_training_orchestration_service.py`)

- **課題**: MLのトレーニング状態が、単一のグローバル変数 `training_status` によって管理されています。これは、複数のトレーニングリクエストを同時に処理できず、サーバーを複数インスタンスでスケールアウトした場合に状態が共有されないという深刻な問題を抱えています。
- **提案**:
    - **状態管理の外部化**: トレーニングジョブの状態を、グローバル変数ではなく、データベース（例: `ml_jobs` テーブル）またはインメモリキャッシュ（Redisなど）で管理するように変更します。
    - **ジョブIDの導入**: 各トレーニングリクエストに対して一意のジョブIDを発行し、クライアントはそのIDを使って進捗や結果を非同期に問い合わせるアーキテクチャに変更します。
    - **`BackgroundTaskManager` の活用**: `background_task_manager.py` を活用し、タスクごとのリソースと状態をより堅牢に管理します。

### 2.2. 複雑な特徴量生成パイプライン (`EnhancedFeatureEngineeringService`)

- **課題**: `calculate_enhanced_features` メソッドが、手動特徴量、TSFresh、Featuretools、AutoFeatの処理を逐次的に呼び出す長大なものになっており、可読性と拡張性が低下しています。
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

### 2.3. 設定管理の一貫性の欠如 (`config/ml_config.py`)

- **課題**: ML関連の設定が `dataclass` で定義されており、プロジェクトの他の部分で使用されている `pydantic-settings` と一貫性がありません。これにより、環境変数からの設定オーバーライドが困難になっています。
- **提案**:
    - **`pydantic-settings` への統一**: `DataProcessingConfig`, `ModelConfig` などの全てのML設定クラスを `pydantic_settings.BaseSettings` を継承するように変更します。これにより、`.env` ファイルや環境変数からの設定読み込みが容易になり、プロジェクト全体での設定管理方法が標準化されます。

### 2.4. 責務の曖昧さ (`MLTrainingService` と `EnsembleTrainer`)

- **課題**: `MLTrainingService` がハイパーパラメータ最適化 (`_train_with_optimization`) のような複雑なロジックを保持しており、`EnsembleTrainer` との責務分担がやや曖昧です。
- **提案**:
    - **`MLTrainingService` の責務を限定**: `MLTrainingService` の役割は、APIからのリクエストを解釈し、適切なトレーナー（`EnsembleTrainer`）を初期化することに限定します。
    - **トレーナーへの責務移譲**: ハイパーパラメータ最適化のロジックは、`EnsembleTrainer` 自身、または専用の `OptimizationCoordinator` のようなクラスに移動します。これにより、トレーナーが自身の学習プロセス（通常学習と最適化学習）を完全に管理できるようになります。

### 2.5. モデルラッパーの配置 (`ensemble/base_ensemble.py`)

- **課題**: `LightGBMModel` や `XGBoostModel` といったモデルごとのラッパークラスが、アンサンブルの基底クラスと同じ `base_ensemble.py` 内に定義されており、ファイルの関心事が混在しています。
- **提案**:
    - **`ml/models/` ディレクトリの作成**: 新たに `ml/models/` ディレクトリを作成します。
    - **モデルラッパーの分離**: `lightgbm_wrapper.py`, `xgboost_wrapper.py` のように、モデルごとのラッパーを個別のファイルに分離します。これにより、`base_ensemble.py` はアンサンブルの基底クラス定義に集中でき、各モデルのラッパーは独立して管理・拡張できます。

## 3. 期待される効果

- **堅牢性とスケーラビリティの向上**: グローバルな状態管理を排除することで、システムの堅牢性が向上し、将来的なスケールアウトに対応可能になります。
- **保守性と拡張性の向上**: パイプラインパターンや責務の明確化により、新しい特徴量生成手法やモデルの追加、既存ロジックの変更が容易になります。
- **設定管理の統一**: プロジェクト全体で設定管理の方法が統一され、開発体験が向上します。
- **コードの可読性向上**: 各モジュールの関心事が明確になり、コードが理解しやすくなります。
