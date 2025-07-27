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

#### 提案 2: API エンドポイントの再編と責務の明確化

- **現状**: データソースや機能ごとに API ファイルが細分化されており (`data_collection.py`, `funding_rates.py`, `open_interest.py` など)、API 全体の構造が複雑化しています。
- **改善案**: REST の設計原則に基づき、関連性の高いエンドポイントをリソース指向でグルーピングし、統合します。
  - **データ管理 API の統合**: `/api/data` というエンドポイントを新設。
    - `POST /data/ohlcv/collect`: OHLCV データ収集
    - `GET /data/ohlcv`: OHLCV データ取得
    - `DELETE /data/all`: 全データリセット
    - `GET /data/funding-rates`: ファンディングレート取得
  - **ML 関連 API の統合**: `/api/ml` というエンドポイントに集約。
    - `POST /ml/train`: モデル学習
    - `GET /ml/models`: 学習済みモデル一覧
    - `POST /ml/features/generate`: AutoML 特徴量生成
- **メリット**: API の見通しが良くなり、クライアント側の実装が簡潔になります。バックエンドでも、関連ロジックが 1 つの`OrchestrationService`に集約されるため、保守性が向上します。

#### 提案 3: Auto-Strategy サービスの再設計とカプセル化

- **現状**: `auto_strategy`サービスが多数のサブモジュール（`calculators`, `engines`, `evaluators`など）に分割され、コンポーネント間の依存関係が複雑化しています。
- **改善案**:
  - **Facade パターンの徹底**: `AutoStrategyService`を Facade として位置づけ、GA の実行、進捗管理、結果取得などの主要な操作のみを公開します。内部の複雑なワークフロー（個体生成、評価、交叉、突然変異）は完全にカプセル化します。
  - **コンポーネントの独立性向上**: `IndicatorCalculator`のような汎用的な計算コンポーネントは、`auto_strategy`から`core/calculators`のような共通ディレクトリに移動させ、特定のサービスへの依存をなくします。
- **メリット**: 複雑な遺伝的アルゴリズムのロジックが隠蔽され、サービスの利用が容易になります。また、コンポーネントの再利用性とテスト容易性が向上します。

#### 提案 4: データベースリポジトリの汎用化　 ✅ **完了済み**

- **現状**: 各データモデル（`OHLCV`, `FundingRate`など）ごとにリポジトリクラスが作成されており、`get_latest_timestamp`のような共通メソッドが重複して実装されています。
- **改善案**: Python のジェネリクス (`Generic[T]`) を活用した`BaseRepository`を導入します。共通の CRUD 操作やクエリメソッドをこの基底クラスに実装し、各リポジトリはこれを継承してモデル固有のロジックのみを記述するように変更します。
- **対象ファイル**: `database/repositories/base_repository.py` および各リポジトリファイル。
- **メリット**: コードの重複を大幅に削減し、新しいデータモデルを追加する際の開発効率を向上させます。

#### 提案 5: 状態管理の外部化と堅牢化

- **現状**: `ml_training_orchestration_service.py`内で、グローバル変数`training_status`によって ML の学習状態が管理されています。これは単一プロセスでの動作を前提としており、将来的に API サーバーを複数プロセスで実行（スケールアウト）した場合に、状態の不整合を引き起こします。
- **改善案**: Redis や、より手軽なインメモリキャッシュライブラリ（例: `cachetools`）を導入し、学習ジョブの状態（実行中、完了、エラーなど）を外部で一元管理します。
- **対象ファイル**: `app/core/services/ml/orchestration/ml_training_orchestration_service.py`
- **メリット**: システムのスケーラビリティが向上し、API サーバーを再起動しても学習状態が失われない、より堅牢なシステムになります。

#### 提案 6: データ収集サービスの共通ロジック集約 ✅ **完了済み**

- **現状**: `FundingRateService`と`OpenInterestService`の両方で、ページネーションによって全期間のデータを取得するための類似したロジック (`_fetch_paginated_data`) が実装されています。
- **改善案**: このページネーションの共通ロジックを、基底クラスである`BybitService`に移動させ、各サービスクラスから呼び出すようにリファクタリングします。
- **対象ファイル**: `app/core/services/data_collection/bybit/bybit_service.py`, `funding_rate_service.py`, `open_interest_service.py`
- **メリット**: コードの重複（Don't Repeat Yourself）を排除し、保守性を向上させます。将来的に新しいデータソースを追加する際も、共通ロジックを再利用できます。

**実装完了内容**:

- ✅ `BybitService`基底クラスに汎用的な`_fetch_paginated_data`メソッドを実装
- ✅ 2 つのページネーション戦略を導入：
  - `until`戦略：ファンディングレート用（時刻指定による逆順取得）
  - `time_range`戦略：オープンインタレスト用（時間範囲指定による取得）
- ✅ 共通のページデータ処理ロジック（`_process_page_data`）を実装
- ✅ 重複チェック、差分更新、エラーハンドリングを統一
- ✅ `FundingRateService`と`OpenInterestService`から重複コードを削除
- ✅ 包括的なテストスイートを作成（11 個のテストケース）
- ✅ 統合テストで両サービスの互換性を確認

#### 提案 7: バックグラウンドタスクの確実な停止メカニズムの導入

- **現状**: `stop_training`は状態フラグを更新するのみで、実際に計算負荷の高い学習プロセスを強制的に停止する仕組みがありません。
- **改善案**: `multiprocessing.Process`などを使って学習タスクを別プロセスで実行し、そのプロセス ID を状態管理ストア（提案 7 の Redis など）に保存します。停止リクエストがあった際には、そのプロセス ID に対して停止シグナルを送ることで、確実にリソースを解放できます。
- **対象ファイル**: `orchestration/ml_training_orchestration_service.py`
- **メリット**: 不要な計算リソースの消費を防ぎ、システムの応答性と安定性を高めます。

---

## 3. ML Services Refactoring Proposal

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

### 3.3. 設定管理の一貫性の欠如 (`config/ml_config.py`) ✅ **完了済み**

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
