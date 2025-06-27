### オートストラテジーシステムアーキテクチャ分析

#### 概要

このオートストラテジーシステムは、遺伝的アルゴリズム（GA）を用いて自動的に取引戦略を生成・最適化するためのバックエンドサービスです。`backtesting.py` フレームワークと統合されており、生成された戦略のパフォーマンスを評価し、その結果を永続化します。システムは、GA の実行、進捗管理、結果保存といった複雑なプロセスを複数の専門コンポーネントに分離し、高いモジュール性と拡張性を実現しています。

#### 主要コンポーネント

1.  **`AutoStrategyService`**:

    - **役割**: システム全体のオーケストレーター。GA の開始、進捗の監視、結果の保存、戦略の検証など、オートストラテジー生成プロセスの高レベルな管理を行います。
    - **相互作用**: `ExperimentManager`, `ProgressTracker`, `GeneticAlgorithmEngine`, `BacktestService`, `StrategyFactory`, および各種データベースリポジトリと連携します。

2.  **`GeneticAlgorithmEngine`**:

    - **役割**: 遺伝的アルゴリズムの中核を実装。DEAP ライブラリを利用し、戦略遺伝子の生成、評価、選択、交叉、突然変異といった進化プロセスを管理します。
    - **相互作用**: `BacktestService` を利用して戦略のパフォーマンスを評価し、`StrategyFactory` を利用して実行可能な戦略クラスを生成します。`GAConfig` に基づいて動作し、`ProgressManager` を通じて進捗を報告します。

3.  **`StrategyFactory`**:

    - **役割**: `StrategyGene` データモデルから `backtesting.py` 互換の動的な取引戦略クラスを生成します。
    - **相互作用**: `IndicatorInitializer` (テクニカル指標の初期化), `ConditionEvaluator` (エントリー/イグジット条件の評価), `DataConverter` (データ形式変換) と連携し、戦略の内部ロジックを構築します。

4.  **`FitnessCalculator`**:

    - **役割**: GA における個体（戦略遺伝子）の適応度を計算します。バックテスト結果からパフォーマンス指標を抽出し、`GAConfig` で定義された重みと制約条件に基づいてフィットネススコアを算出します。
    - **相互作用**: `GeneticAlgorithmEngine` から呼び出され、`BacktestService` を利用してバックテストを実行し、`StrategyFactory` を利用して戦略クラスを生成します。

5.  **`ExperimentManager`**:

    - **役割**: GA 実験のライフサイクル（作成、開始、停止、完了、失敗）と、バックグラウンドスレッドでの実行を管理します。実験情報をメモリとデータベースに永続化します。
    - **相互作用**: `AutoStrategyService` から呼び出され、`GAExperimentRepository` を使用してデータベースと連携します。

6.  **`ProgressTracker`**:

    - **役割**: GA 実験のリアルタイム進捗を追跡し、コールバックメカニズムを通じて外部に通知します。進捗データをメモリとデータベースに保存します。
    - **相互作用**: `GeneticAlgorithmEngine` から進捗情報を受け取り、`AutoStrategyService` に通知します。`GAExperimentRepository` を使用してデータベースと連携します。

7.  **データモデル (`GAConfig`, `StrategyGene`, `GAProgress`)**:

    - **`GAConfig`**: 遺伝的アルゴリズムの実行パラメータ（個体数、世代数、フィットネス重みなど）を定義します。
    - **`StrategyGene`**: 取引戦略の遺伝子表現（テクニカル指標、エントリー/イグジット条件、リスク管理設定など）を定義します。
    - **`GAProgress`**: GA 実行のリアルタイム進捗情報（現在の世代、最高フィットネス、実行時間など）を保持します。

8.  **リポジトリ層**:
    - **役割**: データベースとの永続化層を提供し、OHLCV データ、Open Interest、Funding Rate、生成された戦略、GA 実験結果、バックテスト結果などのデータを保存・取得します。
    - **相互作用**: `AutoStrategyService`, `ExperimentManager`, `ProgressTracker`, `BacktestService` など、データが必要な各サービスから利用されます。

#### データフロー

1.  **実験開始**: API リクエストにより `AutoStrategyService.start_strategy_generation` が呼び出され、`GAConfig` とバックテスト設定が渡されます。
2.  **実験登録**: `ExperimentManager` が実験 ID を生成し、実験情報をデータベース (`GAExperimentRepository`) に保存し、バックグラウンドスレッドで GA 実行を開始します。
3.  **GA 実行**: `GeneticAlgorithmEngine` が `GAConfig` に基づいて初期個体群（`StrategyGene`）を生成し、進化ループを開始します。
4.  **戦略評価**: 各世代で、`GeneticAlgorithmEngine` は個体（`StrategyGene`）を `FitnessCalculator` に渡し評価を依頼します。
5.  **バックテスト実行**: `FitnessCalculator` は `StrategyGene` から `StrategyFactory` を介して動的な戦略クラスを生成し、`BacktestService` にバックテストを依頼します。`BacktestService` は `BacktestDataService` を通じて各種リポジトリから市場データ（OHLCV, OI, FR）を取得し、戦略を実行します。
6.  **フィットネス計算**: `BacktestService` から返されたバックテスト結果（パフォーマンス指標）を基に、`FitnessCalculator` がフィットネススコアを計算し、`GeneticAlgorithmEngine` に返します。
7.  **進捗更新**: `GeneticAlgorithmEngine` は各世代の完了時に `ProgressTracker` を通じて進捗情報を更新します。`ProgressTracker` はメモリ上の進捗データを更新し、登録されたコールバックを通じてリアルタイムで進捗を通知し、データベース (`GAExperimentRepository`) にも進捗を保存します。
8.  **進捗更新**: `GeneticAlgorithmEngine` は各世代の完了時に `ProgressManager` を通じて進捗情報（最高フィットネス、平均フィットネス、実行時間など）を作成し、`ProgressTracker` を通じて進捗を更新します。`ProgressTracker` はメモリ上の進捗データを更新し、登録されたコールバックを通じてリアルタイムで進捗を通知し、データベース (`GAExperimentRepository`) にも進捗を保存します。
9.  **結果保存**: GA の全世代が完了すると、`GeneticAlgorithmEngine` は最も適応度の高い戦略と最終結果を `AutoStrategyService` に返します。`AutoStrategyService` はこの結果を `GeneratedStrategyRepository` (生成戦略) と `BacktestResultRepository` (詳細バックテスト結果) を使用してデータベースに永続化します。
10. **実験完了**: `ExperimentManager` が実験を「完了」状態に更新します。

#### 技術スタック

- **プログラミング言語**: Python
- **遺伝的アルゴリズムフレームワーク**: DEAP
- **バックテストフレームワーク**: backtesting.py
- **データベース**:
  - ORM: SQLAlchemy
  - 開発/テスト: SQLite
  - 本番: PostgreSQL / TimescaleDB (推測)
- **データモデル**: Pydantic, dataclasses
- **並行処理**: `threading` モジュール (バックグラウンドでの GA 実行)
- **ロギング**: Python 標準の `logging` モジュール

#### 設計パターン

- **Factory Pattern**: `StrategyFactory` が `StrategyGene` から動的に戦略クラスを生成し、具体的なクラス生成ロジックを抽象化しています。
- **Strategy Pattern**: `backtesting.py` の `Strategy` クラスと、それから動的に生成される `GeneratedStrategy` が、異なる取引ロジックを独立して定義・切り替え可能にしています。
- **Observer Pattern**: `ProgressTracker` のコールバックメカニズムにより、GA の進捗更新がリアルタイムで複数のコンシューマに通知されます。
- **Facade Pattern**: `AutoStrategyService` が、GA エンジン、実験管理、進捗追跡、バックテストサービスなど、複数の複雑なサブシステムへの統一されたインターフェースを提供し、クライアントからのアクセスを簡素化しています。
- **Repository Pattern**: 各種リポジトリクラスがデータベース操作を抽象化し、ドメインロジックから永続化の詳細を分離しています。
- **Dependency Injection**: `AutoStrategyService` や `GeneticAlgorithmEngine` などのコンポーネントが、コンストラクタを通じて依存関係（例: `BacktestService`, `StrategyFactory`）を受け取ることで、結合度を低減し、テスト容易性を向上させています。
- **Separation of Concerns**: 各モジュールが明確な単一責任を持つように設計されており、コードの保守性、拡張性、理解しやすさが向上しています。
- **Data Transfer Object (DTO)**: `GAConfig`, `StrategyGene`, `GAProgress` などのデータクラスが、異なる層間でのデータ転送に利用されています。
