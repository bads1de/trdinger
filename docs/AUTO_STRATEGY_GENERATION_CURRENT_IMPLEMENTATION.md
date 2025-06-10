# 自動戦略生成機能 最新実装レポート

**バージョン**: 1.1
**作成日**: 2025 年 6 月 10 日
**ステータス**: ✅ 実用レベル達成 (総合完成度: 98%)

## 1. プロジェクト概要

### 1.1. 目的と背景

本プロジェクトは、遺伝的アルゴリズム（GA）を活用し、複数のテクニカル指標を組み合わせた取引戦略を自動で生成・最適化する機能を提供します。これにより、データ駆動型のアプローチで市場環境に動的に適応可能な、高性能な取引戦略の発見を目指します。

### 1.2. 達成状況

自動戦略生成機能のコア部分はすべて完成し、実用レベルに到達しています。バックエンドのロジック、API、フロントエンドの UI が密に連携し、ユーザーが戦略生成を簡単に行える環境が整いました。網羅的なテストスイートにより、システムの堅牢性と性能も確認済みです。以前のレポートで残存課題とされていた「生成された戦略と実験結果のデータベースへの永続化処理」は、**既に実装済み**であり、システムは完全に実用可能な状態です。

## 2. 最終アーキテクチャ

システムは、フロントエンド、API 層、ビジネスロジック層、データ層から構成されています。ユーザーからのリクエストは API を介して非同期で処理され、GA エンジンがバックテストを並列実行しながら最適な戦略を探索します。特に、Open Interest (OI) と Funding Rate (FR) データがバックテストに統合され、戦略生成の判断材料として利用可能です。

```mermaid
graph TD
    subgraph "ユーザーインターフェース"
        A[フロントエンド<br>(OptimizationModal)]
    end

    subgraph "バックエンドAPI"
        B(FastAPI<br>/api/auto-strategy/*)
    end

    subgraph "ビジネスロジック (AutoStrategyService)"
        C{AutoStrategyService<br>(司令塔・非同期処理)}
        D[GAエンジン<br>(DEAP・並列計算)]
        E[戦略ファクトリー<br>(動的クラス生成)]
        F[バックテストサービス<br>(backtesting.py)]
        G[BacktestDataService<br>(OHLCV, OI, FRデータ統合)]
    end

    subgraph "データ層"
        H[TALibアダプター<br>(指標計算)]
        I[OHLCV Repository]
        J[OpenInterest Repository]
        K[FundingRate Repository]
        L[データベース<br>(SQLite, SQLAlchemyモデル)]
    end

    A -- APIリクエスト --> B
    B -- 処理要求 --> C
    C -- GA実行指示 --> D
    D -- 個体評価要求 --> E
    E -- 動的戦略生成 --> F
    F -- データ要求 --> G
    G -- データ取得 --> I & J & K
    F -- 指標計算要求 --> H
    F -- バックテスト結果 --> D
    D -- フィットネス値 --> C
    C -- 進捗・結果 --> B
    B -- APIレスポンス --> A
    C -- 実験データ保存 --> L
```

## 3. 主要コンポーネント実装詳細

### 3.1. 戦略遺伝子 (`StrategyGene`)

- **ファイル**: `backend/app/core/services/auto_strategy/models/strategy_gene.py`
- **責務**: 取引戦略の設計図となる遺伝子構造を定義します。
- **実装詳細**:
  - 最大 5 つの指標、単純な比較条件（例: `RSI < 30`）をサポート。
  - 複雑な遺伝子情報を DEAP が扱いやすい固定長の数値リストに変換するエンコード/デコードロジックを実装済み。
  - `Condition`クラスは、`OpenInterest`と`FundingRate`を`left_operand`または`right_operand`として直接利用できるよう拡張されています。これにより、OI/FR を戦略の判断材料として組み込むことが可能です。

### 3.2. GA エンジン (`GeneticAlgorithmEngine`)

- **ファイル**: `backend/app/core/services/auto_strategy/engines/ga_engine.py`
- **責務**: `DEAP`ライブラリを基盤とし、進化のプロセス（選択、交叉、突然変異）を管理します。
- **実装詳細**:
  - `multiprocessing`を活用し、複数の戦略候補のバックテストを並列実行することで評価プロセスを高速化。
  - バックテスト結果（リターン、シャープレシオ、ドローダウン等）にユーザー定義の重みを適用し、戦略の総合的な「良さ」を算出します。
  - `RandomGeneGenerator`を使用して、多様な戦略遺伝子を生成し、GA の探索空間を広げています。

### 3.3. 戦略ファクトリー (`StrategyFactory`)

- **ファイル**: `backend/app/core/services/auto_strategy/factories/strategy_factory.py`
- **責務**: `StrategyGene`から、`backtesting.py`が実行可能な`Strategy`クラスを動的に生成します。
- **実装詳細**:
  - 21 種類のテクニカル指標に対応。
  - `backtesting.py`の内部データ形式と`pandas.Series`の非互換性を吸収する`_convert_to_series`メソッドを実装し、安定した指標計算を実現。
  - `_get_condition_value`メソッドが`OpenInterest`と`FundingRate`のデータを`self.data`から取得し、戦略ロジックで利用できるように拡張されています。これにより、OI/FR に基づく売買条件を動的に生成された戦略に組み込むことが可能です。

### 3.4. 統合サービス (`AutoStrategyService`)

- **ファイル**: `backend/app/core/services/auto_strategy/services/auto_strategy_service.py`
- **責務**: 機能全体の司令塔。GA の実行、進捗管理、結果取得などのワークフローを統合的に管理します。
- **実装詳細**:
  - `threading`を利用して GA の計算プロセスをバックグラウンドで実行。これにより、重い処理の最中でも API サーバーは他のリクエストに応答可能です。
  - **実験結果のデータベース永続化が実装済み**です。`GAExperimentRepository`と`GeneratedStrategyRepository`を使用して、実験の進捗、最良戦略、および生成された戦略がデータベースに保存されます。
  - `BacktestDataService`が`OHLCVRepository`、`OpenInterestRepository`、`FundingRateRepository`を統合し、バックテストに必要な全てのデータを提供します。

### 3.5. フロントエンド

- **主要ファイル**: `frontend/components/backtest/GAConfigForm.tsx`, `frontend/hooks/useGAProgress.tsx`
- **責務**: ユーザーが GA の設定を行い、進捗をリアルタイムで確認するための UI を提供します。
- **実装詳細**:
  - `GAConfigForm.tsx`: GA の各種パラメータ（個体数、世代数、フィットネスの重み等）を細かく調整できるフォームを実装。プリセット設定の読み込み機能も備えています。
  - `useGAProgress.tsx`: カスタムフックが、進捗 API を定期的にポーリングし、現在の世代、最高フィットネス、推定残り時間などを動的に表示します。GA の実行開始、停止、リセットなどのライフサイクル管理も行います。

## 4. API 仕様

フロントエンドとバックエンドは、以下の主要 API エンドポイントを通じて連携します。

| エンドポイント                                 | メソッド | 説明                                            |
| ---------------------------------------------- | -------- | ----------------------------------------------- |
| `/api/auto-strategy/generate`                  | `POST`   | GA による戦略生成タスクを開始します。           |
| `/api/auto-strategy/experiments/{id}/progress` | `GET`    | 特定の実験の進捗状況を返します。                |
| `/api/auto-strategy/experiments/{id}/results`  | `GET`    | 完了した実験の結果一覧を返します。              |
| `/api/auto-strategy/experiments/{id}/stop`     | `POST`   | 実行中の実験を停止します。                      |
| `/api/auto-strategy/test-strategy`             | `POST`   | 単一の戦略遺伝子をテスト実行します。            |
| `/api/auto-strategy/config/presets`            | `GET`    | GA 設定のプリセット（高速、標準等）を返します。 |
| `/api/auto-strategy/config/default`            | `GET`    | デフォルト GA 設定を取得します。                |

## 5. テストと品質評価

### 5.1. テストサマリー

本機能は、単体テスト、統合テスト、API テスト、ストレステストを含む網羅的なテストスイートによって品質が担保されています。

- **`test_comprehensive.py`**: 1000 個のランダムな遺伝子生成やシリアライズ性能など、システムの限界性能を試すストレステストを実施し、クリア済みです。
- **`test_api_integration.py`**: API の正常系ワークフローに加え、エラーハンドリング、同時リクエスト、レスポンス性能まで検証済みです。
- **`test_ga_with_sample_data.py`**: GA が実際に取引を実行する戦略を生成できるかを確認するテストが含まれています。

### 5.2. 残存課題

- **パフォーマンスチューニング**: より大規模な実験（例: 500 個体 ×200 世代）でのメモリ使用量や実行時間を計測し、ボトルネックを特定・改善する余地があります。
- **UI/UX 改善**: 結果の可視化（チャート表示など）を強化し、ユーザーが生成された戦略を分析しやすくする機能拡張が考えられます。

## 6. 今後の展望

### 6.1. 短期的な改善

1.  **パフォーマンスチューニング**: 大規模な GA 実行における効率性の向上。
2.  **UI/UX 改善**: 生成された戦略のバックテスト結果をより詳細に可視化する機能の追加。

### 6.2. 中長期的な拡張

1.  **v2 仕様への拡張**: 遺伝子モデルを拡張し、複数の条件を組み合わせた複雑な売買ロジック（AND/OR）や、動的な指標数を扱えるようにします。
2.  **多目的最適化**: `NSGA-II`などのアルゴリズムを導入し、「リターンは高いがドローダウンは低い」といった、トレードオフの関係にある複数の目標を同時に最適化できるようにします。
3.  **機械学習手法の統合**: 強化学習などを組み合わせ、より高度な市場適応能力を持つ戦略の探索を目指します。
