# オートストラテジー概要：AI による自動戦略生成の全貌

## 1. はじめに

「オートストラテジー」は、AI の一技術である遺伝的アルゴリズム（Genetic Algorithm, GA）を活用し、無数の可能性の中から最適な取引戦略を自動的に探索・生成するシステムです。本ドキュメントでは、ユーザーがリクエストを送信してから、システムがどのようにして最良の戦略を発見し、その結果を保存するのか、その技術的な仕組みと一連の流れを解説します。

## 2. エンドツーエンドのフロー

ユーザーが戦略生成を開始してから最終的な結果を受け取るまでの流れは、以下のシーケンス図で表されます。API、サービス層、GA エンジン、バックテストエンジンといった各コンポーネントが連携し、非同期で処理が進みます。

```mermaid
%%{init: { 'theme': 'dark', 'themeVariables': {'fontSize': '16px'}}}%%
sequenceDiagram
    actor User
    participant Frontend

    box "Backend System"
        participant API as "API層"
        participant Service as "AutoStrategyService"
        participant Engine as "GA Engine"
        participant Factory as "StrategyFactory"
        participant Backtest as "BacktestService"
        participant DB as "データベース"
    end

    User->>Frontend: 1. 戦略生成を開始
    Frontend->>API: 2. 戦略生成リクエスト (POST /api/auto-strategy/generate)
    API->>Service: 3. 非同期で戦略生成を開始
    API-->>Frontend: 4. 受付ID (experiment_id) を即時返却

    par "進捗ポーリング" and "バックエンドでの戦略生成"
        loop 定期的に進捗を確認
            Frontend->>API: GET /experiments/{id}/progress
            API-->>Frontend: 現在の進捗状況を返す
        end
    and
        Service->>Engine: 5. 遺伝的アルゴリズム実行
        note over Engine, Backtest: 世代交代を繰り返しながら最適戦略を探索

        loop GAループ (各世代・各個体)
            Engine->>Factory: 遺伝子から動的戦略クラスを生成
            Factory-->>Engine: 戦略クラス
            Engine->>Backtest: 生成された戦略でバックテスト実行
            Backtest->>DB: 過去データを取得
            DB-->>Backtest: OHLCVデータ
            Backtest-->>Engine: パフォーマンス結果
            note right of Engine: 適応度を計算し評価
        end

        Engine-->>Service: 6. GA完了、最良の戦略を返す

        Service->>DB: 7. 最良戦略の「設計図」(遺伝子)を保存
        note over Service, DB: `generated_strategies` テーブル

        Service->>Backtest: 8. 最良戦略で詳細なバックテストを再実行
        Backtest-->>Service: 詳細なパフォーマンス結果
        Service->>DB: 9. 詳細バックテスト結果を保存
        note over Service, DB: `backtest_results` テーブル
    end

    loop 最終結果の確認
        Frontend->>API: GET /experiments/{id}/results
        API->>Service: 保存された結果を取得
        Service-->>API: 最終結果
        API-->>Frontend: 10. 最終結果を表示
    end
```

### フローチャートによる処理概要

```mermaid
%%{init: { 'theme': 'dark', 'themeVariables': {'fontSize': '14px'}}}%%
graph TD
    subgraph "Phase 1: Request"
        A[User starts generation] --> B(Frontend);
        B --> C{API Request};
        C --> D[Service starts async job];
    end

    subgraph "Phase 2: GA Evolution"
        D --> E[GA Engine starts];
        E --> F{GA Loop};
        F -- "Generate & Evaluate" --> G[Create Strategy];
        G --> H[Run Backtest];
        H --> I[Get Market Data];
        I --> H;
        H --> J[Calculate Fitness];
        J --> F;
        F -- "Evolution Complete" --> K[Return Best Strategy];
    end

    subgraph "Phase 3: Save & Display"
        K --> L[Save Strategy Gene];
        L --> M[Run Final Backtest];
        M --> N[Save Full Results];
        N --> O{API Request for Result};
        O --> P[Display Final Result];
        B -.-> O;
    end

    style A fill:#3498DB,stroke:#FFF,stroke-width:2px,color:#FFF
    style P fill:#2ECC71,stroke:#FFF,stroke-width:2px,color:#FFF
    classDef default fill:#444,stroke:#FFF,stroke-width:1px,color:#FFF;
```

## 3. 主要コンポーネントの役割

システムの裏側では、各コンポーネントが専門的な役割を担い、連携して動作します。

- **`AutoStrategyService` (サービス層)**: **司令塔**

  - 戦略生成プロセスの全体を統括する中心的なサービスです。
  - API からのリクエストを受け、バックグラウンドで `GeneticAlgorithmEngine` を起動します。
  - GA 完了後、**二段階の保存処理**を実行します。
    1.  まず、見つかった最良戦略の**遺伝子情報**（インジケーターやルールの組み合わせ）を `generated_strategies` テーブルに保存します。
    2.  次に、同じ最良戦略を用いて再度**詳細なバックテスト**を実行させ、その完全な結果（資産曲線、全取引履歴など）を `backtest_results` テーブルに保存します。

- **`GeneticAlgorithmEngine` (エンジン層)**: **進化の探求者**

  - 遺伝的アルゴリズムのコアロジックを実行するエンジンです。
  - 「評価 → 選択 → 交叉 → 突然変異」というサイクルを繰り返し、世代交代を通じて戦略を進化させ、有望な戦略を探索します。

- **`StrategyFactory` (ファクトリー)**: **動的戦略ビルダー**

  - GA ループの「評価」フェーズで重要な役割を担います。
  - 戦略の遺伝子情報（`StrategyGene`）を受け取り、それを基に `backtesting.py` ライブラリと互換性のある**動的な戦略クラス**をその場で生成します。これにより、遺伝子情報で定義された任意のロジックをバックテストで実行可能にします。

- **`BacktestService` (バックテスト)**: **戦略テスター**

  - `StrategyFactory` によって生成された動的戦略クラスを受け取り、過去の市場データを用いてバックテストを実行します。
  - 戦略のパフォーマンス（総リターン、シャープレシオ、最大ドローダウンなど）を算出します。

- **`FitnessCalculator` (適応度計算機)**: **評価者**

  - `BacktestService` から受け取った複数のパフォーマンス指標を基に、戦略の総合的な優劣を判断するための単一のスコア、すなわち**適応度（Fitness）**を計算します。この適応度が、GA における個体の評価基準となります。

- **データベース層 (Repositories)**: **記録保管庫**
  - `generated_strategies` や `backtest_results` をはじめとする各種テーブルを管理し、データの永続化を担当します。

## 4. 成果物

この一連のプロセスを経て、最終的に以下の 2 つの重要なデータがデータベースに保存され、ユーザーはいつでも参照・活用できます。

1.  **戦略の遺伝子情報 (`generated_strategies` テーブル)**
    - AI が発見した最良戦略の具体的な構成要素（使用インジケーター、パラメータ、売買ロジックなど）です。これは戦略の「設計図」に相当します。
2.  **詳細なバックテスト結果 (`backtest_results` テーブル)**
    - その最良戦略が過去の市場でどのようなパフォーマンスを発揮したかを示す、詳細なレポートです。これには、日々の資産推移、全取引の記録、各種パフォーマンス指標などが含まれます。

これにより、ユーザーは再現性の高い戦略設計と、その信頼性を裏付ける詳細な検証結果の両方を手に入れることができます。
