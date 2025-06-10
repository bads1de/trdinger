# 自動戦略生成機能 v2 開発計画 (アプローチ B: ランダム+GA)

## 1. 目的

既存の遺伝的アルゴリズム(GA)ベースの自動戦略生成機能を拡張し、新しいデータソースとして Open Interest (OI)と Funding Rate (FR)を戦略の探索空間に組み込む。これにより、より多様でパフォーマンスの高い取引戦略の発見を目指す。

## 2. 全体アーキテクチャ

既存のアーキテクチャをベースに、データ層とビジネスロジック層を拡張する。

```mermaid
graph TD
    subgraph "ユーザーインターフェース"
        A[フロントエンド]
    end

    subgraph "バックエンドAPI"
        B(FastAPI<br>/api/auto-strategy/*)
    end

    subgraph "ビジネスロジック (AutoStrategyService)"
        C{AutoStrategyService}
        D[GAエンジン<br>(DEAP・並列計算)]
        E[戦略ファクトリー]
        F[バックテストサービス<br>(backtesting.py)]
        NEW_G[<font color=green>BacktestDataService (拡張)</font>]
    end

    subgraph "データ層"
        H[OHLCV Repository]
        NEW_I[<font color=green>OpenInterest Repository</font>]
        NEW_J[<font color=green>FundingRate Repository</font>]
    end

    A -- APIリクエスト --> B
    B -- 処理要求 --> C
    C -- GA実行指示 --> D
    D -- 個体評価要求 --> E
    E -- 動的戦略生成 --> F
    F -- データ要求 --> NEW_G
    NEW_G -- OHLCVデータ --> H
    NEW_G -- OIデータ --> NEW_I
    NEW_G -- FRデータ --> NEW_J
    F -- バックテスト結果 --> D
    D -- フィットネス値 --> C
    C -- 進捗・結果 --> B
    B -- APIレスポンス --> A
```

## 3. 開発フェーズ

### フェーズ 1: データ層とサービス層の拡張

**目的:** OI と FR のデータをバックテストで利用可能にするための基盤を整備する。

**ステップ 1.1: `BacktestDataService` の機能拡張**

- **ファイル:** `backend/app/core/services/backtest_data_service.py`
- **タスク:**
  1. `__init__` を改修し、`OpenInterestRepository` と `FundingRateRepository` を受け取れるようにする。
  2. `get_ohlcv_for_backtest` を `get_data_for_backtest` のような汎用的な名前に変更する。
  3. 新メソッド内で、指定期間の OHLCV, OI, FR データをそれぞれのリポジトリから取得する。
  4. `pd.merge_asof` を使用して、OI と FR のデータを OHLCV の DataFrame にタイムスタンプを基準にマージする。
  5. 最終的に、`Open`, `High`, `Low`, `Close`, `Volume` に加え、`OpenInterest` と `FundingRate` カラムを持つ DataFrame を返すようにする。

**ステップ 1.2: `StrategyFactory` でのデータカラム名の標準化**

- **ファイル:** `backend/app/core/services/auto_strategy/factories/strategy_factory.py`
- **タスク:**
  - `backtesting.py` の `Strategy` クラス内で `self.data.OpenInterest` や `self.data.FundingRate` としてアクセスできるよう、カラム名をここで定義・管理する。

### フェーズ 2: 遺伝子モデルと GA エンジンの拡張

**目的:** 拡張されたデータを実際に戦略の条件として組み込めるように、遺伝子の表現力と生成ロジックを改修する。

**ステップ 2.1: `StrategyGene` モデルの拡張**

- **ファイル:** `backend/app/core/services/auto_strategy/models/strategy_gene.py`
- **タスク:**
  1. **新しいオペランドの許容:** `Condition` クラスの `left_operand` と `right_operand` で、`'OpenInterest'` や `'FundingRate'` といった新しい文字列を許容する。
  2. **新しい指標タイプの追加:** `IndicatorGene` で、OI や FR 自体を一種の「指標」として扱えるようにする (例: `type: 'OpenInterest'`)。これにより、これらのデータに対する移動平均（例：`OIの20期間SMA`）の計算などが可能になる。

**ステップ 2.2: ランダム遺伝子生成ロジックの作成**

- **ファイル:** `backend/app/core/services/auto_strategy/engines/ga_engine.py` (または新規モジュール `gene_generator.py`)
- **タスク:**
  1. `decode_list_to_gene` に代わる、新しいランダム遺伝子生成関数を作成する。
  2. 利用可能な全ての指標（SMA, RSI 等）と新しいデータソース（Price, OI, FR）の中からランダムにオペランドを選択するロジックを実装する。
  3. 比較演算子もランダムに選択する。
  4. 右辺のオペランドとして、別の指標/データソース、または妥当な範囲の固定値をランダムに選択する。
  5. これにより、多様な条件を持つ遺伝子を生成する。

**ステップ 2.3: GA エンジンの適合**

- **ファイル:** `backend/app/core/services/auto_strategy/engines/ga_engine.py`
- **タスク:**
  - GA の初期個体群を、ステップ 2.2 で作成した新しいランダム遺伝子生成関数を使って生成するように変更する。
  - `encode`/`decode` の仕組みは、より表現力の高い JSON ベースの遺伝子表現を直接扱うように変更するか、この機会に廃止し、GA ライブラリ（DEAP）が直接 `StrategyGene` オブジェクトを扱えるように改修することを検討する。
