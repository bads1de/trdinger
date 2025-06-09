# 自動ストラテジー生成機能 実装計画書

## 1. はじめに

本ドキュメントは、「自動ストラテジー生成機能」の最終的な実装計画を定義するものです。遺伝的アルゴリズム（GA）ライブラリ`DEAP`と既存のバックテストシステムを統合し、データ駆動で取引戦略を自動生成する機能を実現します。

これまでの設計と調査（既存コードベース、DEAP ドキュメント、Web 検索）の結果、および最終的なダブルチェックで明らかになったリスクへの対策をすべて反映しています。

## 2. アーキテクチャ概要

基本的なアーキテクチャは初期設計を踏襲します。フロントエンドからのリクエストを受け、API 層、ビジネスロジック層を経て、既存のバックテストシステムと連携します。

```mermaid
graph TD
    A[フロントエンド<br>(OptimizationModal)] -->|API Request| B(API層<br>/api/auto-strategy/generate)
    B --> C{ビジネスロジック層}
    C --> D[GAエンジン<br>(DEAP Toolbox)]
    C --> E[戦略ファクトリー]
    D -- 評価要求 --> E
    E -- 動的戦略クラス生成 --> F[バックテストサービス<br>(backtesting.py)]
    F -- 指標計算 --> G[TALibアダプター]
    F -- 結果 --> D
    D -- 進捗/結果 --> B
    B -->|Progress/Result| A
```

## 3. 主要コンポーネント設計

### 3.1. GeneticAlgorithmEngine

- 内部的に`DEAP`の`Toolbox`を活用し、評価、選択、交叉、突然変異の各操作を管理します。
- `StrategyGene`オブジェクトと`DEAP`が扱う数値リスト間のエンコード/デコードロジックを内包します。
- 評価関数（`evaluate`）は、`DEAP`の`toolbox.map`機能を利用して複数のバックテストを並列実行し、パフォーマンスを最大化します。

### 3.2. StrategyFactory

- `GAEngine`からデコード済みの`StrategyGene`を受け取ります。
- `StrategyGene`の定義に基づき、`backtesting.py`が実行可能な`Strategy`クラスを動的に生成します。
- 指標計算には既存の`TALibAdapter`を全面的に利用します。

### 3.3. 制約管理マネージャー

- `GAEngine`内に設置し、無効な遺伝子の生成・評価を防ぎます。
- `DEAP`のデコレータ機能を活用し、交叉・突然変異の操作後に制約チェック関数を自動的に呼び出します。
- **チェック項目例**:
  - パラメータ間の大小関係（例: `sma_short_period < sma_long_period`）
  - パラメータの有効範囲（例: `rsi_period > 1`）

## 4. 遺伝子エンコード/デコード仕様 (v1)

遺伝子の表現は本機能の核となるため、初期バージョン（v1）では仕様を単純化し、リスクを低減します。

- **基本方針**: `StrategyGene`オブジェクトを、`DEAP`で扱いやすい**固定長の数値リスト**に変換します。
- **v1 仕様**:
  - **最大指標数**: 5 つ
  - **売買条件**: エントリー・イグジットそれぞれ 1 組の単純な比較（`指標A > 指標B`）のみ。AND/OR などの論理結合は v2 以降で対応。
  - **エンコード形式（例）**:
    - `[indicator1_id, param1_val, indicator2_id, param2_val, ..., entry_idx_A, entry_op, entry_idx_B, exit_idx_A, exit_op, exit_idx_B]`
    - `indicator_id`: 24 種類の既存指標に対応する ID (0-23)。0 は不使用を示す。
    - `param_val`: 正規化されたパラメータ値 (0.0-1.0)。評価時に実際の範囲に変換。
    - `entry_idx_A`: エントリー条件左辺の指標インデックス (0-4)。
    - `entry_op`: 比較演算子 ID (例: 0=`>`, 1=`<`)。

このエンコード/デコードロジックの厳密な仕様策定と実装を、開発の最優先タスクとします。

## 5. API 設計

初期設計に基づき、以下の API エンドポイントを実装します。

- `POST /api/auto-strategy/generate`: GA による戦略生成タスクを開始します。
- `GET /api/auto-strategy/experiments/{experiment_id}/progress`: 特定の実験の進捗状況を返します。
- `GET /api/auto-strategy/experiments/{experiment_id}/results`: 完了した実験の結果一覧を返します。

## 6. データベース設計

既存の SQLite データベースとの互換性を確保するため、以下のスキーマでテーブルを実装します。

```sql
-- 自動生成された戦略の遺伝子情報
CREATE TABLE generated_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    gene_data TEXT NOT NULL,          -- 戦略遺伝子 (JSON)
    generation INTEGER NOT NULL,
    fitness_score REAL,
    FOREIGN KEY (experiment_id) REFERENCES ga_experiments(id)
);

-- GAの実験情報
CREATE TABLE ga_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    config TEXT NOT NULL,             -- GA設定 (JSON)
    status TEXT DEFAULT 'running',
    progress REAL DEFAULT 0.0,
    best_fitness REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);
```

## 7. フロントエンド実装

- 既存の`frontend/components/backtest/OptimizationModal.tsx`に**「自動生成 (GA)」タブを追加**します。
- GA 設定用のフォームコンポーネントを新規作成します。
- 進捗表示は、`useApiCall`フックをラップした新しい`useGAProgress`フックを作成し、進捗監視 API を定期的にポーリングしてリアルタイムに UI を更新します。

## 8. 実装計画とフェーズ分け

リスク管理と段階的な機能投入のため、以下のフェーズで開発を進めます。

- **フェーズ 0: パフォーマンス・ベースライン測定 (先行タスク)**

  - **内容**: 既存の`BacktestService`を使い、代表的な戦略（SMA クロス等）の単一バックテスト実行時間を複数期間で計測。
  - **目的**: GA 全体の実行時間を現実的に見積もり、ユーザーに提示するパフォーマンス期待値を設定する。

- **フェーズ 1: 基盤実装**

  - **内容**:
    - 遺伝子エンコード/デコード仕様(v1)の厳密な策定と実装。
    - `GAEngine`と`StrategyFactory`のプロトタイプを作成。
    - 制約管理マネージャーの基本機能を実装。
  - **ゴール**: 単純な遺伝子から戦略を生成し、単一のバックテストが実行できることを確認する。

- **フェーズ 2: バックテスト統合と並列化**

  - **内容**:
    - `GAEngine`と`BacktestService`を完全に統合。
    - `DEAP`の`toolbox.map`を用いて、バックテスト評価の並列処理を実装。
  - **ゴール**: 複数の個体からなる 1 世代分の評価が、並列で高速に実行されることを確認する。

- **フェーズ 3: API とフロントエンド**

  - **内容**:
    - バックエンド API（`/generate`, `/progress`, `/results`）を実装。
    - フロントエンドの`OptimizationModal`に GA タブと設定フォームを実装。
    - `useGAProgress`フックを実装し、進捗表示を実現。
  - **ゴール**: UI から GA を実行し、進捗を確認し、最終結果を表示できる。

- **フェーズ 4: 高度機能 (将来拡張)**
  - **内容**:
    - 多目的最適化（NSGA-II）の実装。
    - 遺伝子表現の拡張（AND/OR 条件など）。
    - 非同期タスク管理（キャンセル機能、完了通知）。
  - **目的**: より高度で実用的な戦略探索と、UX の向上。

## 9. リスクと対策

- **リスク 1: パフォーマンスの悪化**
  - **内容**: GA の計算量が想定を上回り、ユーザーが許容できない実行時間になる可能性。
  - **対策**: **フェーズ 0**で先行してパフォーマンスを測定し、現実的な見積もりを行う。初期バージョンでは個体数や世代数に上限を設ける。
- **リスク 2: エンコード/デコードロジックの複雑化**
  - **内容**: 戦略が複雑になるにつれ、エンコード/デコードのロジックが保守困難になる可能性。
  - **対策**: **v1 仕様**で機能を単純なものに限定する。ロジックには十分な単体テストを記述し、段階的に拡張する。

以上
