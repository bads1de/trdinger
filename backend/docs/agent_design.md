# tradinger AI Agent 設計仕様書

## 1. コンセプト：自律型クオンツ・リサーチ・エージェント
`tradinger` の強力な研究エンジン（Optuna, Backtest, ML）を、AIエージェントが「道具」として自由自在に操り、ユーザーの代わりに分析・検証・報告を行う。

- **役割**: 研究工程のオーケストレーター、MLモデルの解説者、24時間相場監視員。
- **目標**: 「パラメータ調整」をAIに任せ、人間が「戦略の着想」に集中できる環境を作る。

---

## 2. システムアーキテクチャ

`Vibe-Trading` の設計を参考に、既存のサービス層を「ツール」としてラップし、LangGraph で制御する階層構造を採用します。

```text
[ Frontend (Next.js) ]
       ▲  ▼ (SSE: 思考プロセスのリアルタイム配信)
[ API Layer (FastAPI) ]
       ▲  ▼
[ Agent Layer (LangGraph) ] <--- 心臓部
       │
       ├─ [ Agent State ] (短期記憶・履歴管理)
       └─ [ Tool Layer ] (スキルのインターフェース)
              │
              ├─ Backtest Tool  ──> (既存: BacktestService)
              ├─ Optuna Tool    ──> (既存: OptunaService)
              ├─ Data Tool      ──> (既存: DataService)
              └─ ML Analyze Tool ──> (既存: MLService)
```

---

## 3. バックエンド設計 (Python/FastAPI)

### 3.1. ディレクトリ構造の拡張
`trading/backend/app/` 内に `agents/` ディレクトリを新設します。

```text
trading/backend/app/
├── agents/
│   ├── __init__.py
│   ├── core/
│   │   ├── graph.py       # LangGraph の定義（思考のフロー）
│   │   └── state.py       # エージェントの変数（状態）管理
│   ├── tools/
│   │   ├── backtest.py    # BacktestService のラッパー
│   │   ├── optuna.py      # OptunaService のラッパー
│   │   └── data.py        # DataService のラッパー
│   └── provider.py        # LLM (OpenAI/Anthropic) の設定
├── api/v1/
│   └── agent.py           # チャットエンドポイント (SSE対応)
```

### 3.2. 思考プロセス (LangGraph)
`Vibe-Trading` の手法に基づき、エージェントを以下のノードで構成します。
1. **Planner**: 依頼を分析し、どのツールが必要か計画を立てる。
2. **Executor**: 実際に `Backtest` や `Optuna` を実行する。
3. **Analyzer**: 実行結果（数値やグラフデータ）を分析し、目標を達成したか判断する。
4. **Responder**: 最終的なレポートを作成し、ユーザーに回答する。

### 3.3. SSE (Server-Sent Events) 実装
AIの「思考の断片」や「ツールの実行状況」をリアルタイムにフロントエンドへ送ります。
- `event: message`: AIのテキスト回答
- `event: status`: 「今、Optunaを30試行実行しています...」などの進捗
- `event: tool_call`: AIが呼び出したツールの名前と引数

---

## 4. フロントエンド設計 (Next.js/React)

### 4.1. エージェント・コンソール
`trading/frontend/app/agent/page.tsx` を新設します。

- **ChatInterface**: 吹き出し形式のチャット。
- **AgentThoughtBoard**: 
  - `Vibe-Trading` 風のステータス表示。
  - 「思考中...」「データ収集中...」「バックテスト実行中...」といった進捗をプログレスバーやアイコンで表示。
- **InlineResult**: 
  - チャットの中でバックテスト結果の図表（Plotlyなど）を直接レンダリング。

---

## 5. 導入ロードマップ (TDDベース)

### フェーズ 1: 基盤構築 (Infrastructure)
- [ ] `pyproject.toml` への依存関係追加 (`langgraph`, `langchain-openai`, `sse-starlette`)。
- [ ] `app/agents/` の骨組み作成。
- [ ] 最初のツール `get_current_price` (DataService) の実装とテスト。

### フェーズ 2: 思考ループの実装 (Reasoning)
- [ ] LangGraph による基本的な「調査 → 回答」サイクルの構築。
- [ ] API エンドポイント (`/api/v1/agent/chat`) の SSE 対応。
- [ ] フロントエンドでのストリーミング受信。

### フェーズ 3: 研究ツールの統合 (Advanced Tools)
- [ ] `BacktestService` をツールとして統合（AIがバックテストを回せるようになる）。
- [ ] `OptunaService` をツールとして統合（AIが最適化を回せるようになる）。
- [ ] 結果を PDF や Markdown で出力する `ReportingTool` の実装。

### フェーズ 4: 高度な自律化 (Autonomy)
- [ ] **Swarm 構成**: 分析担当と実行担当のエージェントに分割。
- [ ] **MCP Server**: 外部の AI (Cursor等) から `tradinger` の研究機能を呼び出せるようにする。

---

## 6. 技術スタック

| カテゴリ | 技術 | 備考 |
| :--- | :--- | :--- |
| **Orchestrator** | LangGraph | 複雑な条件分岐・ループに最適 |
| **LLM Interface** | LangChain | ツール呼び出しの標準 |
| **Data Validation** | Pydantic v2 | ツールの引数チェックに利用 |
| **API** | FastAPI + sse-starlette | 非同期ストリーミング用 |
| **Frontend** | Next.js + Tailwind CSS | UIコンポーネント |
