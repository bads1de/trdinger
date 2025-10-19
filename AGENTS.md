# AGENTS.md

## プロジェクト概要

Trdinger は、遺伝的アルゴリズム（GA）と機械学習を組み合わせ、自動的に最適な取引戦略を生成する暗号通貨取引戦略自動化システムです。このシステムは FastAPI バックエンドと Next.js フロントエンドで構成され、GA の実装に DEAP ライブラリを使用します。

## リポジトリ構造

```
trading/
├── backend/                 # FastAPIバックエンド
│   ├── app/
│   │   ├── api/           # APIエンドポイント
│   │   ├── services/      # コアサービス
│   │   │   ├── auto_strategy/  # GAコアモジュール
│   │   │   ├── backtest/     # バックテストサービス
│   │   │   ├── data_collection/  # 市場データ収集
│   │   │   ├── indicators/     # テクニカルインジケーター
│   │   │   ├── ml/           # 機械学習サービス
│   │   │   └── optimization/   # 最適化サービス
│   │   └── config/        # 設定管理
│   ├── main.py          # アプリケーションエントリーポイント
│   └── pyproject.toml   # Python依存関係とツール
└── frontend/            # Next.jsフロントエンド
    ├── app/           # ページコンポーネント
    ├── components/    # 再利用可能コンポーネント
    ├── types/       # TypeScript定義
    └── package.json # Node.js依存関係
```

## 開発コマンド

### バックエンド (Python/FastAPI)

```bash
# 開発サーバーを起動
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# テストを実行
python -m pytest                          # 全テスト
python -m pytest tests/test_ga_engine.py -v  # 単体テスト
python -m pytest tests/ -k "regime"        # 特定のテストパターンを実行

# コード品質
black . && isort . && mypy . && flake8 app/

# データベースマイグレーション
alembic upgrade head
```

### フロントエンド (Next.js)

```bash
# 開発サーバーを起動
npm run dev

# 本番ビルド
npm run build

# テストを実行
npm run test
npm run test:watch

# リンター
npm run lint
```

## 主要アーキテクチャコンポーネント

### バックエンドサービス

1. **GA エンジン** (`backend/app/services/auto_strategy/core/ga_engine.py`)

   - `GeneticAlgorithmEngine`: メイン GA 実装
   - `EvolutionRunner`: 単一/多目的最適化実行
   - `IndividualEvaluator`: 個体評価
   - `GeneticOperators`: 交叉と突然変異操作

2. **自動戦略モデル** (`backend/app/services/auto_strategy/models/`)

   - `StrategyGene`: 戦略遺伝子
   - `IndicatorGene`: インジケーター遺伝子
   - `Condition`: 取引条件
   - `TPSLGene`: リスク管理遺伝子

3. **機械学習** (`backend/app/services/ml/`)

   - `BaseMLTrainer`: ML トレーニング基底クラス
   - `MLTrainingService`: トレーニングサービス
   - `SingleModelTrainer`: 単一モデルトレーニング
   - `ModelManager`: モデル管理

4. **設定** (`backend/app/config/unified_config.py`)
   - 階層的設定システム
   - 環境変数サポート
   - サービス固有の設定クラス

### API エンドポイント

- `/api/auto_strategy` - GA 戦略生成
- `/api/backtest` - バックテストサービス
- `/api/data_collection` - 市場データ収集
- `/api/ml_training` - ML モデルトレーニング
- `/api/strategies` - 戦略管理
- `/api/market_data` - 市場データアクセス

### フロントエンドページ

- `/` - 機能概要のホームページ
- `/backtest` - バックテストインターフェース
- `/ml` - ML 管理インターフェース
- `/data` - データ管理インターフェース

## コードスタイルガイドライン

### Python

- **Black**: 行長 88 文字
- **MyPy**: 厳密な型チェック
- **Isort**: インポートソート (profile=black)
- **Flake8**: コードリント
- **Pydantic**: データ検証
- Google スタイルの docstring
- 既知のファーストパーティ: ["app", "backtest", "scripts"]

### TypeScript

- 厳密モード有効
- CamelCase 命名規則
- 型安全性強制
- Radix UI + Tailwind CSS

### 共通ルール

- **インポート**: isort によるアルファベット順。未使用インポートなし。
- **命名**: snake_case (Python)、camelCase (TS)。クラス PascalCase。定数 UPPER_CASE。
- **エラーハンドリング**: 特定の例外での try/except。Pydantic による入力検証。

## テストガイドライン

### バックエンドテスト

- 80%以上のカバレッジ要件で pytest を使用
- 外部サービス（CCXT、市場データ）をモック
- 個別コンポーネントを分離してテスト
- 共通テスト設定にフィクスチャを使用

### フロントエンドテスト

- ユニットテストに Jest
- コンポーネントに React Testing Library
- ブラウザシミュレーションに jsdom 環境

## 設定と設定項目

システムは`backend/app/config/unified_config.py`で階層的設定システムを使用：

- `GAConfig`: 遺伝的アルゴリズムパラメータ
- `AutoStrategyConfig`: 戦略生成設定
- `MLConfig`: 機械学習設定
- `MarketConfig`: 取引所と市場設定
- `BacktestConfig`: バックテストパラメータ

## 重要なファイルとパターン

### 設定管理

- `backend/app/config/unified_config.py` - 統合設定シングルトン
- `__`ネストサポート付き環境変数
- サービス固有の設定クラス

### GA 実装

- `backend/app/services/auto_strategy/core/ga_engine.py` - メイン GA エンジン
- DEAP ツールボックスカスタマイズ
- レジーム対応評価
- 適応度共有実装

### API 構造

- `backend/app/main.py` - FastAPI アプリ作成
- CORS 設定
- グローバル例外処理
- ヘルスチェックエンドポイント

### フロントエンドアーキテクチャ

- App Router 付き Next.js 15
- TypeScript 厳密モード
- Tailwind CSS スタイリング
- Radix UI コンポーネント

## 一般的な開発タスク

### 新しいインジケーターの追加

1. `backend/app/services/indicators/`にインジケーターを作成
2. `IndicatorGene`モデルに追加
3. GA 設定を更新
4. 必要に応じて API エンドポイントを追加

### ML モデルの追加

1. `BaseMLTrainer`クラスを拡張
2. `MLTrainingConfig`で設定
3. モデルマネージャーに追加
4. API エンドポイントを作成

### 取引所の追加

1. 統合設定で`MarketConfig`を更新
2. 取引所固有のロジックを実装
3. サポート取引所リストに追加
4. サンドボックスモードでテスト

## 環境設定

### バックエンド依存関係

```bash
cd backend
pip install -e .
pip install -e .[test,dev]
```

### フロントエンド依存関係

```bash
cd frontend
npm install
```

## Claude ルール (CLAUDE.md より)

- serena を先に使用
- 実装作業に入る前に serena の think を使って思考
- 実装する前に serena のメモリも確認
- 返答は日本語
- わからなければウェブ検索を行い調べてください
- ライブラリは context7 ドキュメントを調べてください
- TDD で開発してください
