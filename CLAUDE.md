# CLAUDE.md

このファイルは、このリポジトリのコードを操作する際にClaude Code (claude.ai/code)にガイダンスを提供します。

## プロジェクト概要

Trdingerは、遺伝的アルゴリズム（GA）と機械学習を組み合わせ、自動的に最適な取引戦略を生成する暗号通貨取引戦略自動化システムです。このシステムはFastAPIバックエンドとReactフロントエンドで構成され、GAの実装にDEAPライブラリを使用します。

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

### バックエンド (Python)

```bash
# 開発サーバーを起動
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# テストを実行
python -m pytest
python -m pytest tests/test_ga_engine.py -v
python -m pytest tests/ -k "regime"  # 特定のテストパターンを実行

# コード品質
black .
isort .
mypy .
flake8 app/

# データベースマイグレーション
alembic upgrade head
```

### フロントエンド (Node.js)

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

1. **GAエンジン** (`backend/app/services/auto_strategy/core/ga_engine.py`)

   - `GeneticAlgorithmEngine`: メインGA実装
   - `EvolutionRunner`: 単一/多目的最適化実行
   - `IndividualEvaluator`: 個体評価
   - `GeneticOperators`: 交叉と突然変異操作

2. **自動戦略モデル** (`backend/app/services/auto_strategy/models/`)

   - `StrategyGene`: 戦略遺伝子
   - `IndicatorGene`: インジケーター遺伝子
   - `Condition`: 取引条件
   - `TPSLGene`: リスク管理遺伝子

3. **機械学習** (`backend/app/services/ml/`)

   - `BaseMLTrainer`: MLトレーニング基底クラス
   - `MLTrainingService`: トレーニングサービス
   - `SingleModelTrainer`: 単一モデルトレーニング
   - `ModelManager`: モデル管理

4. **設定** (`backend/app/config/unified_config.py`)
   - 階層的設定システム
   - 環境変数サポート
   - サービス固有の設定クラス

### APIエンドポイント

- `/api/auto_strategy` - GA戦略生成
- `/api/backtest` - バックテストサービス
- `/api/data_collection` - 市場データ収集
- `/api/ml_training` - MLモデルトレーニング
- `/api/strategies` - 戦略管理
- `/api/market_data` - 市場データアクセス

### フロントエンドページ

- `/` - 機能概要のホームページ
- `/backtest` - バックテストインターフェース
- `/ml` - ML管理インターフェース
- `/data` - データ管理インターフェース

## 設定と設定項目

システムは`backend/app/config/unified_config.py`で階層的設定システムを使用します：

- `GAConfig`: 遺伝的アルゴリズムパラメータ
- `AutoStrategyConfig`: 戦略生成設定
- `MLConfig`: 機械学習設定
- `MarketConfig`: 取引所と市場設定
- `BacktestConfig`: バックテストパラメータ

## テストガイドライン

### バックエンドテスト

- 80%以上のカバレッジ要件でpytestを使用
- 外部サービス（CCXT、市場データ）をモック
- 個別コンポーネントを分離してテスト
- 共通テスト設定にフィクスチャを使用

### フロントエンドテスト

- ユニットテストにJest
- コンポーネントにReact Testing Library
- ブラウザシミュレーションにjsdom環境

## コード品質要件

### Python

- **Black**: 行長88文字
- **MyPy**: 厳密な型チェック
- **Isort**: インポートソート
- **Flake8**: コードリント
- **Pydantic**: データ検証
- Googleスタイルのdocstring

### TypeScript

- 厳密モード有効
- CamelCase命名規則
- 型安全性強制

## 重要なファイルとパターン

### 設定管理

- `backend/app/config/unified_config.py:403` - 統合設定シングルトン
- `__`ネストサポート付き環境変数
- サービス固有の設定クラス

### GA実装

- `backend/app/services/auto_strategy/core/ga_engine.py` - メインGAエンジン
- DEAPツールボックスカスタマイズ
- レジーム対応評価
- 適応度共有実装

### API構造

- `backend/app/main.py:65` - FastAPIアプリ作成
- CORS設定
- グローバル例外処理
- ヘルスチェックエンドポイント

### フロントエンドアーキテクチャ

- App Router付きNext.js 15
- TypeScript厳密モード
- Tailwind CSSスタイリング
- Radix UIコンポーネント

## 一般的な開発タスク

### 新しいインジケーターの追加

1. `backend/app/services/indicators/`にインジケーターを作成
2. `IndicatorGene`モデルに追加
3. GA設定を更新
4. 必要に応じてAPIエンドポイントを追加

### MLモデルの追加

1. `BaseMLTrainer`クラスを拡張
2. `MLTrainingConfig`で設定
3. モデルマネージャーに追加
4. APIエンドポイントを作成

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

## データベースとデータ

- Alembicマイグレーション付きPostgreSQLデータベース
- OHLCV形式で保存された市場データ
- 戦略とモデルメタデータ保存
- バックテスト結果永続化

## ログと監視

- 重複フィルタリング付き構造化ログ
- サービスごとのログレベル
- エラートラッキングとメトリクス
- メモリ使用量監視

## パフォーマンス考慮事項

- 大規模データセットのメモリ管理
- 効率的なGA人口処理
- 市場データのキャッシング
- 並列バックテスト実行

## セキュリティノート

- 環境変数設定
- API認証準備済み
- Pydanticによる入力検証
- 設定での安全なデフォルト