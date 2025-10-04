# AGENTS.md

.最初の指示で serena を使ってください
.実装作業に入る前に serena の think を使って思考してください.
.実装する前に serena のメモリも確認してください
.返答は日本語
.わからなければウェブ検索を行い調べてください
.ライブラリは context7 ドキュメントを調べてください
.TDD で開発してください

## プロジェクト概要

Trdinger は暗号通貨取引のための遺伝的アルゴリズム自動戦略生成システムです。FastAPI バックエンドと React フロントエンドで構成され、DEAP ライブラリを使用した GA と機械学習を組み合わせて最適な取引戦略を自動生成します。

## 開発環境

### Backend (Python)

```bash
# 開発サーバー
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# テスト
python -m pytest
python -m pytest tests/test_ga_engine.py -v

# コード品質
black .
isort .
mypy .
flake8 app/

# データベース
alembic upgrade head
```

### Frontend (Node.js)

```bash
# 開発サーバー
npm run dev

# ビルド
npm run build

# テスト
npm run test
npm run test:watch

# Linting
npm run lint
```

## アーキテクチャ概要

### Backend 構造

```
backend/
├── app/services/auto_strategy/     # GAコアモジュール
│   ├── core/                  # GAエンジン、遺伝的演算子
│   ├── config/               # 設定クラス群
│   ├── models/               # 遺伝子データモデル
│   ├── services/             # 実行サービス
│   └── ml/                 # 機械学習統合
├── app/services/ml/          # ML基盤
│   ├── base_ml_trainer.py     # 学習基盤クラス
│   ├── ml_training_service.py # 学習サービス
│   └── model_manager.py      # モデル管理
└── app/api/                # APIエンドポイント
    └── auto_strategy.py      # GA生成API
```

### Frontend 構造

```
frontend/
├── components/backtest/
│   └── GAConfigForm.tsx      # GA設定フォーム
├── types/
│   ├── auto-strategy.ts      # GA関連型定義
│   └── backtest.ts         # バックテスト型
└── app/api/               # APIルート
```

## 主要コンポーネント

### GA エンジン (backend/app/services/auto_strategy/core/ga_engine.py)

- `GeneticAlgorithmEngine`: 主要 GA 実装
- `EvolutionRunner`: 単一/多目的最適化実行
- `IndividualEvaluator`: 個体評価
- `GeneticOperators`: 交叉・突然変異

### 遺伝子モデル (backend/app/services/auto_strategy/models/)

- `StrategyGene`: 戦略遺伝子
- `IndicatorGene`: インジケータ遺伝子
- `Condition`: 取引条件
- `TPSLGene`: 損益管理遺伝子

### 機械学習統合 (backend/app/services/ml/)

- `BaseMLTrainer`: ML 学習基盤
- `MLTrainingService`: 学習サービス
- `SingleModelTrainer`: 単一モデル学習
- `ModelManager`: モデル管理

## コード品質要件

- **Python**: mypy 厳格型チェック、black 行長 88、Google スタイル docstring
- **TypeScript**: strict モード、camelCase 命名
- **テスト**: pytest カバレッジ 80%以上
- **GA 特有**: Pydantic 設定管理、DEAP ツールボックス使用

## 開発パターン

### GA 関連

- `GAConfig`で設定管理
- `GeneSerializer`で遺伝子のシリアライズ/デシリアライズ
- `safe_operation`デコレータでエラーハンドリング
- DEAP ツールボックスのカスタマイズ

### ML 統合

- `BaseMLTrainer`のテンプレートメソッドパターン
- `safe_ml_operation`でのエラー耐性
- `ModelMetadata`によるメタデータ管理
- `AutoMLFeatureGenerationService`での特徴量生成

## 重要な設定ファイル

- `backend/pyproject.toml`: Python 依存関係とツール設定
- `frontend/package.json`: Node.js 設定
- `backend/app/services/auto_strategy/config/`: GA 設定クラス群

## 常用コマンド

```bash
# GAエンジンテスト
python -m pytest tests/test_ga_engine.py -v

# ML学習テスト
python -m pytest ml/tests/ -v

# 遺伝的演算子テスト
python -m pytest tests/test_genetic_operators.py -v

# バックテスト統合テスト
python -m pytest tests/test_backtest_service.py -v

# 型チェック
mypy app/services/auto_strategy/
mypy app/services/ml/

# GA設定検証
python -c "from app.services.auto_strategy.config import GAConfig; print(GAConfig())"
```
