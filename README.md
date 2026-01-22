# Trdinger: Advanced Algo-Trading & Research Platform

<!-- TODO: Update these badges with your actual repository URL -->
<!-- [![Backend CI](https://github.com/username/trading/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/username/trading/actions/workflows/backend-ci.yml) -->
<!-- [![Frontend CI](https://github.com/username/trading/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/username/trading/actions/workflows/frontend-ci.yml) -->

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-blue)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black)](https://nextjs.org/)

Trdingerは、仮想通貨市場における取引戦略の**科学的な研究、バックテスト、そして自動生成**を行うためのエンタープライズグレードのプラットフォームです。
単なる自動売買ボットではなく、遺伝的アルゴリズムによる戦略探索や、高度な機械学習モデルを用いた市場予測を統合した**「戦略の研究開発ラボ」**として設計されています。

## 🚀 Key Features

### 1. 🧬 Evolutionary Strategy Discovery (遺伝的アルゴリズム)

遺伝的アルゴリズム (Genetic Algorithm) を用いて、収益性の高い取引ルールを**自動的に進化・生成**します。

- **Engine**: `DEAP` フレームワークを採用し、並列処理による高速な評価を実現。
- **Optimization**: 多目的最適化 (NSGA-II) により、「収益性」と「リスク（ドローダウン）」のバランスが取れたパレート最適な戦略群を探索します。
- **Gene Structure**: テクニカル指標、エントリー/エグジット条件、リスク管理（TP/SL）を遺伝子として表現し、柔軟かつ複雑な戦略の生成が可能です。

### 2. 🤖 Advanced Machine Learning Pipeline

LightGBM や XGBoost などの勾配ブースティング決定木 (GBDT)、および深層学習モデル（RNN等）をサポートする強力な予測パイプラインを搭載しています。

- **Feature Engineering**: 分数次微分 (Fractional Differentiation)、Kyle's Lambda などのマイクロストラクチャ特徴量、各種テクニカル指標を組み合わせた高度な特徴量生成ロジック (`FeatureEngineeringService`)。
- **Evaluation**: 時系列データの特性を考慮した Walk-Forward Validation や Purged K-Fold CV をサポートし、オーバーフィッティングを防ぎます。

### 3. 📊 High-Performance Backtesting Engine

- **Speed**: Pandas/Numpy を駆使したベクトル化演算とデータの最適化（dtype調整、キャッシング）により、長期間のティックデータに対するバックテストを高速に実行します。
- **Reality Check**: スリッページ、取引手数料、市場の流動性を厳密に考慮したシミュレーションを行い、実運用に近いパフォーマンス評価を提供します。

### 4. 🖥️ Modern & Interactive Dashboard

- **Tech**: Next.js (App Router) + TypeScript + Tailwind CSS + shadcn/ui。
- **UX**: 複雑な設定や膨大なバックテスト結果、MLモデルの学習状況を直感的に操作・可視化できるレスポンシブなインターフェースを提供します。
- **Visualization**: Recharts を用いたインタラクティブなチャートにより、資産推移やポジション状況を詳細に分析可能です。

## 🛠️ Architecture & Tech Stack

本プロジェクトは、**保守性、テスト容易性、拡張性**を重視したレイヤードアーキテクチャを採用しています。

### Backend (`/backend`)

- **Framework**: FastAPI (Asynchronous Python Web Framework)
- **Design Pattern**:
  - **Dependency Injection (DI)**: FastAPIのDIシステムをフル活用し、サービス層とリポジトリ層を疎結合に保つことで、ユニットテストの容易性を確保しています。
  - **Repository Pattern**: データアクセスロジックを抽象化し、ビジネスロジックから分離。
- **Database**: PostgreSQL + SQLAlchemy (ORM) + Alembic (Migration).
- **Quality Assurance**:
  - **Testing**: Pytest による広範なユニット/統合テスト。`unittest.mock` を活用して外部依存（DB、API）を分離し、純粋なロジックの検証を行っています。
  - **Linting/Type**: Ruff, Black, isort, mypy (Strict mode) による厳格なコード品質チェックをCIで強制。

### Frontend (`/frontend`)

- **Framework**: Next.js 15 (App Router)
- **State Management**: ドメインごとに責務を分割した Custom Hooks (`useAutoStrategy`, `useMLTraining` 等) により、コンポーネントとロジックを分離。
- **UI Components**: `shadcn/ui` をベースにしたアクセシビリティ対応のコンポーネント設計。
- **Quality Assurance**: Jest + React Testing Library によるコンポーネントテスト。

## 📦 Directory Structure

```text
trading/
├── backend/                 # Python/FastAPI Backend
│   ├── app/
│   │   ├── api/           # API Endpoints (Routing & DI)
│   │   ├── services/      # Core Business Logic
│   │   │   ├── auto_strategy/ # Genetic Algorithm Engine (DEAP based)
│   │   │   ├── ml/            # ML Pipeline (Feature Engineering, Training)
│   │   │   ├── backtest/      # Backtesting Engine
│   │   │   └── ...
│   │   └── ...
│   ├── database/          # SQLAlchemy Models & Migrations
│   └── tests/             # Pytest Suites
├── frontend/                # Next.js Frontend
│   ├── app/               # Pages & Layouts (App Router)
│   ├── components/        # UI Components (Atomic design inspired)
│   ├── hooks/             # Custom React Hooks (Business Logic)
│   └── ...
└── .github/                 # CI/CD Workflows
```

## 💻 Getting Started

### Prerequisites

- Docker & Docker Compose (Optional but recommended for DB)

- Python 3.10+

- Node.js 18.17+

### Setup Guide

#### 1. Backend Setup

本プロジェクトは Conda 環境での実行を推奨しています。

```bash

# Create environment

conda create -n trading python=3.10

conda activate trading



# Install dependencies & tools

cd backend

pip install -e .[dev,test]



# Run DB Migrations

alembic upgrade head



# Start Development Server

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```

#### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm ci

# Start Development Server
npm run dev
```

アプリケーションは `http://localhost:3000` で利用可能になります。

## 🧪 Testing & Quality Checks

品質担保のため、以下のコマンドでテストと静的解析を実行できます。

```bash
# Backend Testing & Linting
pytest backend/tests
mypy backend/app
flake8 backend/app

# Frontend Testing
npm test
npm run lint
```

---

_Created for portfolio demonstration purposes._
