# Trdinger プロジェクト (`GEMINI.md`)

このドキュメントは、Trdinger プロジェクトの包括的な概要、構造、および開発規約を提供し、AI 主導の開発のための基本的なコンテキストとして機能します。

## プロジェクト概要

Trdinger は、仮想通貨取引戦略を科学的に研究、バックテスト、最適化するためのフルスタックのエンタープライズグレードプラットフォームです。Python バックエンドと TypeScript/Next.js フロントエンドを含むモノレポとして構築されています。

### 主要コンポーネント

- **バックエンド**: コアエンジンとして機能する強力な FastAPI アプリケーションです。以下の機能を担当します。

  - **データサービス**: CCXT ライブラリを介した市場データ（OHLCV、ファンディングレート、建玉）の収集、保存、取得。API を通じてデータの部分的なリセットや初期化を行うことも可能です。
  - **バックテストエンジン**: 履歴データに対して取引戦略を評価し、詳細な統計分析を提供する洗練されたシステム。
  - **機械学習**: モデルトレーニング（例: LightGBM）、評価、特徴量重要度分析のための統合された ML パイプライン。トレーニング済みモデルのリスト取得や削除など、モデルのライフサイクル管理機能も提供します。
  - **遺伝的アルゴリズム**: 遺伝的アルゴリズム（DEAP）を使用して取引ルールを進化させ、最適化する自動戦略発見モジュール。
  - **取引戦略管理**: API を介して、ユーザーが定義した取引戦略の CRUD（作成、読み取り、更新、削除）操作をサポートします。
  - **ハイパーパラメータ最適化**: Optuna を利用して、機械学習モデルや取引戦略のパラメータを最適化するモジュール。
  - **データベース**: データ永続化には `SQLAlchemy` を使用した `PostgreSQL` を採用し、スキーママイグレーションは `Alembic` で管理します。主要なテーブルとして、市場データ（`OHLCVData`, `FundingRateData` など）、バックテスト結果（`BacktestResult`）、そして遺伝的アルゴリズムによる戦略探索の過程と結果（`GAExperiment`, `GeneratedStrategy`）を保存するモデルが定義されています。これにより、実験の再現性と結果の永続化を保証しています。

- **フロントエンド**: Next.js (App Router) と TypeScript で構築されたモダンでレスポンシブな Web インターフェースです。以下の機能を提供します。
  - **インタラクティブダッシュボード**: バックテスト結果、ML モデルのパフォーマンス、市場データの豊富な視覚化（Recharts を使用）。
  - **設定管理**: バックテスト、ML トレーニングパラメータ、データ収集タスクを設定するためのユーザーフレンドリーなフォーム。
  - **UI/UX**: Tailwind CSS と shadcn/ui をベースに、洗練された一貫性のあるユーザーエクスペリエンスを構築しています。コンポーネントは`components`ディレクトリ以下に機能ごとに整理されており、`backtest`、`ml`、`data`などの主要機能に対応するディレクトリと、`common`、`navigation`、`table`などの共有コンポーネント用のディレクトリが存在します。
  - **状態管理**: 状態管理は、主に`useState`や`useReducer`といった React の標準フックと、`hooks`ディレクトリに配置されたカスタムフック群によって実現されています。API の状態（ローディング、エラーなど）を管理する`useApiCall`のように、各機能領域に特化したフックが状態ロジックをカプセル化しています。

## ディレクトリ構造

```text
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
│   │   │   └── optimization/     # パラメータ最適化サービス
│   │   ├── config/        # 設定管理
│   │   └── utils/         # 共通ユーティリティ
│   ├── database/          # SQLAlchemyモデルとリポジトリ
│   ├── tests/             # Pytestテスト
│   ├── main.py            # アプリケーションエントリーポイント
│   └── pyproject.toml     # Python依存関係とツール設定
├── frontend/                # Next.jsフロントエンド
│   ├── app/               # Next.js App Router（ページとレイアウト）
│   ├── components/        # 再利用可能なUIコンポーネント
│   ├── hooks/             # Reactカスタムフック
│   ├── constants/         # 定数
│   ├── types/             # TypeScript型定義
│   ├── tests/             # Jestテスト
│   └── package.json       # Node.js依存関係
└── .github/                 # GitHub Actionsワークフロー
    └── workflows/
        ├── backend-ci.yml
        └── frontend-ci.yml
```

## ビルドと実行

### バックエンドの実行

バックエンドは、**Conda 環境 `trading`** で管理されます。すべてのコマンドは `conda run -n trading` を使用して実行してください。

> **重要**: アクティベーションを忘れてグローバル環境を汚染することを防ぐため、`conda run`パターンを標準としています。

1. **開発サーバーの起動**: アプリケーションは Uvicorn を使用して提供されます。

   ```powershell
   conda run -n trading uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **テストの実行**: テストは `pytest` を使用して実行されます。

   ```powershell
   # すべてのテスト
   conda run -n trading pytest backend/tests/

   # カバレッジ付き
   conda run -n trading pytest --cov=app backend/tests/

   # 文字化け対策（日本語出力が正しく表示されない場合）
   # PowerShellの場合、コマンドの前に環境変数を設定
   $env:PYTHONIOENCODING="utf-8"; conda run -n trading pytest backend/tests/ -v

   # または、セッション全体で設定（最初に一度実行）
   $env:PYTHONIOENCODING="utf-8"
   # その後、通常通りテスト実行
   conda run -n trading pytest backend/tests/
   ```

3. **リンティングとフォーマット**:

   ```powershell
   # フォーマット
   conda run -n trading black backend/app backend/tests
   conda run -n trading isort backend/app backend/tests

   # リンティング
   conda run -n trading flake8 backend/app backend/tests

   # 型チェック
   conda run -n trading mypy backend/app
   ```

### フロントエンドの実行

フロントエンドは、`npm` で管理される標準的な Node.js プロジェクトです。

1. **依存関係のインストール**: `frontend` ディレクトリに移動し、再現性のあるビルドのために `npm ci` を使用して依存関係をインストールします。

   ```bash
   cd frontend
   npm ci
   ```

2. **開発サーバーの起動**:

   ```bash
   npm run dev
   ```

   アプリケーションは `http://localhost:3000` で利用可能になります。

3. **本番ビルド**:

   ```bash
   npm run build
   ```

4. **テストの実行**: テストは Jest を使用して実行されます。

   ```bash
   npm test
   ```

## 開発規約

### バックエンドの開発規約

- **リンティングとフォーマット**: コードベースは、`black`、`isort`、`flake8` を使用して厳密にフォーマットおよびリンティングされています。これらのチェックは CI パイプラインで強制されます。
- **型チェック**: `mypy` を使用して静的型チェックが強制されます。すべての新しいコードは完全に型アノテーションされるべきです。
- **テスト**: テストフレームワークには `pytest` を使用します。テストは `backend/tests/` ディレクトリに、テスト対象のアプリケーション構造を反映させて配置します。
  - **依存性のモック化**: FastAPI の依存性注入（Dependency Injection）メカニズムを積極的に活用します。テスト実行時には `app.dependency_overrides` を使用して、サービス層やデータベースセッション（`get_..._service`, `get_db`）を `unittest.mock` で作成したモック（`Mock`, `AsyncMock`）に差し替えます。これにより、各レイヤーを分離した純粋なユニットテストを実現します。
  - **フィクスチャ**: `pytest.fixture` を用いて、テストクライアント (`TestClient`)、モックオブジェクト、テストデータを一元的に管理・セットアップします。
  - **構成**: `Test...` クラスで関連テストをグループ化し、`@pytest.mark.parametrize` を活用して入力パターンの異なるテストを効率的に記述します。
- **設定**: アプリケーション設定は `app/config/unified_config.py` に集約されています。`pydantic-settings` ライブラリを利用し、階層的なクラスベースで設定が定義されています (`UnifiedConfig`)。設定値は環境変数から読み込まれ、`PARENT__CHILD__ATTRIBUTE` のような形式（例: `ML__TRAINING__LGB_N_ESTIMATORS`）で上書き可能です。`AppConfig`, `DatabaseConfig`, `MLConfig` など、機能ごとに設定クラスが分割されており、各設定項目には合理的なデフォルト値が設定されています。これにより、設定の追加や変更が型安全かつ容易に行えるようになっています。

### フロントエンドの開発規約

- **言語**: すべてのコードに TypeScript が使用されます。
- **リンティングとフォーマット**: `eslint` と `prettier`（Next.js ツールチェーンと統合）がコード品質と一貫性のために使用されます。
- **コンポーネントライブラリ**: UI は、アクセス可能で構成可能な React コンポーネントのセットを提供する `shadcn/ui` を使用して構築されています。アイコンは `lucide-react` から提供されます。
- **スタイリング**: すべてのスタイリングに `Tailwind CSS` が使用されます。
- **テスト**: `jest` と `@testing-library/react` がコンポーネントおよび統合テストに使用されます。テストファイルは、テスト対象のコンポーネントと同じ場所に配置されるか、`frontend/tests/` ディレクトリに配置されます。
- **データフェッチ戦略**: フロントエンドのデータフェッチは、責務が分離された複数のカスタムフックによって構成されています。
  - **`useParameterizedDataFetching`**: パラメータ付きの GET リクエストによるデータ取得、キャッシュ、ローディング/エラー状態の管理といった、読み取り操作の共通ロジックをカプセル化します。
  - **`useApiCall`**: POST, DELETE などのデータ更新を伴う API 呼び出しを実行します。確認ダイアログの表示や成功/エラー時のコールバック処理も担当します。
  - **ドメイン固有フック**: `useBacktestResults` のように、上記の汎用フックを組み合わせて特定の機能（ドメイン）に関する状態管理と操作（例: 結果の選択、削除処理）を実装します。このパターンにより、コードの再利用性を高め、コンポーネントからデータフェッチロジックを分離しています。

### Git & CI/CD

- **CI パイプライン**: GitHub Actions は、バックエンドとフロントエンドの両方 (`.github/workflows/`) に設定されています。パイプラインは、`main` ブランチへのすべてのプッシュおよびプルリクエストで、リンティング、型チェック、テスト、およびビルド検証を自動的に実行します。
- **コミット**: 明示的に強制されていませんが、コミットメッセージは明確で記述的であるべきで、変更の「内容」と「理由」を説明する必要があります。
