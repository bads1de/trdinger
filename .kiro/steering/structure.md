# プロジェクト構造

## ルートディレクトリレイアウト
```
trdinger/
├── backend/           # Python FastAPIバックエンド
├── frontend/          # Next.js Reactフロントエンド
├── venv/             # Python仮想環境
├── .git/             # Gitリポジトリ
├── .kiro/            # Kiro IDE設定
└── .gitignore        # Git無視ルール
```

## バックエンド構造 (`backend/`)
```
backend/
├── app/              # メインアプリケーションパッケージ
│   ├── api/          # APIルートハンドラー
│   ├── config/       # 設定管理
│   ├── core/         # コアビジネスロジック
│   └── main.py       # FastAPIアプリケーションファクトリ
├── database/         # データベーススキーマとマイグレーション
├── data_collector/   # 市場データ収集モジュール
├── models/           # SQLAlchemyデータベースモデル
├── scripts/          # ユーティリティとメンテナンススクリプト
├── tests/            # テストスイート
├── docs/             # APIドキュメント
├── main.py           # アプリケーションエントリーポイント
├── requirements.txt  # Python依存関係
├── pyproject.toml    # Pythonプロジェクト設定
└── trdinger.db       # SQLiteデータベース（開発用）
```

## フロントエンド構造 (`frontend/`)
```
frontend/
├── app/              # Next.js App Routerページ
│   ├── api/          # APIルートハンドラー
│   ├── backtest/     # バックテストインターフェース
│   ├── data/         # データ管理ページ
│   ├── ml/           # MLモデル管理
│   ├── layout.tsx    # ルートレイアウトコンポーネント
│   ├── page.tsx      # ホームページ
│   └── globals.css   # グローバルスタイル
├── components/       # 再利用可能なUIコンポーネント
├── hooks/            # カスタムReactフック
├── lib/              # ユーティリティライブラリ
├── types/            # TypeScript型定義
├── utils/            # ヘルパー関数
├── constants/        # アプリケーション定数
├── __tests__/        # テストファイル
├── package.json      # Node.js依存関係
├── next.config.js    # Next.js設定
├── tailwind.config.js # Tailwind CSS設定
└── tsconfig.json     # TypeScript設定
```

## 主要なアーキテクチャパターン

### バックエンドパターン
- **レイヤードアーキテクチャ**: API → コアロジック → データベース
- **依存性注入**: FastAPIの組み込みDIシステム
- **リポジトリパターン**: データベースアクセスの抽象化
- **サービス層**: ビジネスロジックの分離
- **設定管理**: 環境ベースの設定

### フロントエンドパターン
- **コンポーネントベースアーキテクチャ**: 再利用可能なUIコンポーネント
- **カスタムフック**: 共有ステートフルロジック
- **型安全API**: バックエンド通信のためのTypeScriptインターフェース
- **デザインシステム**: Radix + Tailwindによる一貫したUI
- **ファイルベースルーティング**: Next.js App Routerの規約

## 命名規則
- **Python**: 関数/変数はsnake_case、クラスはPascalCase
- **TypeScript**: 関数/変数はcamelCase、コンポーネント/型はPascalCase
- **ファイル**: コンポーネントファイルはkebab-case、Pythonモジュールはsnake_case
- **ディレクトリ**: 適切に小文字でハイフンまたはアンダースコア

## インポート整理
- **Python**: 標準ライブラリ → サードパーティ → ローカルインポート
- **TypeScript**: 外部ライブラリ → 内部ユーティリティ → コンポーネント → 型

## テスト構造
- **バックエンド**: `tests/`ディレクトリ内のユニットテスト、アプリ構造をミラー
- **フロントエンド**: `__tests__/`ディレクトリ内のコンポーネントテスト、コンポーネントと同じ場所に配置
- **統合**: 重要なユーザーフローのエンドツーエンドテスト