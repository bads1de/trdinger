# 技術スタック

## バックエンド (Python)
- **フレームワーク**: FastAPI with Uvicorn ASGIサーバー
- **データベース**: PostgreSQL with SQLAlchemy ORM and Alembicマイグレーション
- **取引**: マルチ取引所サポートのためのCCXTライブラリ
- **ML/分析**: pandas, numpy, scikit-learn, LightGBM, joblib
- **バックテスト**: 戦略検証のためのbacktestingライブラリ
- **最適化**: 遺伝的アルゴリズムのためのDEAP、ベイズ最適化のためのscikit-optimize
- **環境管理**: 設定管理のためのpython-dotenv
- **テスト**: asyncioサポート付きpytest
- **コード品質**: black（フォーマット）、isort（インポート）、flake8（リント）、mypy（型チェック）

## フロントエンド (TypeScript/React)
- **フレームワーク**: Next.js 15 with TypeScript
- **UIコンポーネント**: カスタムデザインシステム付きRadix UIプリミティブ
- **スタイリング**: カスタムエンタープライズテーマ付きTailwind CSS
- **フォーム**: Zodバリデーション付きReact Hook Form
- **チャート**: データ可視化のためのRecharts
- **状態管理**: Reactフックとコンテキスト
- **テスト**: React Testing Library付きJest
- **ビルド**: Next.js組み込みバンドラーと最適化

## 開発ツール
- **Pythonバージョン**: 3.10以上が必要
- **Nodeバージョン**: Next.js 15と互換性あり
- **パッケージ管理**: Python用pip/setuptools、Node.js用npm
- **データベースマイグレーション**: スキーマ管理のためのAlembic

## よく使うコマンド

### バックエンド開発
```bash
# 仮想環境のセットアップ
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix

# 依存関係のインストール
pip install -r requirements.txt

# 開発サーバーの実行
python main.py

# テストの実行
pytest

# コードフォーマットとリント
black .
isort .
flake8 .
mypy .

# データベースマイグレーション
alembic upgrade head
alembic revision --autogenerate -m "説明"
```

### フロントエンド開発
```bash
# 依存関係のインストール
npm install

# 開発サーバーの実行
npm run dev

# 本番用ビルド
npm run build
npm start

# テストの実行
npm test
npm run test:watch

# リント
npm run lint
```

## 設定
- バックエンド設定はコード内の設定ファイルで管理
- フロントエンド設定は`next.config.js`内
- データベース接続はSQLAlchemy設定経由
- 取引APIキーはコード内設定で管理