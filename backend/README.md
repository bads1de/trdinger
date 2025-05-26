# Trdinger Trading API Backend

CCXT ライブラリを使用した仮想通貨取引データAPIのバックエンドサービスです。

## 🏗️ アーキテクチャ

```
backend/
├── app/                    # アプリケーションコア
│   ├── main.py            # FastAPIアプリケーション
│   ├── config/            # 設定管理
│   │   ├── settings.py    # 環境設定
│   │   └── market_config.py # 市場データ設定
│   ├── api/               # APIルーター
│   │   └── v1/
│   │       └── market_data.py
│   ├── core/              # ビジネスロジック
│   │   ├── services/      # サービス層
│   │   │   └── market_data_service.py
│   │   └── models/        # ビジネスモデル
│   └── database/          # データベース関連
├── backtest/              # バックテスト機能
│   ├── engine/
│   │   ├── indicators.py
│   │   └── strategy_executor.py
│   └── runner.py
├── data_collector/        # データ収集機能
├── scripts/               # ユーティリティスクリプト
└── tests/                 # テスト
    ├── unit/              # 単体テスト
    └── integration/       # 結合テスト
```

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
# 基本依存関係
pip install -r requirements.txt

# または開発環境用
pip install -e ".[dev,test]"
```

### 2. 環境設定

```bash
# 環境設定ファイルをコピー
cp .env.example .env

# 必要に応じて設定を編集
vim .env
```

### 3. データベースの初期化

```bash
# データベース初期化スクリプトを実行
python scripts/init_database.py
```

### 4. アプリケーションの起動

```bash
# 開発サーバーを起動
python main.py

# または uvicorn を直接使用
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## 🧪 テスト

### テスト実行

```bash
# 全テスト実行
pytest

# 単体テストのみ
pytest tests/unit/

# 結合テストのみ
pytest tests/integration/

# カバレッジ付きで実行
pytest --cov=app --cov-report=html
```

### テスト構造

- **単体テスト**: 個別のクラス・関数のテスト
- **結合テスト**: API エンドポイントやデータベース連携のテスト

## 📡 API エンドポイント

### 市場データ API

- `GET /api/v1/market-data/ohlcv` - OHLCVデータ取得
- `GET /api/v1/market-data/symbols` - サポートシンボル一覧
- `GET /api/v1/market-data/timeframes` - サポート時間軸一覧

### ヘルスチェック

- `GET /health` - アプリケーション状態確認

## 🔧 開発

### コード品質

```bash
# フォーマット
black .
isort .

# リント
flake8 .

# 型チェック
mypy .
```

### 新機能の追加

1. **TDD アプローチ**: まず失敗するテストを書く
2. **実装**: テストが通るように実装
3. **リファクタリング**: コードを改善

### ディレクトリ別責務

- **app/config/**: 設定管理（環境変数、定数）
- **app/api/**: HTTP API エンドポイント
- **app/core/services/**: ビジネスロジック
- **app/core/models/**: データモデル
- **database/**: データベース関連（モデル、リポジトリ）
- **backtest/**: バックテスト機能
- **data_collector/**: データ収集機能
- **scripts/**: ユーティリティスクリプト

## 🔒 セキュリティ

- 環境変数による設定管理
- CORS 設定
- 入力値検証
- エラーハンドリング

## 📊 監視・ログ

- 構造化ログ出力
- エラー追跡
- パフォーマンス監視

## 🚀 デプロイ

### Docker（今後実装予定）

```bash
# イメージビルド
docker build -t trdinger-backend .

# コンテナ実行
docker run -p 8000:8000 trdinger-backend
```

## 🤝 貢献

1. フォークしてブランチを作成
2. 変更を実装（TDD に従う）
3. テストを実行して確認
4. プルリクエストを作成

## 📝 ライセンス

MIT License
