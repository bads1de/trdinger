# uv公式のPythonイメージを使用してビルドステージでuvをセットアップ
FROM python:3.11-slim-bookworm

# uvを公式インストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# 作業ディレクトリを設定
WORKDIR /usr/src/app

# uv関連ファイルを先にコピーして依存関係インストール層をキャッシュ
COPY backend/pyproject.toml backend/uv.lock ./

# 依存関係をインストール (システム全体にインストール、本番用)
RUN uv sync --frozen --no-dev --no-install-project

# プロジェクト本体をインストール
COPY backend/ .
RUN uv sync --frozen --no-dev

# FastAPIのデフォルトポートを公開
EXPOSE 8000

# コンテナ起動時にFastAPIアプリケーションを起動
# Uvicornでリロードなしで起動（本番環境想定）
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
