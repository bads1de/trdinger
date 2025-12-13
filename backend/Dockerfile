# ベースイメージとしてPython 3.10を使用
FROM python:3.10-slim-buster

# 作業ディレクトリを設定
WORKDIR /usr/src/app

# pyproject.tomlをコピーして依存関係をインストール
# backendディレクトリを全てコピーする前にpyproject.tomlをコピーしてpip install .を実行することで、
# pyproject.tomlに変更がなければ依存関係のインストール層がキャッシュされるようにする。
COPY backend/pyproject.toml ./pyproject.toml

# 依存関係をインストール
# setuptoolsを使用しているため、pip install . でプロジェクトの依存関係をインストール可能
# --no-cache-dir オプションでキャッシュを無効にし、コンテナイメージサイズを削減
RUN pip install --no-cache-dir .

# backendディレクトリの残りのファイルを全てコピー
COPY backend/ .

# FastAPIのデフォルトポートを公開
EXPOSE 8000

# コンテナ起動時にFastAPIアプリケーションを起動
# Uvicornでリロードなしで起動（本番環境想定）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
