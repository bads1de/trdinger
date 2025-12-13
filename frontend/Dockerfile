# ベースイメージとしてNode.js 20を使用
FROM node:20-alpine AS development

# 作業ディレクトリを設定
WORKDIR /app

# package.json と package-lock.json をコピーして依存関係をインストール
# キャッシュを活用するため、依存関係ファイルを先にコピーしてインストール
COPY frontend/package.json frontend/package-lock.json ./

# 依存関係をインストール (本番環境用のビルドなので devDependencies は含めない)
# npm ci は package-lock.json を厳密に再現するため、CI/CD環境での利用に推奨される
RUN npm ci

# アプリケーションのコードを全てコピー
COPY frontend/ .

# Next.jsアプリケーションをビルド
RUN npm run build

# 本番環境用のイメージ (マルチステージビルド)
FROM node:20-alpine AS production

WORKDIR /app

# 開発ステージから必要なファイルをコピー
COPY --from=development /app/.next ./.next
COPY --from=development /app/node_modules ./node_modules
COPY --from=development /app/public ./public
COPY frontend/package.json ./package.json

# 環境変数を設定 (Next.jsのポート設定)
ENV PORT 3000

# ポートを公開
EXPOSE 3000

# Next.jsアプリケーションを起動
CMD ["npm", "start"]
