/**
 * Next.js設定ファイル
 *
 * Next.jsアプリケーションのビルドや実行時の設定を定義します。
 *
 * @see https://nextjs.org/docs/api-reference/next.config.js/introduction
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  // 実験的機能の設定
  experimental: {
    // App Routerを有効化（Next.js 13+の新しいルーティングシステム）
    // ファイルベースルーティングとレイアウト機能を提供
    appDir: true,
  },
}

module.exports = nextConfig
