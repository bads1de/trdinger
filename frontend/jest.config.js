/**
 * Jestテスト設定ファイル
 *
 * Next.jsアプリケーション用のJestテスト環境を設定します。
 * Reactコンポーネントのテストやユニットテストに使用されます。
 *
 * @see https://nextjs.org/docs/testing#jest-and-react-testing-library
 */

// Next.js用Jest設定ヘルパーをインポート
const nextJest = require("next/jest");

// Next.js設定を読み込んでJest設定を作成
const createJestConfig = nextJest({
  // Next.jsアプリケーションのルートパス
  // next.config.jsや.envファイルを読み込むために必要
  dir: "./",
});

// Jestのカスタム設定
const customJestConfig = {
  // テスト実行前に読み込むセットアップファイル
  setupFilesAfterEnv: ["<rootDir>/jest.setup.js"],

  // モジュールパスのエイリアス設定
  moduleNameMapper: {
    // @/でルートディレクトリを参照するエイリアス（tsconfig.jsonのpaths設定と連動）
    "^@/(.*)$": "<rootDir>/$1",
  },

  // テスト実行環境（jsdomでブラウザ環境をシミュレート）
  // Reactコンポーネントのテストに必要
  testEnvironment: "jest-environment-jsdom",
};

// Next.jsの非同期設定を正しく読み込むためのエクスポート方法
module.exports = createJestConfig(customJestConfig);
