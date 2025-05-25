/**
 * Tailwind CSS設定ファイル
 *
 * Tailwind CSSのカスタマイズ設定を定義します。
 * コンテンツパス、テーマの拡張、プラグイン設定を含みます。
 *
 * @see https://tailwindcss.com/docs/configuration
 * @type {import('tailwindcss').Config}
 */
module.exports = {
  // Tailwindがスキャンするファイルパスの指定
  // これらのファイル内のTailwindクラスが最終的なCSSに含まれる
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',     // Pages Routerファイル
    './components/**/*.{js,ts,jsx,tsx,mdx}', // コンポーネントファイル
    './app/**/*.{js,ts,jsx,tsx,mdx}',       // App Routerファイル
  ],

  // テーマのカスタマイズ
  theme: {
    extend: {
      // カスタム背景グラデーションの定義
      backgroundImage: {
        // 放射状グラデーション（メインページの背景に使用）
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        // 円錐状グラデーション（メインページの背景に使用）
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },

  // 追加プラグイン（現在は使用していない）
  plugins: [],
}
