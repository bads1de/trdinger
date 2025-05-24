/**
 * PostCSS設定ファイル
 *
 * CSSの後処理を行うPostCSSのプラグイン設定です。
 * Tailwind CSSの処理とブラウザベンダープリフィックスの自動付与を行います。
 *
 * @see https://postcss.org/
 * @see https://tailwindcss.com/docs/using-with-preprocessors#using-post-css-as-your-preprocessor
 */
module.exports = {
  plugins: {
    // Tailwind CSSプラグイン
    // TailwindのユーティリティクラスをCSSに変換
    tailwindcss: {},

    // Autoprefixerプラグイン
    // ブラウザベンダープリフィックスを自動付与（-webkit-, -moz-等）
    autoprefixer: {},
  },
}
