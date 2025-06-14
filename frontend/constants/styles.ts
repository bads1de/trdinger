/**
 * 共通スタイル定数
 *
 * アプリケーション全体で使用される共通のTailwind CSSクラスを定義します。
 * 一貫性のあるデザインシステムを提供し、スタイルの重複を削減します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

/**
 * ボタンの基本スタイル
 */
export const BUTTON_STYLES = {
  // 基本クラス
  base: "inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 disabled:cursor-not-allowed",

  // サイズバリエーション
  sizes: {
    xs: "px-2 py-1 text-xs min-w-[60px]",
    sm: "px-3 py-1.5 text-sm min-w-[80px]",
    md: "px-4 py-2 text-sm min-w-[120px]",
    lg: "px-6 py-3 text-base min-w-[160px]",
    xl: "px-8 py-4 text-lg min-w-[200px]",
  },

  // カラーバリエーション
  variants: {
    primary:
      "bg-primary-600 hover:bg-primary-700 text-white focus:ring-primary-500",
    secondary: "bg-gray-600 hover:bg-gray-700 text-white focus:ring-gray-500",
    success: "bg-green-600 hover:bg-green-700 text-white focus:ring-green-500",
    warning:
      "bg-yellow-600 hover:bg-yellow-700 text-white focus:ring-yellow-500",
    error: "bg-red-600 hover:bg-red-700 text-white focus:ring-red-500",
    outline:
      "border border-gray-600 hover:border-primary-500 text-gray-100 hover:bg-primary-900/20 focus:ring-primary-500",
    ghost: "text-gray-100 hover:bg-gray-800 focus:ring-gray-500",
  },

  // 状態別スタイル
  states: {
    loading: "bg-blue-600 text-white cursor-not-allowed opacity-75",
    success: "bg-green-600 text-white",
    error: "bg-red-600 text-white",
    disabled: "bg-gray-600 text-gray-400 cursor-not-allowed",
  },
} as const;

/**
 * スタイルを組み合わせるヘルパー関数
 */
export const combineStyles = (
  ...styles: (string | undefined | false)[]
): string => {
  return styles.filter(Boolean).join(" ");
};
