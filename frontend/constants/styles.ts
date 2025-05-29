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
    primary: "bg-primary-600 hover:bg-primary-700 text-white focus:ring-primary-500",
    secondary: "bg-gray-600 hover:bg-gray-700 text-white focus:ring-gray-500",
    success: "bg-green-600 hover:bg-green-700 text-white focus:ring-green-500",
    warning: "bg-yellow-600 hover:bg-yellow-700 text-white focus:ring-yellow-500",
    error: "bg-red-600 hover:bg-red-700 text-white focus:ring-red-500",
    outline: "border border-gray-600 hover:border-primary-500 text-gray-100 hover:bg-primary-900/20 focus:ring-primary-500",
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
 * 入力フィールドの共通スタイル
 */
export const INPUT_STYLES = {
  base: "w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200",
  sizes: {
    sm: "px-2 py-1 text-sm",
    md: "px-3 py-2 text-sm",
    lg: "px-4 py-3 text-base",
  },
  states: {
    error: "border-red-500 focus:ring-red-500",
    success: "border-green-500 focus:ring-green-500",
    disabled: "opacity-50 cursor-not-allowed bg-gray-700",
  },
} as const;

/**
 * セレクトボックスの共通スタイル
 */
export const SELECT_STYLES = {
  base: "appearance-none bg-gray-800 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200 cursor-pointer",
  compact: "px-3 py-2 text-sm min-w-[120px]",
  full: "px-4 py-3 text-sm min-w-[200px]",
  enterprise: "px-4 py-3 bg-gray-900 border-gray-700 rounded-enterprise hover:border-primary-400 hover:shadow-enterprise-md min-w-[280px]",
  states: {
    loading: "opacity-50 cursor-not-allowed",
    disabled: "opacity-50 cursor-not-allowed",
    hover: "hover:border-primary-400",
  },
} as const;

/**
 * カードコンポーネントの共通スタイル
 */
export const CARD_STYLES = {
  base: "bg-gray-900 rounded-lg border border-gray-700",
  enterprise: "bg-gray-900 rounded-enterprise-lg border border-gray-700",
  padding: {
    sm: "p-4",
    md: "p-6",
    lg: "p-8",
  },
  shadow: {
    sm: "shadow-sm",
    md: "shadow-md",
    lg: "shadow-lg",
    enterprise: "shadow-enterprise-lg",
  },
} as const;

/**
 * テキストの共通スタイル
 */
export const TEXT_STYLES = {
  headings: {
    h1: "text-3xl font-bold text-gray-100",
    h2: "text-2xl font-semibold text-gray-100",
    h3: "text-xl font-semibold text-gray-100",
    h4: "text-lg font-medium text-gray-100",
    h5: "text-base font-medium text-gray-100",
    h6: "text-sm font-medium text-gray-100",
  },
  body: {
    large: "text-lg text-gray-100",
    base: "text-base text-gray-100",
    small: "text-sm text-gray-300",
    xs: "text-xs text-gray-400",
  },
  colors: {
    primary: "text-primary-400",
    secondary: "text-gray-400",
    success: "text-green-400",
    warning: "text-yellow-400",
    error: "text-red-400",
    muted: "text-gray-500",
  },
} as const;

/**
 * レイアウトの共通スタイル
 */
export const LAYOUT_STYLES = {
  container: {
    sm: "max-w-sm mx-auto",
    md: "max-w-md mx-auto",
    lg: "max-w-lg mx-auto",
    xl: "max-w-xl mx-auto",
    "2xl": "max-w-2xl mx-auto",
    full: "w-full",
  },
  spacing: {
    xs: "space-y-2",
    sm: "space-y-3",
    md: "space-y-4",
    lg: "space-y-6",
    xl: "space-y-8",
  },
  grid: {
    cols1: "grid grid-cols-1",
    cols2: "grid grid-cols-1 md:grid-cols-2",
    cols3: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
    cols4: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
  },
  flex: {
    center: "flex items-center justify-center",
    between: "flex items-center justify-between",
    start: "flex items-center justify-start",
    end: "flex items-center justify-end",
    col: "flex flex-col",
    wrap: "flex flex-wrap",
  },
} as const;

/**
 * アニメーションの共通スタイル
 */
export const ANIMATION_STYLES = {
  transitions: {
    fast: "transition-all duration-150",
    normal: "transition-all duration-200",
    slow: "transition-all duration-300",
  },
  hover: {
    scale: "hover:scale-105",
    shadow: "hover:shadow-lg",
    brightness: "hover:brightness-110",
  },
  loading: {
    spin: "animate-spin rounded-full border-b-2 border-white",
    pulse: "animate-pulse",
    bounce: "animate-bounce",
  },
} as const;

/**
 * 状態インジケーターの共通スタイル
 */
export const STATUS_STYLES = {
  badges: {
    success: "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
    warning: "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    error: "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
    info: "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
    neutral: "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300",
  },
  dots: {
    success: "w-2 h-2 bg-green-500 rounded-full",
    warning: "w-2 h-2 bg-yellow-500 rounded-full",
    error: "w-2 h-2 bg-red-500 rounded-full",
    info: "w-2 h-2 bg-blue-500 rounded-full",
    neutral: "w-2 h-2 bg-gray-500 rounded-full",
    pulse: "w-2 h-2 bg-primary-500 rounded-full animate-pulse",
  },
} as const;

/**
 * スタイルを組み合わせるヘルパー関数
 */
export const combineStyles = (...styles: (string | undefined | false)[]): string => {
  return styles.filter(Boolean).join(" ");
};

/**
 * 条件付きスタイルを適用するヘルパー関数
 */
export const conditionalStyle = (condition: boolean, trueStyle: string, falseStyle?: string): string => {
  return condition ? trueStyle : (falseStyle || "");
};
