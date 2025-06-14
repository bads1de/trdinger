/**
 * チャートテーマ設定
 *
 * Recharts用のダークモード対応テーマとカラーパレット
 */

/**
 * チャートカラーパレット（ダークモード対応）
 */
export const chartColors = {
  // 基本色
  primary: "#3B82F6", // blue-500
  secondary: "#8B5CF6", // violet-500
  success: "#10B981", // emerald-500
  danger: "#EF4444", // red-500
  warning: "#F59E0B", // amber-500
  info: "#06B6D4", // cyan-500
  neutral: "#6B7280", // gray-500

  // 資産曲線用
  equity: "#3B82F6", // blue-500
  equityFill: "#3B82F620", // blue-500 with opacity
  buyHold: "#8B5CF6", // violet-500

  // ドローダウン用
  drawdown: "#EF4444", // red-500
  drawdownFill: "#EF444420", // red-500 with opacity
  maxDrawdown: "#DC2626", // red-600

  // 取引用
  longTrade: "#10B981", // emerald-500
  shortTrade: "#F59E0B", // amber-500
  winTrade: "#10B981", // emerald-500
  lossTrade: "#EF4444", // red-500

  // グラデーション
  equityGradient: ["#3B82F6", "#1D4ED8"], // blue gradient
  drawdownGradient: ["#EF4444", "#DC2626"], // red gradient
  profitGradient: ["#10B981", "#059669"], // emerald gradient

  // 背景・UI
  chartBackground: "#111827", // gray-900
  gridColor: "#374151", // gray-700
  textColor: "#F9FAFB", // gray-50
  textColorSecondary: "#9CA3AF", // gray-400
  borderColor: "#4B5563", // gray-600
  tooltipBackground: "#1F2937", // gray-800
  tooltipBorder: "#4B5563", // gray-600
};

/**
 * レスポンシブブレークポイント
 */
export const chartBreakpoints = {
  mobile: { width: "100%", height: 300 },
  tablet: { width: "100%", height: 400 },
  desktop: { width: "100%", height: 500 },
  large: { width: "100%", height: 600 },
};

/**
 * チャート共通スタイル設定
 */
export const chartStyles = {
  // グリッド線
  grid: {
    stroke: chartColors.gridColor,
    strokeDasharray: "3 3",
    strokeOpacity: 0.3,
  },

  // 軸
  axis: {
    tick: {
      fill: chartColors.textColorSecondary,
      fontSize: 12,
    },
    axisLine: {
      stroke: chartColors.borderColor,
      strokeWidth: 1,
    },
    tickLine: {
      stroke: chartColors.borderColor,
      strokeWidth: 1,
    },
  },

  // ツールチップ
  tooltip: {
    contentStyle: {
      backgroundColor: chartColors.tooltipBackground,
      border: `1px solid ${chartColors.tooltipBorder}`,
      borderRadius: "8px",
      color: chartColors.textColor,
      fontSize: "14px",
      boxShadow:
        "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
    },
    labelStyle: {
      color: chartColors.textColor,
      fontWeight: "bold",
    },
  },

  // 凡例
  legend: {
    wrapperStyle: {
      color: chartColors.textColor,
      fontSize: "14px",
    },
  },
};

/**
 * チャートタイプ別のデフォルト設定
 */
export const chartDefaults = {
  // 線グラフ
  line: {
    strokeWidth: 2,
    dot: false,
    activeDot: { r: 4, strokeWidth: 0 },
  },

  // エリアグラフ
  area: {
    strokeWidth: 2,
    fillOpacity: 0.1,
  },

  // 散布図
  scatter: {
    r: 4,
    strokeWidth: 0,
  },

  // 棒グラフ
  bar: {
    radius: [2, 2, 0, 0],
  },

  // アニメーション
  animation: {
    duration: 300,
    easing: "ease-out",
  },
};

/**
 * カスタムツールチップスタイル
 */
