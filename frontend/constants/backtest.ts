export const GA_OBJECTIVE_OPTIONS = [
  { value: "Sharpe Ratio", label: "シャープレシオ" },
  { value: "Total Return", label: "総リターン" },
  { value: "Max Drawdown", label: "最大ドローダウン" },
  { value: "Profit Factor", label: "プロフィットファクター" },
  { value: "Win Rate", label: "勝率" },
];

export const OPTIMIZATION_METHODS = [
  { value: "grid", label: "グリッドサーチ" },
  { value: "sambo", label: "SAMBO (ベイズ最適化)" },
];

export const ENHANCED_OPTIMIZATION_OBJECTIVES = [
  { value: "Sharpe Ratio", label: "シャープレシオ" },
  { value: "Return [%]", label: "リターン [%]" },
  { value: "-Max. Drawdown [%]", label: "-最大ドローダウン [%]" },
  { value: "Profit Factor", label: "プロフィットファクター" },
];

export const TIME_FRAME_OPTIONS = [
  { value: "15m", label: "15分" },
  { value: "30m", label: "30分" },
  { value: "1h", label: "1時間" },
  { value: "4h", label: "4時間" },
  { value: "1d", label: "1日" },
];
