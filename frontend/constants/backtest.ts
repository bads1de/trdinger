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

import { ObjectiveDefinition } from "@/types/optimization";

// 利用可能な目的関数
export const AVAILABLE_OBJECTIVES: ObjectiveDefinition[] = [
  {
    name: "total_return",
    display_name: "総リターン",
    description: "投資期間全体での総リターン率",
    type: "maximize",
    weight: 1.0,
  },
  {
    name: "sharpe_ratio",
    display_name: "シャープレシオ",
    description: "リスク調整後リターンの指標",
    type: "maximize",
    weight: 1.0,
  },
  {
    name: "max_drawdown",
    display_name: "最大ドローダウン",
    description: "最大の資産減少率（最小化したい）",
    type: "minimize",
    weight: -1.0,
  },
  {
    name: "win_rate",
    display_name: "勝率",
    description: "勝ちトレードの割合",
    type: "maximize",
    weight: 1.0,
  },
  {
    name: "profit_factor",
    display_name: "プロフィットファクター",
    description: "総利益と総損失の比率",
    type: "maximize",
    weight: 1.0,
  },
  {
    name: "sortino_ratio",
    display_name: "ソルティーノレシオ",
    description: "下方リスク調整後リターンの指標",
    type: "maximize",
    weight: 1.0,
  },
];
