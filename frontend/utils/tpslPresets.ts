/**
 * TP/SL自動決定のプリセット設定
 * 
 * ユーザーが簡単に選択できるプリセット設定を提供し、
 * 複雑な手動設定の負担を軽減します。
 */

import { TPSLPreset, TPSLStrategy, VolatilitySensitivity } from '../types/optimization';

// プリセット設定の定義
export const TPSL_PRESETS: Record<string, TPSLPreset> = {
  conservative: {
    name: "保守的",
    description: "低リスク・安定志向。リスクリワード比1:1.5、低ボラティリティ感度",
    strategy: "risk_reward",
    max_risk_per_trade: 0.02, // 2%
    preferred_risk_reward_ratio: 1.5,
    volatility_sensitivity: "low"
  },
  
  balanced: {
    name: "バランス型",
    description: "リスクとリターンのバランス。リスクリワード比1:2、中程度のボラティリティ感度",
    strategy: "auto_optimal",
    max_risk_per_trade: 0.03, // 3%
    preferred_risk_reward_ratio: 2.0,
    volatility_sensitivity: "medium"
  },
  
  aggressive: {
    name: "積極的",
    description: "高リターン志向。リスクリワード比1:3、高ボラティリティ感度",
    strategy: "volatility_adaptive",
    max_risk_per_trade: 0.05, // 5%
    preferred_risk_reward_ratio: 3.0,
    volatility_sensitivity: "high"
  },
  
  statistical: {
    name: "統計的最適化",
    description: "過去データに基づく統計的優位性を活用。データドリブンなアプローチ",
    strategy: "statistical",
    max_risk_per_trade: 0.03, // 3%
    preferred_risk_reward_ratio: 2.5,
    volatility_sensitivity: "medium"
  },
  
  volatility_adaptive: {
    name: "ボラティリティ適応",
    description: "市場のボラティリティに動的に適応。ATRベースの調整",
    strategy: "volatility_adaptive",
    max_risk_per_trade: 0.04, // 4%
    preferred_risk_reward_ratio: 2.0,
    volatility_sensitivity: "high"
  }
};

// デフォルトプリセット
export const DEFAULT_PRESET = "balanced";

// プリセット選択肢（UI表示用）
export const PRESET_OPTIONS = [
  { value: "conservative", label: "保守的 (低リスク)" },
  { value: "balanced", label: "バランス型 (推奨)" },
  { value: "aggressive", label: "積極的 (高リターン)" },
  { value: "statistical", label: "統計的最適化" },
  { value: "volatility_adaptive", label: "ボラティリティ適応" },
  { value: "custom", label: "カスタム設定" }
];

// TP/SL戦略の選択肢
export const TPSL_STRATEGY_OPTIONS = [
  { value: "auto_optimal", label: "自動最適化 (推奨)" },
  { value: "risk_reward", label: "リスクリワード比ベース" },
  { value: "volatility_adaptive", label: "ボラティリティ適応" },
  { value: "statistical", label: "統計的優位性" },
  { value: "random", label: "ランダム生成" },
  { value: "legacy", label: "従来方式" }
];

// ボラティリティ感度の選択肢
export const VOLATILITY_SENSITIVITY_OPTIONS = [
  { value: "low", label: "低感度 (安定志向)" },
  { value: "medium", label: "中感度 (バランス)" },
  { value: "high", label: "高感度 (積極的)" }
];

/**
 * プリセット設定を取得
 */
export function getPresetConfig(presetName: string): TPSLPreset | null {
  return TPSL_PRESETS[presetName] || null;
}

/**
 * プリセット設定をGA設定形式に変換
 */
export function convertPresetToGAConfig(preset: TPSLPreset) {
  return {
    tpsl_strategy: preset.strategy,
    max_risk_per_trade: preset.max_risk_per_trade,
    preferred_risk_reward_ratio: preset.preferred_risk_reward_ratio,
    volatility_sensitivity: preset.volatility_sensitivity,
    enable_advanced_tpsl: preset.strategy !== "legacy"
  };
}

/**
 * カスタム設定の妥当性チェック
 */
export function validateCustomTPSLConfig(config: {
  strategy: TPSLStrategy;
  max_risk_per_trade: number;
  preferred_risk_reward_ratio: number;
  volatility_sensitivity: VolatilitySensitivity;
}): string[] {
  const errors: string[] = [];
  
  // リスク設定の検証
  if (config.max_risk_per_trade < 0.005 || config.max_risk_per_trade > 0.1) {
    errors.push("1取引あたりの最大リスクは0.5%〜10%の範囲で設定してください");
  }
  
  // リスクリワード比の検証
  if (config.preferred_risk_reward_ratio < 1.0 || config.preferred_risk_reward_ratio > 5.0) {
    errors.push("リスクリワード比は1:1〜1:5の範囲で設定してください");
  }
  
  return errors;
}

/**
 * プリセット設定の説明文を取得
 */
export function getPresetDescription(presetName: string): string {
  const preset = TPSL_PRESETS[presetName];
  if (!preset) return "";
  
  return `${preset.description}\n\n設定詳細:\n` +
    `• 戦略: ${getStrategyDisplayName(preset.strategy)}\n` +
    `• 最大リスク: ${(preset.max_risk_per_trade * 100).toFixed(1)}%\n` +
    `• リスクリワード比: 1:${preset.preferred_risk_reward_ratio}\n` +
    `• ボラティリティ感度: ${getVolatilitySensitivityDisplayName(preset.volatility_sensitivity)}`;
}

/**
 * 戦略の表示名を取得
 */
function getStrategyDisplayName(strategy: TPSLStrategy): string {
  const strategyNames: Record<TPSLStrategy, string> = {
    "auto_optimal": "自動最適化",
    "risk_reward": "リスクリワード比ベース",
    "volatility_adaptive": "ボラティリティ適応",
    "statistical": "統計的優位性",
    "random": "ランダム生成",
    "legacy": "従来方式"
  };
  
  return strategyNames[strategy] || strategy;
}

/**
 * ボラティリティ感度の表示名を取得
 */
function getVolatilitySensitivityDisplayName(sensitivity: VolatilitySensitivity): string {
  const sensitivityNames: Record<VolatilitySensitivity, string> = {
    "low": "低感度",
    "medium": "中感度", 
    "high": "高感度"
  };
  
  return sensitivityNames[sensitivity] || sensitivity;
}

/**
 * 推奨プリセットを取得（ユーザーの経験レベルに基づく）
 */
export function getRecommendedPreset(experienceLevel: "beginner" | "intermediate" | "advanced"): string {
  switch (experienceLevel) {
    case "beginner":
      return "conservative";
    case "intermediate":
      return "balanced";
    case "advanced":
      return "statistical";
    default:
      return DEFAULT_PRESET;
  }
}

/**
 * プリセット設定の比較表示用データを生成
 */
export function generatePresetComparison() {
  return Object.entries(TPSL_PRESETS).map(([key, preset]) => ({
    name: key,
    displayName: preset.name,
    risk: `${(preset.max_risk_per_trade * 100).toFixed(1)}%`,
    reward: `1:${preset.preferred_risk_reward_ratio}`,
    strategy: getStrategyDisplayName(preset.strategy),
    volatility: getVolatilitySensitivityDisplayName(preset.volatility_sensitivity),
    description: preset.description
  }));
}
