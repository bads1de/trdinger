/**
 * StrategyGene変換ロジック
 *
 * ストラテジービルダーで作成した戦略をStrategyGene形式に変換し、
 * バックテスト設定と組み合わせてBacktestConfigを生成します。
 */

import { BacktestConfig } from "@/types/backtest";

// StrategyGene関連の型定義
interface IndicatorGene {
  type: string;
  parameters: Record<string, any>;
  enabled: boolean;
  json_config: {
    indicator_name: string;
    parameters: Record<string, any>;
  };
}

interface StrategyCondition {
  type: string;
  operator: string;
  indicator?: string;
  value?: number;
  indicator1?: string;
  indicator2?: string;
}

interface StrategyGene {
  id: string;
  indicators: IndicatorGene[];
  entry_conditions: StrategyCondition[];
  exit_conditions: StrategyCondition[];
  risk_management: {
    stop_loss_pct: number;
    take_profit_pct: number;
    position_sizing: string;
  };
  metadata: {
    created_by: string;
    version: string;
    created_at: string;
  };
}

type BacktestSettings = Pick<
  BacktestConfig,
  | "symbol"
  | "timeframe"
  | "start_date"
  | "end_date"
  | "initial_capital"
  | "commission_rate"
>;

/**
 * StrategyGeneとバックテスト設定をBacktestConfigに変換
 */
export function convertToBacktestConfig(
  strategyGene: StrategyGene,
  backtestSettings: BacktestSettings
): BacktestConfig {
  // 一意の戦略名を生成
  const strategyName = `CUSTOM_${Date.now()}`;

  return {
    strategy_name: strategyName,
    symbol: backtestSettings.symbol,
    timeframe: backtestSettings.timeframe,
    start_date: backtestSettings.start_date,
    end_date: backtestSettings.end_date,
    initial_capital: backtestSettings.initial_capital,
    commission_rate: backtestSettings.commission_rate,
    strategy_config: {
      strategy_type: "USER_CUSTOM",
      parameters: {
        strategy_gene: strategyGene,
      },
    },
  };
}

/**
 * StrategyGeneの妥当性を検証
 */
export function validateStrategyGene(strategyGene: StrategyGene): {
  isValid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  // 基本的な妥当性チェック
  if (!strategyGene.id) {
    errors.push("戦略IDが設定されていません");
  }

  if (strategyGene.indicators.length === 0) {
    errors.push("少なくとも1つの指標が必要です");
  }

  if (
    strategyGene.entry_conditions.length === 0 &&
    strategyGene.exit_conditions.length === 0
  ) {
    errors.push("エントリー条件またはイグジット条件が必要です");
  }

  // 指標の妥当性チェック
  strategyGene.indicators.forEach((indicator, index) => {
    if (!indicator.type) {
      errors.push(`指標${index + 1}: タイプが設定されていません`);
    }
    if (!indicator.parameters) {
      errors.push(`指標${index + 1}: パラメータが設定されていません`);
    }
  });

  // 条件の妥当性チェック
  const checkConditions = (conditions: StrategyCondition[], type: string) => {
    conditions.forEach((condition, index) => {
      if (!condition.type) {
        errors.push(`${type}条件${index + 1}: タイプが設定されていません`);
      }
      if (!condition.operator) {
        errors.push(`${type}条件${index + 1}: オペレータが設定されていません`);
      }
    });
  };

  checkConditions(strategyGene.entry_conditions, "エントリー");
  checkConditions(strategyGene.exit_conditions, "イグジット");

  return {
    isValid: errors.length === 0,
    errors,
  };
}

/**
 * BacktestConfigの妥当性を検証
 */
export function validateBacktestConfig(config: BacktestConfig): {
  isValid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  // 必須フィールドのチェック
  if (!config.strategy_name) {
    errors.push("戦略名が設定されていません");
  }
  if (!config.symbol) {
    errors.push("銘柄が設定されていません");
  }
  if (!config.timeframe) {
    errors.push("時間軸が設定されていません");
  }
  if (!config.start_date) {
    errors.push("開始日が設定されていません");
  }
  if (!config.end_date) {
    errors.push("終了日が設定されていません");
  }

  // 数値フィールドのチェック
  if (config.initial_capital <= 0) {
    errors.push("初期資金は正の値である必要があります");
  }
  if (config.commission_rate < 0 || config.commission_rate > 1) {
    errors.push("手数料率は0から1の間である必要があります");
  }

  // 日付の妥当性チェック
  if (config.start_date && config.end_date) {
    const startDate = new Date(config.start_date);
    const endDate = new Date(config.end_date);

    if (startDate >= endDate) {
      errors.push("開始日は終了日より前である必要があります");
    }
  }

  // 戦略設定のチェック
  if (!config.strategy_config) {
    errors.push("戦略設定が設定されていません");
  } else {
    if (config.strategy_config.strategy_type !== "USER_CUSTOM") {
      errors.push("戦略タイプはUSER_CUSTOMである必要があります");
    }
    if (!config.strategy_config.parameters?.strategy_gene) {
      errors.push("StrategyGeneが設定されていません");
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
}
