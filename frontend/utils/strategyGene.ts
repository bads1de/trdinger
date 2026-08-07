/**
 * 戦略遺伝子（GA戦略）の表示ユーティリティ
 */

import { StrategyConditionOrGroup } from "@/types/backtest";

/**
 * 戦略遺伝子の条件（単一条件 or 条件グループ）を文字列に整形する
 * @param condition - 整形する条件
 * @returns 条件を表す文字列（例: "rsi > 30"、"(ema > 100 OR ema < 50)"）
 * @example
 * // returns 'rsi > 30'
 * formatCondition({ left_operand: "rsi", operator: ">", right_operand: 30 })
 * // returns '(ema > 100 OR ema < 50)'
 * formatCondition({ conditions: [{ left_operand: "ema", operator: ">", right_operand: 100 }, ...] })
 */
export const formatCondition = (condition: StrategyConditionOrGroup): string => {
  if ("conditions" in condition) {
    // ConditionGroupの場合
    const subConditions = condition.conditions.map(
      (subCond) =>
        `${subCond.left_operand} ${subCond.operator} ${subCond.right_operand}`
    );
    return `(${subConditions.join(" OR ")})`;
  } else {
    // Conditionの場合
    return `${condition.left_operand} ${condition.operator} ${condition.right_operand}`;
  }
};
