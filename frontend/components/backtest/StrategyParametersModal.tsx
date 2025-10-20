"use client";

import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Info, TrendingUp, Shield, DollarSign, BarChart3 } from "lucide-react";
import { BacktestResult } from "@/types/backtest";
import { formatPercentage, formatNumber } from "@/utils/formatters";
import { formatCurrency } from "@/utils/financialFormatters";

interface StrategyParametersModalProps {
  result: BacktestResult;
}

interface Condition {
  left_operand: string;
  operator: string;
  right_operand: string | number;
}

interface ConditionGroup {
  conditions: Condition[];
}

type ConditionOrGroup = Condition | ConditionGroup;

interface StrategyGene {
  id?: string;
  indicators?: Array<{
    type: string;
    parameters: Record<string, any>;
    enabled: boolean;
  }>;
  entry_conditions?: ConditionOrGroup[];
  long_entry_conditions?: ConditionOrGroup[];
  short_entry_conditions?: ConditionOrGroup[];
  exit_conditions?: ConditionOrGroup[];
  tpsl_gene?: Record<string, any>;
  position_sizing_gene?: Record<string, any>;
  metadata?: Record<string, any>;
}

export default function StrategyParametersModal({
  result,
}: StrategyParametersModalProps) {
  const strategyGene = result.config_json?.strategy_config?.parameters
    ?.strategy_gene as StrategyGene | undefined;

  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return "N/A";
    if (typeof value === "boolean") return value ? "有効" : "無効";
    if (typeof value === "number") {
      if (value >= 0 && value <= 1 && value.toString().includes(".")) {
        return `${(value * 100).toFixed(2)}%`;
      }
      if (value >= 1000) {
        return formatNumber(value, 0, 2);
      }
      return formatNumber(value, 0, 4);
    }
    if (typeof value === "object") {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  };

  const getEffectiveLongConditions = (): ConditionOrGroup[] => {
    if (!strategyGene) return [];
    if (
      strategyGene.long_entry_conditions &&
      strategyGene.long_entry_conditions.length > 0
    ) {
      return strategyGene.long_entry_conditions;
    }
    return strategyGene.entry_conditions || [];
  };

  const getEffectiveShortConditions = (): ConditionOrGroup[] => {
    if (!strategyGene) return [];
    if (
      strategyGene.short_entry_conditions &&
      strategyGene.short_entry_conditions.length > 0
    ) {
      return strategyGene.short_entry_conditions;
    }
    if (
      !strategyGene.long_entry_conditions ||
      strategyGene.long_entry_conditions.length === 0
    ) {
      return strategyGene.entry_conditions || [];
    }
    return [];
  };

  const formatCondition = (condition: ConditionOrGroup): string => {
    if ("conditions" in condition) {
      const subConditions = condition.conditions.map(
        (subCond) =>
          `${subCond.left_operand} ${subCond.operator} ${subCond.right_operand}`
      );
      return `(${subConditions.join(" OR ")})`;
    } else {
      return `${condition.left_operand} ${condition.operator} ${condition.right_operand}`;
    }
  };

  const effectiveLongConditions = getEffectiveLongConditions();
  const effectiveShortConditions = getEffectiveShortConditions();

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="ml-auto text-cyan-400 hover:text-cyan-300 hover:bg-cyan-400/10"
        >
          <Info className="w-4 h-4 mr-1" />
          詳細を表示
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto bg-gradient-to-br from-gray-900 to-black border-cyan-500/30">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 font-mono">
            戦略パラメータ詳細
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {/* 基本設定セクション */}
          <section>
            <div className="flex items-center mb-3">
              <BarChart3 className="w-5 h-5 text-purple-400 mr-2" />
              <h3 className="text-lg font-semibold text-purple-400 font-mono">
                基本設定
              </h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                <p className="text-sm text-gray-400 mb-1">戦略名</p>
                <p className="text-lg font-semibold text-white">
                  {result.strategy_name}
                </p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                <p className="text-sm text-gray-400 mb-1">取引ペア</p>
                <p className="text-lg font-semibold text-cyan-300">
                  {result.symbol}
                </p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                <p className="text-sm text-gray-400 mb-1">初期資金</p>
                <p className="text-lg font-semibold text-green-400">
                  {formatCurrency(result.initial_capital)}
                </p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                <p className="text-sm text-gray-400 mb-1">手数料率</p>
                <p className="text-lg font-semibold text-yellow-400">
                  {formatPercentage(result.commission_rate)}
                </p>
              </div>
            </div>
          </section>

          {/* 使用インジケーター */}
          {strategyGene?.indicators && strategyGene.indicators.length > 0 && (
            <section>
              <div className="flex items-center mb-3">
                <TrendingUp className="w-5 h-5 text-green-400 mr-2" />
                <h3 className="text-lg font-semibold text-green-400 font-mono">
                  使用インジケーター
                </h3>
              </div>
              <div className="space-y-2">
                {strategyGene.indicators.map((indicator, index) => (
                  <div
                    key={index}
                    className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 rounded-lg p-4 border border-green-500/30"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-green-300 font-semibold font-mono">
                        {indicator.type}
                      </span>
                      {!indicator.enabled && (
                        <span className="text-xs px-2 py-1 rounded bg-red-500/20 text-red-300 border border-red-500/30">
                          無効
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(indicator.parameters).map(
                        ([key, value]) => (
                          <div key={key} className="text-sm">
                            <span className="text-gray-400">{key}: </span>
                            <span className="text-green-200 font-mono">
                              {formatValue(value)}
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* エントリーロジック */}
          <section>
            <div className="flex items-center mb-3">
              <TrendingUp className="w-5 h-5 text-blue-400 mr-2" />
              <h3 className="text-lg font-semibold text-blue-400 font-mono">
                エントリーロジック
              </h3>
            </div>
            <div className="space-y-4">
              {/* ロング条件 */}
              <div>
                <div className="flex items-center mb-2">
                  <div className="w-3 h-3 bg-green-400 rounded-full mr-2" />
                  <h4 className="text-md font-semibold text-green-300">
                    ロングエントリー条件
                  </h4>
                  <span className="ml-2 text-xs px-2 py-1 rounded bg-green-400/10 text-green-300">
                    {effectiveLongConditions.length}個
                  </span>
                </div>
                {effectiveLongConditions.length > 0 ? (
                  <div className="space-y-2">
                    {effectiveLongConditions.map((condition, index) => (
                      <div
                        key={index}
                        className="bg-green-900/20 rounded-lg p-3 border border-green-500/20"
                      >
                        <code className="text-sm text-green-200 font-mono">
                          {formatCondition(condition)}
                        </code>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 italic">条件なし</p>
                )}
              </div>

              {/* ショート条件 */}
              <div>
                <div className="flex items-center mb-2">
                  <div className="w-3 h-3 bg-red-400 rounded-full mr-2" />
                  <h4 className="text-md font-semibold text-red-300">
                    ショートエントリー条件
                  </h4>
                  <span className="ml-2 text-xs px-2 py-1 rounded bg-red-400/10 text-red-300">
                    {effectiveShortConditions.length}個
                  </span>
                </div>
                {effectiveShortConditions.length > 0 ? (
                  <div className="space-y-2">
                    {effectiveShortConditions.map((condition, index) => (
                      <div
                        key={index}
                        className="bg-red-900/20 rounded-lg p-3 border border-red-500/20"
                      >
                        <code className="text-sm text-red-200 font-mono">
                          {formatCondition(condition)}
                        </code>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 italic">条件なし</p>
                )}
              </div>
            </div>
          </section>

          {/* リスク管理 (TP/SL) */}
          {strategyGene?.tpsl_gene && (
            <section>
              <div className="flex items-center mb-3">
                <Shield className="w-5 h-5 text-pink-400 mr-2" />
                <h3 className="text-lg font-semibold text-pink-400 font-mono">
                  リスク管理 (TP/SL)
                </h3>
              </div>
              <div className="bg-gradient-to-r from-pink-900/20 to-rose-900/20 rounded-lg p-4 border border-pink-500/30">
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(strategyGene.tpsl_gene).map(
                    ([key, value]) => (
                      <div key={key}>
                        <p className="text-sm text-gray-400 mb-1">
                          {key === "enabled" ? "有効/無効" :
                           key === "tp_pct" ? "利確 (TP)" :
                           key === "sl_pct" ? "損切 (SL)" :
                           key === "trailing_stop" ? "トレーリングストップ" :
                           key === "trailing_stop_pct" ? "トレーリング幅" :
                           key}
                        </p>
                        <p className="text-md font-semibold text-pink-200 font-mono">
                          {formatValue(value)}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </div>
            </section>
          )}

          {/* 資金管理 */}
          {strategyGene?.position_sizing_gene && (
            <section>
              <div className="flex items-center mb-3">
                <DollarSign className="w-5 h-5 text-emerald-400 mr-2" />
                <h3 className="text-lg font-semibold text-emerald-400 font-mono">
                  資金管理
                </h3>
              </div>
              <div className="bg-gradient-to-r from-emerald-900/20 to-green-900/20 rounded-lg p-4 border border-emerald-500/30">
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(strategyGene.position_sizing_gene).map(
                    ([key, value]) => (
                      <div key={key}>
                        <p className="text-sm text-gray-400 mb-1">
                          {key === "method" ? "方式" :
                           key === "risk_per_trade" ? "1トレードのリスク" :
                           key === "max_position_size" ? "最大ポジションサイズ" :
                           key === "kelly_fraction" ? "ケリー係数" :
                           key === "base_size" ? "基本サイズ" :
                           key === "leverage" ? "レバレッジ" :
                           key}
                        </p>
                        <p className="text-md font-semibold text-emerald-200 font-mono">
                          {formatValue(value)}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </div>
            </section>
          )}

          {!strategyGene && (
            <div className="text-center py-8">
              <div className="text-6xl mb-4">⚠️</div>
              <p className="text-lg text-gray-400">
                戦略パラメータの詳細情報がありません
              </p>
              <p className="text-sm text-gray-500 mt-2">
                この結果は古いバージョンで作成された可能性があります
              </p>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
