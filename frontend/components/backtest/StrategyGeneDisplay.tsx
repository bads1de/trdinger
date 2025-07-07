/**
 * 戦略遺伝子表示コンポーネント
 *
 * 戦略遺伝子の有効な条件を後方互換性を考慮して表示します。
 */

"use client";

import React, { useState } from "react";
import { ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/24/outline";

interface StrategyGene {
  id?: string;
  indicators?: Array<{
    type: string;
    parameters: Record<string, any>;
    enabled: boolean;
  }>;
  entry_conditions?: Array<{
    left_operand: string;
    operator: string;
    right_operand: string | number;
  }>;
  long_entry_conditions?: Array<{
    left_operand: string;
    operator: string;
    right_operand: string | number;
  }>;
  short_entry_conditions?: Array<{
    left_operand: string;
    operator: string;
    right_operand: string | number;
  }>;
  exit_conditions?: Array<{
    left_operand: string;
    operator: string;
    right_operand: string | number;
  }>;
  risk_management?: Record<string, any>;
  tpsl_gene?: Record<string, any>;
  position_sizing_gene?: Record<string, any>;
  metadata?: Record<string, any>;
}

interface StrategyGeneDisplayProps {
  strategyGene: StrategyGene;
}

const StrategyGeneDisplay: React.FC<StrategyGeneDisplayProps> = ({
  strategyGene,
}) => {
  const [expandedSections, setExpandedSections] = useState<
    Record<string, boolean>
  >({
    indicators: false,
    conditions: true,
    risk: false,
    tpsl: false,
    position_sizing: false,
  });

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  // 有効なロング条件を取得（後方互換性を考慮）
  const getEffectiveLongConditions = () => {
    if (
      strategyGene.long_entry_conditions &&
      strategyGene.long_entry_conditions.length > 0
    ) {
      return strategyGene.long_entry_conditions;
    }
    return strategyGene.entry_conditions || [];
  };

  // 有効なショート条件を取得（後方互換性を考慮）
  const getEffectiveShortConditions = () => {
    if (
      strategyGene.short_entry_conditions &&
      strategyGene.short_entry_conditions.length > 0
    ) {
      return strategyGene.short_entry_conditions;
    }
    // ロング・ショート分離がされていない場合は、entry_conditionsをショート条件としても使用
    if (
      !strategyGene.long_entry_conditions ||
      strategyGene.long_entry_conditions.length === 0
    ) {
      return strategyGene.entry_conditions || [];
    }
    return [];
  };

  const formatCondition = (condition: any) => {
    return `${condition.left_operand} ${condition.operator} ${condition.right_operand}`;
  };

  const effectiveLongConditions = getEffectiveLongConditions();
  const effectiveShortConditions = getEffectiveShortConditions();

  return (
    <div className="space-y-4">
      {/* 指標セクション */}
      <div className="bg-gradient-to-r from-green-900/30 to-emerald-900/30 rounded-lg p-4 border border-green-500/30">
        <button
          onClick={() => toggleSection("indicators")}
          className="flex items-center w-full text-left"
        >
          {expandedSections.indicators ? (
            <ChevronDownIcon className="w-4 h-4 text-green-400 mr-2" />
          ) : (
            <ChevronRightIcon className="w-4 h-4 text-green-400 mr-2" />
          )}
          <span className="text-green-300 text-sm font-mono uppercase tracking-wider">
            指標 ({strategyGene.indicators?.length || 0}個)
          </span>
        </button>

        {expandedSections.indicators && strategyGene.indicators && (
          <div className="mt-3 space-y-2">
            {strategyGene.indicators.map((indicator, index) => (
              <div
                key={index}
                className="bg-black/20 rounded p-2 text-xs font-mono"
              >
                <span className="text-green-400">{indicator.type}</span>
                <span className="text-gray-400 ml-2">
                  {Object.entries(indicator.parameters)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(", ")}
                </span>
                {!indicator.enabled && (
                  <span className="text-red-400 ml-2">(無効)</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 条件セクション */}
      <div className="bg-gradient-to-r from-blue-900/30 to-cyan-900/30 rounded-lg p-4 border border-blue-500/30">
        <button
          onClick={() => toggleSection("conditions")}
          className="flex items-center w-full text-left"
        >
          {expandedSections.conditions ? (
            <ChevronDownIcon className="w-4 h-4 text-blue-400 mr-2" />
          ) : (
            <ChevronRightIcon className="w-4 h-4 text-blue-400 mr-2" />
          )}
          <span className="text-blue-300 text-sm font-mono uppercase tracking-wider">
            取引条件
          </span>
        </button>

        {expandedSections.conditions && (
          <div className="mt-3 space-y-3">
            {/* ロング条件 */}
            <div>
              <div className="flex items-center mb-2">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                <span className="text-green-300 text-xs font-mono uppercase">
                  ロングエントリー条件 ({effectiveLongConditions.length}個)
                </span>
                {effectiveLongConditions.length > 0 &&
                  (!strategyGene.long_entry_conditions ||
                    strategyGene.long_entry_conditions.length === 0) && (
                    <span className="ml-2 text-xs text-yellow-400 bg-yellow-400/10 px-2 py-1 rounded">
                      後方互換
                    </span>
                  )}
              </div>
              {effectiveLongConditions.length > 0 ? (
                <div className="space-y-1">
                  {effectiveLongConditions.map((condition, index) => (
                    <div
                      key={index}
                      className="bg-green-900/20 rounded p-2 text-xs font-mono text-green-200"
                    >
                      {formatCondition(condition)}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-gray-500 italic">条件なし</div>
              )}
            </div>

            {/* ショート条件 */}
            <div>
              <div className="flex items-center mb-2">
                <div className="w-2 h-2 bg-red-400 rounded-full mr-2"></div>
                <span className="text-red-300 text-xs font-mono uppercase">
                  ショートエントリー条件 ({effectiveShortConditions.length}個)
                </span>
                {effectiveShortConditions.length > 0 &&
                  (!strategyGene.short_entry_conditions ||
                    strategyGene.short_entry_conditions.length === 0) && (
                    <span className="ml-2 text-xs text-yellow-400 bg-yellow-400/10 px-2 py-1 rounded">
                      後方互換
                    </span>
                  )}
              </div>
              {effectiveShortConditions.length > 0 ? (
                <div className="space-y-1">
                  {effectiveShortConditions.map((condition, index) => (
                    <div
                      key={index}
                      className="bg-red-900/20 rounded p-2 text-xs font-mono text-red-200"
                    >
                      {formatCondition(condition)}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-gray-500 italic">条件なし</div>
              )}
            </div>

            {/* エグジット条件 */}
            <div>
              <div className="flex items-center mb-2">
                <div className="w-2 h-2 bg-purple-400 rounded-full mr-2"></div>
                <span className="text-purple-300 text-xs font-mono uppercase">
                  エグジット条件 ({strategyGene.exit_conditions?.length || 0}個)
                </span>
              </div>
              {strategyGene.exit_conditions &&
              strategyGene.exit_conditions.length > 0 ? (
                <div className="space-y-1">
                  {strategyGene.exit_conditions.map((condition, index) => (
                    <div
                      key={index}
                      className="bg-purple-900/20 rounded p-2 text-xs font-mono text-purple-200"
                    >
                      {formatCondition(condition)}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-gray-500 italic">条件なし</div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* リスク管理セクション */}
      {strategyGene.risk_management && (
        <div className="bg-gradient-to-r from-yellow-900/30 to-orange-900/30 rounded-lg p-4 border border-yellow-500/30">
          <button
            onClick={() => toggleSection("risk")}
            className="flex items-center w-full text-left"
          >
            {expandedSections.risk ? (
              <ChevronDownIcon className="w-4 h-4 text-yellow-400 mr-2" />
            ) : (
              <ChevronRightIcon className="w-4 h-4 text-yellow-400 mr-2" />
            )}
            <span className="text-yellow-300 text-sm font-mono uppercase tracking-wider">
              リスク管理
            </span>
          </button>

          {expandedSections.risk && (
            <div className="mt-3">
              <div className="bg-black/20 rounded p-2 text-xs font-mono">
                {Object.entries(strategyGene.risk_management).map(
                  ([key, value]) => (
                    <div key={key} className="text-yellow-200">
                      <span className="text-yellow-400">{key}:</span>{" "}
                      {String(value)}
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TP/SL遺伝子セクション */}
      {strategyGene.tpsl_gene && (
        <div className="bg-gradient-to-r from-pink-900/30 to-rose-900/30 rounded-lg p-4 border border-pink-500/30">
          <button
            onClick={() => toggleSection("tpsl")}
            className="flex items-center w-full text-left"
          >
            {expandedSections.tpsl ? (
              <ChevronDownIcon className="w-4 h-4 text-pink-400 mr-2" />
            ) : (
              <ChevronRightIcon className="w-4 h-4 text-pink-400 mr-2" />
            )}
            <span className="text-pink-300 text-sm font-mono uppercase tracking-wider">
              TP/SL設定
            </span>
          </button>

          {expandedSections.tpsl && (
            <div className="mt-3">
              <div className="bg-black/20 rounded p-2 text-xs font-mono space-y-1">
                {Object.entries(strategyGene.tpsl_gene).map(([key, value]) => (
                  <div key={key} className="text-pink-200">
                    <span className="text-pink-400">{key}:</span>{" "}
                    {typeof value === "object"
                      ? JSON.stringify(value)
                      : String(value)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ポジションサイジング遺伝子セクション */}
      {strategyGene.position_sizing_gene && (
        <div className="bg-gradient-to-r from-emerald-900/30 to-green-900/30 rounded-lg p-4 border border-emerald-500/30">
          <button
            onClick={() => toggleSection("position_sizing")}
            className="flex items-center w-full text-left"
          >
            {expandedSections.position_sizing ? (
              <ChevronDownIcon className="h-4 w-4 mr-2 text-emerald-400" />
            ) : (
              <ChevronRightIcon className="h-4 w-4 mr-2 text-emerald-400" />
            )}
            <span className="text-emerald-300 font-medium">
              ポジションサイジング遺伝子
            </span>
          </button>

          {expandedSections.position_sizing && (
            <div className="mt-3">
              <div className="bg-black/20 rounded p-2 text-xs font-mono space-y-1">
                {Object.entries(strategyGene.position_sizing_gene).map(
                  ([key, value]) => (
                    <div key={key} className="text-emerald-200">
                      <span className="text-emerald-400">{key}:</span>{" "}
                      {typeof value === "object"
                        ? JSON.stringify(value)
                        : String(value)}
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default StrategyGeneDisplay;
