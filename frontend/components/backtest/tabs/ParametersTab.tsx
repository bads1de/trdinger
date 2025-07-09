import React from "react";
import { BacktestResult } from "@/types/backtest";
import CollapsibleJson from "@/components/common/CollapsibleJson";
import StrategyGeneDisplay from "../StrategyGeneDisplay";
import { formatCurrency, formatPercentage } from "@/utils/formatters";

interface ParametersTabProps {
  result: BacktestResult;
}

export default function ParametersTab({ result }: ParametersTabProps) {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-gray-900/90 to-black/80 rounded-xl p-6 border border-gray-700/50 shadow-2xl backdrop-blur-sm">
        <div className="flex items-center mb-6">
          <div className="w-4 h-4 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full mr-3 animate-pulse shadow-lg shadow-cyan-500/50"></div>
          <h3 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 font-mono tracking-wide">
            STRATEGY PARAMETERS
          </h3>
        </div>

        {/* デバッグ情報 - 折りたたみ可能なJSON表示 */}
        <CollapsibleJson
          data={result.config_json}
          title="DEBUG: CONFIG_JSON"
          defaultExpanded={false}
          theme="matrix"
          className="mb-4"
        />

        {result.config_json && result.config_json.strategy_config ? (
          <div className="space-y-4">
            {/* 戦略タイプ */}
            <div className="space-y-4">
              <div className="flex items-center mb-4">
                <div className="w-2 h-2 bg-purple-400 rounded-full mr-2"></div>
                <h4 className="text-purple-400 font-mono text-md font-semibold tracking-wide">
                  STRATEGY TYPE
                </h4>
              </div>


              {/* 戦略遺伝子の有効条件表示 */}
              {result.config_json.strategy_config.parameters?.strategy_gene && (
                <div className="mt-4">
                  <StrategyGeneDisplay
                    strategyGene={
                      result.config_json.strategy_config.parameters
                        .strategy_gene
                    }
                  />
                </div>
              )}
            </div>

            {/* バックテスト設定 */}
            <div>
              <div className="flex items-center mb-4">
                <div className="w-2 h-2 bg-purple-400 rounded-full mr-2"></div>
                <h4 className="text-purple-400 font-mono text-md font-semibold tracking-wide">
                  BACKTEST CONFIGURATION
                </h4>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-gray-900/80 to-black/60 rounded-lg p-4 border border-purple-500/30 shadow-lg hover:border-purple-400/50 transition-all duration-300 hover:shadow-purple-500/20">
                  <div className="flex flex-col space-y-2">
                    <span className="text-purple-300 text-sm font-mono uppercase tracking-wider">
                      初期資金
                    </span>
                    <span className="text-green-300 font-mono text-lg font-semibold">
                      {formatCurrency(result.initial_capital)}
                    </span>
                  </div>
                  <div className="absolute inset-0 bg-purple-400/5 rounded-lg pointer-events-none"></div>
                </div>
                <div className="bg-gradient-to-br from-gray-900/80 to-black/60 rounded-lg p-4 border border-purple-500/30 shadow-lg hover:border-purple-400/50 transition-all duration-300 hover:shadow-purple-500/20">
                  <div className="flex flex-col space-y-2">
                    <span className="text-purple-300 text-sm font-mono uppercase tracking-wider">
                      手数料率
                    </span>
                    <span className="text-yellow-300 font-mono text-lg font-semibold">
                      {formatPercentage(result.commission_rate)}
                    </span>
                  </div>
                  <div className="absolute inset-0 bg-purple-400/5 rounded-lg pointer-events-none"></div>
                </div>
                <div className="bg-gradient-to-br from-gray-900/80 to-black/60 rounded-lg p-4 border border-purple-500/30 shadow-lg hover:border-purple-400/50 transition-all duration-300 hover:shadow-purple-500/20">
                  <div className="flex flex-col space-y-2">
                    <span className="text-purple-300 text-sm font-mono uppercase tracking-wider">
                      取引ペア
                    </span>
                    <span className="text-cyan-300 font-mono text-lg font-semibold">
                      {result.symbol}
                    </span>
                  </div>
                  <div className="absolute inset-0 bg-purple-400/5 rounded-lg pointer-events-none"></div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* 基本情報のフォールバック表示 */}
            <div>
              <h4 className="text-md font-medium mb-3 text-white">基本設定</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                  <div className="flex flex-col space-y-1">
                    <span className="text-gray-400 text-sm">戦略名</span>
                    <span className="text-white font-medium text-lg">
                      {result.strategy_name}
                    </span>
                  </div>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                  <div className="flex flex-col space-y-1">
                    <span className="text-gray-400 text-sm">初期資金</span>
                    <span className="text-green-400 font-medium text-lg">
                      {formatCurrency(result.initial_capital)}
                    </span>
                  </div>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                  <div className="flex flex-col space-y-1">
                    <span className="text-gray-400 text-sm">手数料率</span>
                    <span className="text-yellow-400 font-medium text-lg">
                      {formatPercentage(result.commission_rate)}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="text-center py-8 text-gray-400">
              <div className="text-6xl mb-4">⚙️</div>
              <p className="text-lg">詳細なパラメータ情報がありません</p>
              <p className="text-sm mt-2">
                この結果は古いバージョンで作成されたため、戦略パラメータの詳細情報が含まれていません
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
