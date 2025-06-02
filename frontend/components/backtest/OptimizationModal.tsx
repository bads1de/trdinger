/**
 * 最適化設定モーダルコンポーネント
 *
 * バックテスト最適化の設定を行うためのモーダルコンポーネントです。
 * 既存のOptimizationFormを内包し、モーダル形式で表示します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";
import Modal from "@/components/common/Modal";
import OptimizationForm from "./OptimizationForm";

/**
 * 最適化設定の型定義（既存のOptimizationFormから）
 */
interface OptimizationConfig {
  base_config: {
    strategy_name: string;
    symbol: string;
    timeframe: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    commission_rate: number;
    strategy_config: {
      strategy_type: string;
      parameters: Record<string, number>;
    };
  };
  optimization_params: {
    method: "grid" | "sambo";
    max_tries?: number;
    maximize: string;
    return_heatmap: boolean;
    return_optimization?: boolean;
    random_state?: number;
    constraint?: string;
    parameters: Record<string, number[]>;
  };
}

interface MultiObjectiveConfig {
  base_config: {
    strategy_name: string;
    symbol: string;
    timeframe: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    commission_rate: number;
    strategy_config: {
      strategy_type: string;
      parameters: Record<string, number>;
    };
  };
  optimization_params: {
    objectives: string[];
    weights: number[];
    method: "grid" | "sambo";
    max_tries?: number;
    parameters: Record<string, number[]>;
  };
}

interface RobustnessConfig {
  base_config: {
    strategy_name: string;
    symbol: string;
    timeframe: string;
    initial_capital: number;
    commission_rate: number;
    strategy_config: {
      strategy_type: string;
      parameters: Record<string, number>;
    };
  };
  test_periods: string[][];
  optimization_params: {
    method: "grid" | "sambo";
    max_tries?: number;
    maximize: string;
    parameters: Record<string, number[]>;
  };
}

/**
 * バックテスト結果の型定義（選択された結果から設定を取得するため）
 */
interface BacktestResult {
  id: string;
  strategy_name: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate: number;
  performance_metrics: {
    total_return: number | null;
    sharpe_ratio: number | null;
    max_drawdown: number | null;
    win_rate: number | null;
    profit_factor: number | null;
    total_trades: number | null;
    winning_trades: number | null;
    losing_trades: number | null;
    avg_win: number | null;
    avg_loss: number | null;
  };
}

/**
 * 最適化モーダルのプロパティ
 */
interface OptimizationModalProps {
  /** モーダルの表示状態 */
  isOpen: boolean;
  /** モーダルを閉じる関数 */
  onClose: () => void;
  /** 拡張最適化実行時のコールバック */
  onEnhancedOptimization: (config: OptimizationConfig) => void;
  /** マルチ目的最適化実行時のコールバック */
  onMultiObjectiveOptimization: (config: MultiObjectiveConfig) => void;
  /** ロバストネステスト実行時のコールバック */
  onRobustnessTest: (config: RobustnessConfig) => void;
  /** 最適化実行中かどうか */
  isLoading?: boolean;
  /** 選択されたバックテスト結果（設定を引き継ぐため） */
  selectedResult?: BacktestResult | null;
}

/**
 * 最適化設定モーダルコンポーネント
 */
const OptimizationModal: React.FC<OptimizationModalProps> = ({
  isOpen,
  onClose,
  onEnhancedOptimization,
  onMultiObjectiveOptimization,
  onRobustnessTest,
  isLoading = false,
  selectedResult = null,
}) => {
  // 最適化実行後にモーダルを閉じる処理を追加
  const handleEnhancedOptimization = (config: OptimizationConfig) => {
    onEnhancedOptimization(config);
    onClose();
  };

  const handleMultiObjectiveOptimization = (config: MultiObjectiveConfig) => {
    onMultiObjectiveOptimization(config);
    onClose();
  };

  const handleRobustnessTest = (config: RobustnessConfig) => {
    onRobustnessTest(config);
    onClose();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="バックテスト最適化設定"
      size="2xl"
      closeOnOverlayClick={!isLoading}
      closeOnEscape={!isLoading}
      showCloseButton={!isLoading}
      contentClassName="p-0"
    >
      <div className="max-h-[70vh] overflow-y-auto">
        <OptimizationForm
          onEnhancedOptimization={handleEnhancedOptimization}
          onMultiObjectiveOptimization={handleMultiObjectiveOptimization}
          onRobustnessTest={handleRobustnessTest}
          isLoading={isLoading}
          initialConfig={selectedResult}
        />
      </div>
    </Modal>
  );
};

export default OptimizationModal;
