/**
 * 最適化設定モーダルコンポーネント
 *
 * バックテスト最適化の設定を行うためのモーダルコンポーネントです。
 * 既存のOptimizationFormを内包し、モーダル形式で表示します。
 *
 */

"use client";

import React from "react";
import Modal from "@/components/common/Modal";
import OptimizationForm from "./OptimizationForm";
import {
  OptimizationConfig,
  MultiObjectiveConfig,
  RobustnessConfig,
  GAConfig,
  OptimizationModalProps,
  BacktestConfig,
  BacktestResult,
} from "@/types/optimization";

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
  /** GA戦略生成実行時のコールバック */
  onGAGeneration?: (config: GAConfig) => void;

  /** 最適化実行中かどうか */
  isLoading?: boolean;
  /** 選択されたバックテスト結果（設定を引き継ぐため） */
  selectedResult?: BacktestResult | null;
  /** 現在のバックテスト設定（基本設定を引き継ぐため） */
  currentBacktestConfig?: BacktestConfig | null;
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
  onGAGeneration,
  isLoading = false,
  selectedResult = null,
  currentBacktestConfig = null,
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

  const handleGAGeneration = (config: GAConfig) => {
    if (onGAGeneration) {
      onGAGeneration(config);
    }
    // GAの場合はモーダルを閉じない（進捗表示のため）
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
          onGAGeneration={handleGAGeneration}
          isLoading={isLoading}
          initialConfig={selectedResult}
          currentBacktestConfig={currentBacktestConfig}
        />
      </div>
    </Modal>
  );
};

export default OptimizationModal;
