"use client";

import React from "react";
import Modal from "@/components/common/Modal";
import ActionButton from "@/components/common/ActionButton";
import { useBayesianOptimization } from "@/hooks/useBayesianOptimization";
import BayesianOptimizationForm from "./BayesianOptimizationForm";
import BayesianOptimizationResults from "./BayesianOptimizationResults";
import { BacktestConfig } from "@/types/optimization";

interface BayesianOptimizationModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentBacktestConfig?: BacktestConfig | null;
}

const BayesianOptimizationModal: React.FC<BayesianOptimizationModalProps> = ({
  isOpen,
  onClose,
  currentBacktestConfig = null,
}) => {
  const { result, error, isLoading, runMLOptimization, reset } =
    useBayesianOptimization();

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleReset = () => {
    reset();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="ベイジアン最適化"
      size="2xl"
      closeOnOverlayClick={!isLoading}
      closeOnEscape={!isLoading}
      showCloseButton={!isLoading}
      contentClassName="p-6"
    >
      <div className="max-h-[85vh]">
        {!result && !error && (
          <BayesianOptimizationForm
            onMLOptimization={runMLOptimization}
            isLoading={isLoading}
            currentBacktestConfig={currentBacktestConfig}
          />
        )}

        {isLoading && (
          <>
            <div className="flex items-center justify-center space-x-3">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="text-lg">ベイジアン最適化を実行中...</span>
            </div>
            <div className="mt-4 text-center text-sm text-gray-600">
              この処理には数分かかる場合があります
            </div>
          </>
        )}

        {error && (
          <>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
              <h3 className="text-lg font-medium text-red-800 mb-2">
                エラーが発生しました
              </h3>
              <p className="text-red-700">{error}</p>
            </div>
            <div className="flex justify-center space-x-3">
              <ActionButton onClick={handleReset} variant="secondary">
                再試行
              </ActionButton>
              <ActionButton onClick={handleClose}>閉じる</ActionButton>
            </div>
          </>
        )}

        {result && (
          <>
            <div className="mb-4 flex justify-between items-center">
              <h2 className="text-xl font-bold">最適化完了</h2>
              <div className="space-x-2">
                <ActionButton
                  onClick={handleReset}
                  variant="secondary"
                  size="sm"
                >
                  新しい最適化
                </ActionButton>
                <ActionButton onClick={handleClose} size="sm">
                  閉じる
                </ActionButton>
              </div>
            </div>
            <BayesianOptimizationResults result={result} />
          </>
        )}
      </div>
    </Modal>
  );
};

export default BayesianOptimizationModal;
