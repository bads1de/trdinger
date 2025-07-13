"use client";

import React, { useState } from "react";
import Modal from "@/components/common/Modal";
import ActionButton from "@/components/common/ActionButton";
import BayesianOptimizationForm from "./BayesianOptimizationForm";
import BayesianOptimizationResults from "./BayesianOptimizationResults";
import {
  BayesianOptimizationConfig,
  BayesianOptimizationResult,
  BayesianOptimizationResponse,
} from "@/types/bayesian-optimization";
import { BacktestConfig } from "@/types/backtest";

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
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<BayesianOptimizationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGAOptimization = async (config: BayesianOptimizationConfig) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/bayesian-optimization/ga-parameters", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      const data: BayesianOptimizationResponse = await response.json();

      if (data.success && data.result) {
        setResult(data.result);
      } else {
        setError(data.error || "GAパラメータの最適化に失敗しました");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "予期せぬエラーが発生しました");
    } finally {
      setIsLoading(false);
    }
  };

  const handleMLOptimization = async (config: BayesianOptimizationConfig) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/bayesian-optimization/ml-hyperparameters", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      const data: BayesianOptimizationResponse = await response.json();

      if (data.success && data.result) {
        setResult(data.result);
      } else {
        setError(data.error || "MLハイパーパラメータの最適化に失敗しました");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "予期せぬエラーが発生しました");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    setResult(null);
    setError(null);
    onClose();
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="ベイジアン最適化"
      size="xl"
      closeOnOverlayClick={!isLoading}
      closeOnEscape={!isLoading}
      showCloseButton={!isLoading}
      contentClassName="p-0"
    >
      <div className="max-h-[80vh] overflow-y-auto">
        {!result && !error && (
          <div className="p-6">
            <BayesianOptimizationForm
              onGAOptimization={handleGAOptimization}
              onMLOptimization={handleMLOptimization}
              isLoading={isLoading}
              currentBacktestConfig={currentBacktestConfig}
            />
          </div>
        )}

        {isLoading && (
          <div className="p-6">
            <div className="flex items-center justify-center space-x-3">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="text-lg">ベイジアン最適化を実行中...</span>
            </div>
            <div className="mt-4 text-center text-sm text-gray-600">
              この処理には数分かかる場合があります
            </div>
          </div>
        )}

        {error && (
          <div className="p-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
              <h3 className="text-lg font-medium text-red-800 mb-2">エラーが発生しました</h3>
              <p className="text-red-700">{error}</p>
            </div>
            <div className="flex justify-center space-x-3">
              <ActionButton onClick={handleReset} variant="secondary">
                再試行
              </ActionButton>
              <ActionButton onClick={handleClose}>
                閉じる
              </ActionButton>
            </div>
          </div>
        )}

        {result && (
          <div className="p-6">
            <div className="mb-4 flex justify-between items-center">
              <h2 className="text-xl font-bold">最適化完了</h2>
              <div className="space-x-2">
                <ActionButton onClick={handleReset} variant="secondary" size="sm">
                  新しい最適化
                </ActionButton>
                <ActionButton onClick={handleClose} size="sm">
                  閉じる
                </ActionButton>
              </div>
            </div>
            <BayesianOptimizationResults result={result} />
          </div>
        )}
      </div>
    </Modal>
  );
};

export default BayesianOptimizationModal;
