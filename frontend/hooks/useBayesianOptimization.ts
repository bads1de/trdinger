import { useState, useCallback } from "react";
import { useApiCall } from "./useApiCall";
import {
  BayesianOptimizationConfig,
  BayesianOptimizationResult,
  BayesianOptimizationResponse,
} from "@/types/bayesian-optimization";

export const useBayesianOptimization = () => {
  const [result, setResult] = useState<BayesianOptimizationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { execute: runOptimization, loading: isLoading } =
    useApiCall<BayesianOptimizationResponse>();

  const runMLOptimization = useCallback(
    async (config: BayesianOptimizationConfig) => {
      setError(null);
      setResult(null);

      await runOptimization("/api/bayesian-optimization/ml-hyperparameters", {
        method: "POST",
        body: config,
        onSuccess: (data) => {
          if (data.success && data.result) {
            setResult(data.result);
          } else {
            setError(
              data.error || "MLハイパーパラメータの最適化に失敗しました"
            );
          }
        },
        onError: (errorMessage) => {
          setError(errorMessage);
        },
      });
    },
    [runOptimization]
  );

  const reset = () => {
    setResult(null);
    setError(null);
  };

  return {
    result,
    error,
    isLoading,
    runMLOptimization,
    reset,
  };
};
