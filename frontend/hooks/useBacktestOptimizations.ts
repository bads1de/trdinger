import { useState } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { BacktestConfig } from "@/types/backtest";

export const useBacktestOptimizations = () => {
  const [optimizationResult, setOptimizationResult] = useState<any>(null);
  const [optimizationType, setOptimizationType] = useState<
    "enhanced" | "multi" | "robustness"
  >("enhanced");
  const [isOptimizationModalOpen, setIsOptimizationModalOpen] = useState(false);
  const [currentBacktestConfig, setCurrentBacktestConfig] =
    useState<BacktestConfig | null>(null);

  const {
    execute: runEnhancedOptimization,
    loading: enhancedOptimizationLoading,
  } = useApiCall();
  const {
    execute: runMultiObjectiveOptimization,
    loading: multiOptimizationLoading,
  } = useApiCall();
  const { execute: runRobustnessTest, loading: robustnessTestLoading } =
    useApiCall();

  // 拡張最適化実行
  const handleEnhancedOptimization = async (config: any) => {
    setOptimizationType("enhanced");
    const response = await runEnhancedOptimization(
      "/api/backtest/optimize-enhanced",
      {
        method: "POST",
        body: config,
        onSuccess: (data) => {
          setOptimizationResult(data.result);
        },
        onError: (error) => {
          console.error("Enhanced optimization failed:", error);
        },
      }
    );
  };

  // マルチ目的最適化実行
  const handleMultiObjectiveOptimization = async (config: any) => {
    setOptimizationType("multi");
    const response = await runMultiObjectiveOptimization(
      "/api/backtest/multi-objective-optimization",
      {
        method: "POST",
        body: config,
        onSuccess: (data) => {
          setOptimizationResult(data.result);
        },
        onError: (error) => {
          console.error("Multi-objective optimization failed:", error);
        },
      }
    );
  };

  // ロバストネステスト実行
  const handleRobustnessTest = async (config: any) => {
    setOptimizationType("robustness");
    const response = await runRobustnessTest("/api/backtest/robustness-test", {
      method: "POST",
      body: config,
      onSuccess: (data) => {
        setOptimizationResult(data.result);
      },
      onError: (error) => {
        console.error("Robustness test failed:", error);
      },
    });
  };

  const isOptimizationLoading =
    enhancedOptimizationLoading ||
    multiOptimizationLoading ||
    robustnessTestLoading;

  return {
    optimizationResult,
    optimizationType,
    isOptimizationModalOpen,
    currentBacktestConfig,
    enhancedOptimizationLoading,
    multiOptimizationLoading,
    robustnessTestLoading,
    isOptimizationLoading,
    setOptimizationResult,
    setOptimizationType,
    setIsOptimizationModalOpen,
    setCurrentBacktestConfig,
    handleEnhancedOptimization,
    handleMultiObjectiveOptimization,
    handleRobustnessTest,
  };
};
