import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";
import { BacktestConfig, BacktestResult } from "@/types/optimization";
import { UnifiedStrategy } from "@/types/auto-strategy";

export const useOptimizationForm = (
  initialConfig: BacktestResult | null,
  currentBacktestConfig: BacktestConfig | null
) => {
  const [strategies, setStrategies] = useState<Record<string, UnifiedStrategy>>(
    {}
  );
  const [selectedStrategy, setSelectedStrategy] = useState<string>("");
  const [baseConfig, setBaseConfig] = useState<
    Omit<BacktestConfig, "strategy_config">
  >({
    strategy_name: "OPTIMIZED_STRATEGY",
    symbol: "BTC/USDT",
    timeframe: "1d",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 10000000,
    commission_rate: 0.00055,
  });

  const { execute: fetchStrategies, loading: isLoadingStrategies } =
    useApiCall();

  const loadStrategies = useCallback(async () => {
    try {
      const response = await fetchStrategies("/api/backtest/strategies");
      if (response?.success && Object.keys(response.strategies).length > 0) {
        setStrategies(response.strategies);
        if (!selectedStrategy) {
          setSelectedStrategy(Object.keys(response.strategies)[0]);
        }
      }
    } catch (error) {
      console.error("Failed to load strategies:", error);
    }
  }, [fetchStrategies, selectedStrategy]);

  useEffect(() => {
    loadStrategies();
  }, [loadStrategies]);

  useEffect(() => {
    if (currentBacktestConfig) {
      setBaseConfig({
        strategy_name: `${currentBacktestConfig.strategy_name}_OPTIMIZED`,
        symbol: currentBacktestConfig.symbol,
        timeframe: currentBacktestConfig.timeframe,
        start_date: currentBacktestConfig.start_date,
        end_date: currentBacktestConfig.end_date,
        initial_capital: currentBacktestConfig.initial_capital,
        commission_rate: currentBacktestConfig.commission_rate,
      });
      setSelectedStrategy(currentBacktestConfig.strategy_config.strategy_type);
    } else if (initialConfig) {
      setBaseConfig({
        strategy_name: `${initialConfig.strategy_name}_OPTIMIZED`,
        symbol: initialConfig.symbol,
        timeframe: initialConfig.timeframe,
        start_date: initialConfig.start_date,
        end_date: initialConfig.end_date,
        initial_capital: initialConfig.initial_capital,
        commission_rate: initialConfig.commission_rate,
      });
    }
  }, [currentBacktestConfig, initialConfig]);

  return {
    strategies,
    selectedStrategy,
    baseConfig,
    setBaseConfig,
    isLoadingStrategies,
  };
};
