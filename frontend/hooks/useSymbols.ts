import { useState, useEffect } from "react";
import { TradingPair } from "@/types/strategy";
import { SUPPORTED_TRADING_PAIRS } from "@/constants";

export const useSymbols = () => {
  const [symbols, setSymbols] = useState<TradingPair[]>([]);

  useEffect(() => {
    setSymbols(SUPPORTED_TRADING_PAIRS);
  }, []);

  const refetch = () => {};

  return {
    symbols,
    loading: false,
    error: "",
    refetch,
  };
};
