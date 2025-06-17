import { useState, useEffect, useCallback } from "react";
import { TradingPair } from "@/types/strategy";

interface SymbolsResponse {
  success: boolean;
  data: TradingPair[];
  message?: string;
}

export const useSymbols = () => {
  const [symbols, setSymbols] = useState<TradingPair[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  const fetchSymbols = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/data/symbols");
      const result: SymbolsResponse = await response.json();

      if (result.success) {
        setSymbols(result.data);
      } else {
        setError(result.message || "Failed to fetch symbols");
      }
    } catch (err) {
      setError("An error occurred while fetching symbols");
      console.error("Error fetching symbols:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSymbols();
  }, [fetchSymbols]);

  return { symbols, loading, error, refetch: fetchSymbols };
};
