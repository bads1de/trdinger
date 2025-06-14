/**
 * 指標データを管理するカスタムフック
 */

import { useState, useEffect } from "react";

export interface IndicatorInfo {
  name: string;
  category: string;
  min_period: number;
}

export interface IndicatorCategories {
  trend: string[];
  momentum: string[];
  volatility: string[];
  volume: string[];
  price_transform: string[];
  other: string[];
}

/**
 * 指標リストを取得するカスタムフック
 */
export const useIndicators = () => {
  const [indicators, setIndicators] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchIndicators = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/indicators");

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setIndicators(data);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch indicators:", err);
        setError(err instanceof Error ? err.message : "Unknown error");

        // フォールバック: 基本的な指標リスト
        setIndicators([
          "SMA",
          "EMA",
          "RSI",
          "MACD",
          "BB",
          "STOCH",
          "CCI",
          "ADX",
          "ATR",
          "OBV",
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchIndicators();
  }, []);

  return { indicators, loading, error };
};

/**
 * カテゴリ別指標リストを取得するカスタムフック
 */
export const useIndicatorCategories = () => {
  const [categories, setCategories] = useState<IndicatorCategories | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchCategories = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/indicators/categories");

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setCategories(data);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch indicator categories:", err);
        setError(err instanceof Error ? err.message : "Unknown error");

        // フォールバック: 基本的なカテゴリ
        setCategories({
          trend: ["SMA", "EMA", "MACD"],
          momentum: ["RSI", "STOCH", "CCI"],
          volatility: ["BB", "ATR"],
          volume: ["OBV"],
          price_transform: [],
          other: ["ADX"],
        });
      } finally {
        setLoading(false);
      }
    };

    fetchCategories();
  }, []);

  return { categories, loading, error };
};

/**
 * 指標情報を取得するカスタムフック
 */
export const useIndicatorInfo = () => {
  const [indicatorInfo, setIndicatorInfo] = useState<
    Record<string, IndicatorInfo>
  >({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchIndicatorInfo = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/indicators/info");

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setIndicatorInfo(data);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch indicator info:", err);
        setError(err instanceof Error ? err.message : "Unknown error");

        // フォールバック: 基本的な指標情報
        setIndicatorInfo({
          SMA: {
            name: "Simple Moving Average",
            category: "trend",
            min_period: 2,
          },
          EMA: {
            name: "Exponential Moving Average",
            category: "trend",
            min_period: 2,
          },
          RSI: {
            name: "Relative Strength Index",
            category: "momentum",
            min_period: 2,
          },
        });
      } finally {
        setLoading(false);
      }
    };

    fetchIndicatorInfo();
  }, []);

  return { indicatorInfo, loading, error };
};
