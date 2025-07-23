"use client";

import { useState, useEffect, useCallback } from "react";
import { AutoMLPreset } from "./useMLTraining";
import { useApiCall } from "./useApiCall";
import {
  MARKET_CONDITION_LABELS,
  TRADING_STRATEGY_LABELS,
  DATA_SIZE_LABELS,
} from "@/constants/automl-presets-constants";

interface AutoMLPresetsData {
  presets: AutoMLPreset[];
  summary: {
    total_presets: number;
    strategies: string[];
    market_conditions: string[];
    data_sizes: string[];
    preset_names: string[];
  };
}

interface RecommendationCriteria {
  market_condition?: string;
  trading_strategy?: string;
  data_size?: string;
}

interface RecommendationResponse {
  recommended_preset: AutoMLPreset;
  recommendation_criteria: RecommendationCriteria;
}

export const useAutoMLPresets = () => {
  const [presets, setPresets] = useState<AutoMLPreset[]>([]);
  const [summary, setSummary] = useState<AutoMLPresetsData["summary"] | null>(
    null
  );
  const { execute: fetchApi, loading, error } = useApiCall<any>();

  const fetchPresets = useCallback(async () => {
    await fetchApi("/api/ml/automl-presets", {
      onSuccess: (data: AutoMLPresetsData) => {
        setPresets(data.presets);
        setSummary(data.summary);
      },
    });
  }, [fetchApi]);

  const getPresetByName = useCallback(
    async (name: string): Promise<AutoMLPreset | null> => {
      const data = await fetchApi(`/api/ml/automl-presets/${name}`);
      return data as AutoMLPreset | null;
    },
    [fetchApi]
  );

  const recommendPreset = useCallback(
    async (criteria: RecommendationCriteria): Promise<AutoMLPreset | null> => {
      const params = new URLSearchParams();
      if (criteria.market_condition) {
        params.append("market_condition", criteria.market_condition);
      }
      if (criteria.trading_strategy) {
        params.append("trading_strategy", criteria.trading_strategy);
      }
      if (criteria.data_size) {
        params.append("data_size", criteria.data_size);
      }

      const data: RecommendationResponse | null = await fetchApi(
        `/api/ml/automl-presets/recommend?${params}`,
        {
          method: "POST",
        }
      );

      return data ? data.recommended_preset : null;
    },
    [fetchApi]
  );

  // フィルタリング関数
  const getPresetsByStrategy = useCallback(
    (strategy: string) => {
      return presets.filter((preset) => preset.trading_strategy === strategy);
    },
    [presets]
  );

  const getPresetsByMarketCondition = useCallback(
    (condition: string) => {
      return presets.filter((preset) => preset.market_condition === condition);
    },
    [presets]
  );

  const getPresetsByDataSize = useCallback(
    (size: string) => {
      return presets.filter((preset) => preset.data_size === size);
    },
    [presets]
  );

  useEffect(() => {
    fetchPresets();
  }, [fetchPresets]);

  return {
    presets,
    summary,
    loading,
    error,
    fetchPresets,
    getPresetByName,
    recommendPreset,
    getPresetsByStrategy,
    getPresetsByMarketCondition,
    getPresetsByDataSize,
  };
};

// プリセット選択用のヘルパー関数
export const getPresetDisplayName = (preset: AutoMLPreset): string => {
  return `${preset.name} (${preset.trading_strategy})`;
};

export const getPresetDescription = (preset: AutoMLPreset): string => {
  return `${preset.description}\n\n市場条件: ${preset.market_condition}\n戦略: ${preset.trading_strategy}\nデータサイズ: ${preset.data_size}\n\n${preset.performance_notes}`;
};
