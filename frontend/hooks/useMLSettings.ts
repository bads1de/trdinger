import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";

export interface AutoMLConfig {
  tsfresh: {
    enabled: boolean;
    feature_selection: boolean;
    fdr_level: number;
    feature_count_limit: number;
    parallel_jobs: number;
    performance_mode: string;
  };
  autofeat: {
    enabled: boolean;
    max_features: number;
    generations: number;
    population_size: number;
  };
}

export interface MLConfig {
  data_processing: {
    max_ohlcv_rows: number;
    max_feature_rows: number;
    feature_calculation_timeout: number;
    model_training_timeout: number;
  };
  model: {
    model_save_path: string;
    max_model_versions: number;
    model_retention_days: number;
  };
  lightgbm: {
    learning_rate: number;
    num_leaves: number;
    feature_fraction: number;
    bagging_fraction: number;
    num_boost_round: number;
    early_stopping_rounds: number;
  };
  training: {
    train_test_split: number;
    prediction_horizon: number;
    threshold_up: number;
    threshold_down: number;
    min_training_samples: number;
  };
  prediction: {
    default_up_prob: number;
    default_down_prob: number;
    default_range_prob: number;
  };
}


export const useMLSettings = () => {
  const [config, setConfig] = useState<MLConfig | null>(null);
  const [automlConfig, setAutomlConfig] = useState<AutoMLConfig | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const { execute: fetchConfigApi, loading: isLoading, error: fetchError } = useApiCall<MLConfig>();
  const { execute: saveConfigApi, loading: isSaving, error: saveError } = useApiCall<MLConfig>();
  const { execute: resetApi, loading: isResetting, error: resetError } = useApiCall<MLConfig>();
  const { execute: cleanupApi, loading: isCleaning, error: cleanupError } = useApiCall();
  const { execute: fetchAutoMLConfigApi, loading: isAutomlLoading, error: fetchAutoMLError } = useApiCall<{ config: AutoMLConfig }>();
  const { execute: validateAutoMLConfigApi, error: validateAutoMLError } = useApiCall();
  const { execute: generateAutoMLFeaturesApi, loading: isAutomlSaving, error: generateAutoMLError } = useApiCall();
  const { execute: clearAutoMLCacheApi, error: clearAutoMLCacheError } = useApiCall();

  const error = fetchError || saveError || resetError || cleanupError || fetchAutoMLError || validateAutoMLError || generateAutoMLError || clearAutoMLCacheError;

  const showSuccessMessage = (message: string) => {
    setSuccessMessage(message);
    setTimeout(() => setSuccessMessage(null), 3000);
  };

  const fetchConfig = useCallback(async () => {
    await fetchConfigApi("/api/ml/config", {
      onSuccess: setConfig,
    });
  }, [fetchConfigApi]);

  const saveConfig = useCallback(async (newConfig: MLConfig) => {
    if (!newConfig) return;
    await saveConfigApi("/api/ml/config", {
      method: "PUT",
      body: newConfig,
      onSuccess: (data) => {
        setConfig(data);
        showSuccessMessage("設定が正常に保存されました");
      },
    });
  }, [saveConfigApi]);

  const resetToDefaults = useCallback(async () => {
    await resetApi("/api/ml/config/reset", {
        method: "POST",
        confirmMessage: "設定をデフォルト値にリセットしますか？",
        onSuccess: (data) => {
            setConfig(data);
            showSuccessMessage("設定がデフォルト値にリセットされました");
        }
    });
  }, [resetApi]);

  const cleanupOldModels = useCallback(async () => {
    await cleanupApi("/api/ml/models/cleanup", {
        method: "POST",
        confirmMessage: "古いモデルファイルを削除しますか？この操作は取り消せません。",
        onSuccess: () => {
            showSuccessMessage("古いモデルファイルが削除されました");
        }
    });
  }, [cleanupApi]);

  const updateConfig = (section: keyof MLConfig, key: string, value: any) => {
    if (!config) return;
    const newConfig = { ...config, [section]: { ...config[section], [key]: value } };
    setConfig(newConfig);
  };

  const fetchAutoMLConfig = useCallback(async () => {
    await fetchAutoMLConfigApi("/api/automl-features/default-config", {
      onSuccess: (data) => setAutomlConfig(data.config),
    });
  }, [fetchAutoMLConfigApi]);

  const validateAutoMLConfig = useCallback(async (config: AutoMLConfig) => {
    return await validateAutoMLConfigApi("/api/automl-features/validate-config", {
      method: "POST",
      body: config,
    });
  }, [validateAutoMLConfigApi]);

  const generateAutoMLFeatures = useCallback(async (
    symbol: string,
    timeframe: string = "1h",
    limit: number = 1000,
    automlConfig?: AutoMLConfig,
    includeTarget: boolean = false
  ) => {
    const result = await generateAutoMLFeaturesApi("/api/automl-features/generate", {
      method: "POST",
      body: { symbol, timeframe, limit, automl_config: automlConfig, include_target: includeTarget },
      onSuccess: () => showSuccessMessage("AutoML特徴量が正常に生成されました"),
    });
    return result;
  }, [generateAutoMLFeaturesApi]);

  const clearAutoMLCache = useCallback(async () => {
    const result = await clearAutoMLCacheApi("/api/automl-features/clear-cache", {
      method: "POST",
      onSuccess: () => showSuccessMessage("AutoMLキャッシュがクリアされました"),
    });
    return result;
  }, [clearAutoMLCacheApi]);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  return {
    config,
    automlConfig,
    isLoading,
    isSaving,
    isResetting,
    isCleaning,
    isAutomlLoading,
    isAutomlSaving,
    error,
    successMessage,
    fetchConfig,
    saveConfig,
    resetToDefaults,
    cleanupOldModels,
    updateConfig,
    setConfig,
    fetchAutoMLConfig,
    validateAutoMLConfig,
    generateAutoMLFeatures,
    clearAutoMLCache,
    setAutomlConfig,
  };
};
