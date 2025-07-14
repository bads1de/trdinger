import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";

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
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const {
    loading: isLoading,
    error: fetchError,
    execute: fetchConfigApi,
  } = useApiCall<MLConfig>();
  const {
    loading: isSaving,
    error: saveError,
    execute: saveConfigApi,
  } = useApiCall();
  const {
    loading: isResetting,
    error: resetError,
    execute: resetConfigApi,
  } = useApiCall();
  const {
    loading: isCleaning,
    error: cleanupError,
    execute: cleanupApi,
  } = useApiCall();

  const error = fetchError || saveError || resetError || cleanupError;

  const showSuccessMessage = (message: string) => {
    setSuccessMessage(message);
    setTimeout(() => setSuccessMessage(null), 3000);
  };

  const fetchConfig = useCallback(async () => {
    const result = await fetchConfigApi("/api/ml/config");
    if (result) {
      const { success, ...configData } = result as any;
      setConfig(configData);
    }
  }, [fetchConfigApi]);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  const saveConfig = useCallback(
    async (newConfig: MLConfig) => {
      if (!newConfig) return;
      await saveConfigApi("/api/ml/config", {
        method: "PUT",
        body: newConfig,
        onSuccess: () => showSuccessMessage("設定が正常に保存されました"),
      });
    },
    [saveConfigApi]
  );

  const resetToDefaults = useCallback(async () => {
    await resetConfigApi("/api/ml/config/reset", {
      method: "POST",
      confirmMessage: "設定をデフォルト値にリセットしますか？",
      onSuccess: () => {
        fetchConfig();
        showSuccessMessage("設定がデフォルト値にリセットされました");
      },
    });
  }, [resetConfigApi, fetchConfig]);

  const cleanupOldModels = useCallback(async () => {
    await cleanupApi("/api/ml/models/cleanup", {
      method: "POST",
      confirmMessage:
        "古いモデルファイルを削除しますか？この操作は取り消せません。",
      onSuccess: () => showSuccessMessage("古いモデルファイルが削除されました"),
    });
  }, [cleanupApi]);

  const updateConfig = (section: keyof MLConfig, key: string, value: any) => {
    if (!config) return;

    setConfig((prev) => ({
      ...prev!,
      [section]: {
        ...prev![section],
        [key]: value,
      },
    }));
  };

  return {
    config,
    isLoading,
    isSaving,
    isResetting,
    isCleaning,
    error,
    successMessage,
    fetchConfig,
    saveConfig,
    resetToDefaults,
    cleanupOldModels,
    updateConfig,
    setConfig,
  };
};
