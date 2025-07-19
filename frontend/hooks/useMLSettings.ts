import { useState, useEffect, useCallback } from "react";
import { useDataFetching } from "./useDataFetching";
import { usePostRequest } from "./usePostRequest";
import { usePutRequest } from "./usePutRequest";

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
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const {
    data: config,
    loading: isLoading,
    error: fetchError,
    refetch: fetchConfig,
    setData: setConfig,
  } = useDataFetching<MLConfig>({
    endpoint: "/api/ml/config",
    transform: (response) => {
      const { success, ...configData } = response as any;
      return success ? [configData] : [];
    },
  });

  const {
    sendPutRequest,
    isLoading: isSaving,
    error: saveError,
  } = usePutRequest<any, MLConfig>();
  const {
    sendPostRequest: sendResetRequest,
    isLoading: isResetting,
    error: resetError,
  } = usePostRequest();
  const {
    sendPostRequest: sendCleanupRequest,
    isLoading: isCleaning,
    error: cleanupError,
  } = usePostRequest();

  const error = fetchError || saveError || resetError || cleanupError;

  const showSuccessMessage = (message: string) => {
    setSuccessMessage(message);
    setTimeout(() => setSuccessMessage(null), 3000);
  };

  const saveConfig = useCallback(
    async (newConfig: MLConfig) => {
      if (!newConfig) return;
      const { success } = await sendPutRequest("/api/ml/config", newConfig);
      if (success) {
        showSuccessMessage("設定が正常に保存されました");
      }
    },
    [sendPutRequest]
  );

  const resetToDefaults = useCallback(async () => {
    if (window.confirm("設定をデフォルト値にリセットしますか？")) {
      const { success } = await sendResetRequest("/api/ml/config/reset");
      if (success) {
        fetchConfig();
        showSuccessMessage("設定がデフォルト値にリセットされました");
      }
    }
  }, [sendResetRequest, fetchConfig]);

  const cleanupOldModels = useCallback(async () => {
    if (
      window.confirm(
        "古いモデルファイルを削除しますか？この操作は取り消せません。"
      )
    ) {
      const { success } = await sendCleanupRequest("/api/ml/models/cleanup");
      if (success) {
        showSuccessMessage("古いモデルファイルが削除されました");
      }
    }
  }, [sendCleanupRequest]);

  const updateConfig = (section: keyof MLConfig, key: string, value: any) => {
    if (!config || config.length === 0) return;

    const newConfig = {
      ...config[0],
      [section]: {
        ...config[0][section],
        [key]: value,
      },
    };
    setConfig([newConfig]);
  };

  return {
    config: config && config.length > 0 ? config[0] : null,
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
    setConfig: (newConfig: MLConfig | null) => setConfig(newConfig ? [newConfig] : []),
  };
};
