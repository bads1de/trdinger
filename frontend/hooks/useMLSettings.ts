import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";

export interface MLConfig {
  data_processing: {
    max_ohlcv_rows: number;
    max_feature_rows: number;
    feature_calculation_timeout: number;
    model_training_timeout: number;
    model_prediction_timeout: number;
    memory_warning_threshold: number;
    memory_limit_threshold: number;
    debug_mode: boolean;
    log_level: string;
  };
  model: {
    model_save_path: string;
    model_file_extension: string;
    model_name_prefix: string;
    auto_strategy_model_name: string;
    max_model_versions: number;
    model_retention_days: number;
  };
  training: {
    train_test_split: number;
    cross_validation_folds: number;
    prediction_horizon: number;
    label_method: string;
    volatility_window: number;
    threshold_multiplier: number;
    min_threshold: number;
    max_threshold: number;
    threshold_up: number;
    threshold_down: number;
  };
  prediction: {
    default_up_prob: number;
    default_down_prob: number;
    default_range_prob: number;
    fallback_up_prob: number;
    fallback_down_prob: number;
    fallback_range_prob: number;
    min_probability: number;
    max_probability: number;
    probability_sum_min: number;
    probability_sum_max: number;
    expand_to_data_length: boolean;
    default_indicator_length: number;
  };
  ensemble: {
    enabled: boolean;
    algorithms: string[];
    voting_method: string;
    default_method: string;
    stacking_cv_folds: number;
    stacking_use_probas: boolean;
  };
  retraining: {
    check_interval_seconds: number;
    max_concurrent_jobs: number;
    job_timeout_seconds: number;
    data_retention_days: number;
    incremental_training_enabled: boolean;
    performance_degradation_threshold: number;
    data_drift_threshold: number;
  };
}

/**
 * ML設定管理フック
 *
 * 機械学習関連の各種設定を取得・管理します。
 * 基本設定、AutoML設定、設定の保存・リセット、モデルのクリーンアップなどの機能を提供します。
 *
 * @example
 * ```tsx
 * const {
 *   config,
 *   isLoading,
 *   isSaving,
 *   fetchConfig,
 *   saveConfig,
 *   resetToDefaults,
 *   cleanupOldModels,
 *   fetchAutoMLConfig,
 *   validateAutoMLConfig,
 *   generateAutoMLFeatures,
 *   clearAutoMLCache
 * } = useMLSettings();
 *
 * // 設定を取得
 * fetchConfig();
 *
 * // 設定を保存
 * saveConfig(newConfig);
 *
 * // AutoML特徴量を生成
 * generateAutoMLFeatures('BTC/USDT:USDT', '1h', 1000);
 * ```
 *
 * @returns {{
 *   config: MLConfig | null,
 *   isLoading: boolean,
 *   isSaving: boolean,
 *   isResetting: boolean,
 *   isCleaning: boolean,
 *   isAutomlLoading: boolean,
 *   isAutomlSaving: boolean,
 *   error: string | null,
 *   successMessage: string | null,
 *   fetchConfig: () => Promise<void>,
 *   saveConfig: (newConfig: MLConfig) => Promise<void>,
 *   resetToDefaults: () => Promise<void>,
 *   cleanupOldModels: () => Promise<void>,
 *   updateConfig: (section: keyof MLConfig, key: string, value: any) => void,
 *   setConfig: (config: MLConfig) => void,
 * }} ML設定管理関連の状態と操作関数
 */
export const useMLSettings = () => {
  const [config, setConfig] = useState<MLConfig | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const {
    execute: fetchConfigApi,
    loading: isLoading,
    error: fetchError,
  } = useApiCall<MLConfig>();
  const {
    execute: saveConfigApi,
    loading: isSaving,
    error: saveError,
  } = useApiCall<MLConfig>();
  const {
    execute: resetApi,
    loading: isResetting,
    error: resetError,
  } = useApiCall<MLConfig>();
  const {
    execute: cleanupApi,
    loading: isCleaning,
    error: cleanupError,
  } = useApiCall();
  const error =
    fetchError ||
    saveError ||
    resetError ||
    cleanupError;

  const showSuccessMessage = (message: string) => {
    setSuccessMessage(message);
    setTimeout(() => setSuccessMessage(null), 3000);
  };

  const fetchConfig = useCallback(async () => {
    await fetchConfigApi("/api/ml/config", {
      onSuccess: setConfig,
    });
  }, [fetchConfigApi]);

  const saveConfig = useCallback(
    async (newConfig: MLConfig) => {
      if (!newConfig) return;
      await saveConfigApi("/api/ml/config", {
        method: "PUT",
        body: newConfig,
        onSuccess: (data) => {
          setConfig(data);
          showSuccessMessage("設定が正常に保存されました");
        },
      });
    },
    [saveConfigApi]
  );

  const resetToDefaults = useCallback(async () => {
    await resetApi("/api/ml/config/reset", {
      method: "POST",
      confirmMessage: "設定をデフォルト値にリセットしますか？",
      onSuccess: (data) => {
        setConfig(data);
        showSuccessMessage("設定がデフォルト値にリセットされました");
      },
    });
  }, [resetApi]);

  const cleanupOldModels = useCallback(async () => {
    await cleanupApi("/api/ml/models/cleanup", {
      method: "POST",
      confirmMessage:
        "古いモデルファイルを削除しますか？この操作は取り消せません。",
      onSuccess: () => {
        showSuccessMessage("古いモデルファイルが削除されました");
      },
    });
  }, [cleanupApi]);

  const updateConfig = (section: keyof MLConfig, key: string, value: any) => {
    if (!config) return;
    const newConfig = {
      ...config,
      [section]: { ...config[section], [key]: value },
    };
    setConfig(newConfig);
  };

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

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
