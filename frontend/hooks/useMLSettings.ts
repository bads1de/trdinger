import { useState, useEffect, useCallback } from "react";
import { BACKEND_API_URL } from "@/constants";

export interface AutoMLConfig {
  tsfresh: {
    enabled: boolean;
    feature_selection: boolean;
    fdr_level: number;
    feature_count_limit: number;
    parallel_jobs: number;
    performance_mode: string;
  };
  featuretools: {
    enabled: boolean;
    max_depth: number;
    max_features: number;
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
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isCleaning, setIsCleaning] = useState(false);
  const [isAutomlLoading, setIsAutomlLoading] = useState(false);
  const [isAutomlSaving, setIsAutomlSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const showSuccessMessage = (message: string) => {
    setSuccessMessage(message);
    setTimeout(() => setSuccessMessage(null), 3000);
  };

  const fetchConfig = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_API_URL}/api/ml/config`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "ML設定取得に失敗しました");
      }

      setConfig(data);
    } catch (error) {
      console.error("ML設定取得エラー:", error);
      setError(
        error instanceof Error ? error.message : "サーバーエラーが発生しました"
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  const saveConfig = useCallback(async (newConfig: MLConfig) => {
    if (!newConfig) return;

    setIsSaving(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_API_URL}/api/ml/config`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newConfig),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "ML設定の更新に失敗しました");
      }

      setConfig(data);
      showSuccessMessage("設定が正常に保存されました");
    } catch (error) {
      console.error("ML設定更新エラー:", error);
      setError(
        error instanceof Error ? error.message : "サーバーエラーが発生しました"
      );
    } finally {
      setIsSaving(false);
    }
  }, []);

  const resetToDefaults = useCallback(async () => {
    if (window.confirm("設定をデフォルト値にリセットしますか？")) {
      setIsResetting(true);
      setError(null);

      try {
        const response = await fetch(`${BACKEND_API_URL}/api/ml/config/reset`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "設定のリセットに失敗しました");
        }

        setConfig(data);
        showSuccessMessage("設定がデフォルト値にリセットされました");
      } catch (error) {
        console.error("ML設定リセットエラー:", error);
        setError(
          error instanceof Error
            ? error.message
            : "サーバーエラーが発生しました"
        );
      } finally {
        setIsResetting(false);
      }
    }
  }, []);

  const cleanupOldModels = useCallback(async () => {
    if (
      window.confirm(
        "古いモデルファイルを削除しますか？この操作は取り消せません。"
      )
    ) {
      setIsCleaning(true);
      setError(null);

      try {
        const response = await fetch(
          `${BACKEND_API_URL}/api/ml/models/cleanup`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "モデルファイルの削除に失敗しました");
        }

        showSuccessMessage("古いモデルファイルが削除されました");
      } catch (error) {
        console.error("モデルファイル削除エラー:", error);
        setError(
          error instanceof Error
            ? error.message
            : "サーバーエラーが発生しました"
        );
      } finally {
        setIsCleaning(false);
      }
    }
  }, []);

  const updateConfig = (section: keyof MLConfig, key: string, value: any) => {
    if (!config) return;

    const newConfig = {
      ...config,
      [section]: {
        ...config[section],
        [key]: value,
      },
    };
    setConfig(newConfig);
  };

  // AutoML設定関連の関数
  const fetchAutoMLConfig = useCallback(async () => {
    setIsAutomlLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${BACKEND_API_URL}/api/automl-features/default-config`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "AutoML設定取得に失敗しました");
      }

      setAutomlConfig(data.config);
    } catch (error) {
      console.error("AutoML設定取得エラー:", error);
      setError(
        error instanceof Error ? error.message : "サーバーエラーが発生しました"
      );
    } finally {
      setIsAutomlLoading(false);
    }
  }, []);

  const validateAutoMLConfig = useCallback(async (config: AutoMLConfig) => {
    try {
      const response = await fetch(
        `${BACKEND_API_URL}/api/automl-features/validate-config`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(config),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "AutoML設定検証に失敗しました");
      }

      return data;
    } catch (error) {
      console.error("AutoML設定検証エラー:", error);
      throw error;
    }
  }, []);

  const generateAutoMLFeatures = useCallback(
    async (
      symbol: string,
      timeframe: string = "1h",
      limit: number = 1000,
      automlConfig?: AutoMLConfig,
      includeTarget: boolean = false
    ) => {
      setIsAutomlSaving(true);
      setError(null);

      try {
        const response = await fetch(
          `${BACKEND_API_URL}/api/automl-features/generate`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              symbol,
              timeframe,
              limit,
              automl_config: automlConfig,
              include_target: includeTarget,
            }),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "AutoML特徴量生成に失敗しました");
        }

        showSuccessMessage("AutoML特徴量が正常に生成されました");
        return data;
      } catch (error) {
        console.error("AutoML特徴量生成エラー:", error);
        setError(
          error instanceof Error
            ? error.message
            : "サーバーエラーが発生しました"
        );
        throw error;
      } finally {
        setIsAutomlSaving(false);
      }
    },
    []
  );

  const clearAutoMLCache = useCallback(async () => {
    try {
      const response = await fetch(
        `${BACKEND_API_URL}/api/automl-features/clear-cache`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "AutoMLキャッシュクリアに失敗しました");
      }

      showSuccessMessage("AutoMLキャッシュがクリアされました");
      return data;
    } catch (error) {
      console.error("AutoMLキャッシュクリアエラー:", error);
      setError(
        error instanceof Error ? error.message : "サーバーエラーが発生しました"
      );
      throw error;
    }
  }, []);

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
    // AutoML関連の関数
    fetchAutoMLConfig,
    validateAutoMLConfig,
    generateAutoMLFeatures,
    clearAutoMLCache,
    setAutomlConfig,
  };
};
