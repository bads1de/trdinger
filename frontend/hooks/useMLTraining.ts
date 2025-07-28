import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";
import { EnsembleSettingsConfig } from "@/components/ml/EnsembleSettings";

export interface ParameterSpaceConfig {
  type: "real" | "integer" | "categorical";
  low?: number;
  high?: number;
  categories?: string[];
}

export interface OptimizationSettingsConfig {
  enabled: boolean;
  method: "optuna";
  n_calls: number;
  parameter_space: Record<string, ParameterSpaceConfig>;
}

export interface AutoMLFeatureConfig {
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

export interface TrainingConfig {
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  save_model: boolean;
  train_test_split: number;
  random_state: number;
  optimization_settings?: OptimizationSettingsConfig;
  automl_config?: AutoMLFeatureConfig;
}

export interface TrainingStatus {
  is_training: boolean;
  progress: number;
  status: string;
  message: string;
  start_time?: string;
  end_time?: string;
  error?: string;
  process_id?: string; // プロセスID追加
  model_info?: {
    accuracy: number;
    feature_count: number;
    training_samples: number;
    test_samples: number;
  };
}

export interface ProcessInfo {
  process_id: string;
  task_name: string;
  status: string;
  start_time: string;
  end_time?: string;
  metadata: Record<string, any>;
  is_alive: boolean;
}

export interface ProcessListResponse {
  processes: Record<string, ProcessInfo>;
  count: number;
}

// デフォルトのAutoML設定を作成
export const getDefaultAutoMLConfig = (): AutoMLFeatureConfig => ({
  tsfresh: {
    enabled: true,
    feature_selection: true,
    fdr_level: 0.05,
    feature_count_limit: 100,
    parallel_jobs: 2,
    performance_mode: "balanced",
  },
  featuretools: {
    enabled: true,
    max_depth: 2,
    max_features: 50,
  },
  autofeat: {
    enabled: false, // デフォルトでは無効（計算コストが高いため）
    max_features: 50,
    generations: 10,
    population_size: 30,
  },
});

// 金融最適化AutoML設定を作成
export const getFinancialOptimizedAutoMLConfig = (): AutoMLFeatureConfig => ({
  tsfresh: {
    enabled: true,
    feature_selection: true,
    fdr_level: 0.01,
    feature_count_limit: 200,
    parallel_jobs: 4,
    performance_mode: "financial_optimized",
  },
  featuretools: {
    enabled: true,
    max_depth: 3,
    max_features: 100,
  },
  autofeat: {
    enabled: true,
    max_features: 100,
    generations: 20,
    population_size: 50,
  },
});

export const useMLTraining = () => {
  const [config, setConfig] = useState<TrainingConfig>({
    symbol: "BTC/USDT:USDT",
    timeframe: "1h",
    start_date: "2020-03-05",
    end_date: "2025-07-01",
    save_model: true,
    train_test_split: 0.8,
    random_state: 42,
  });

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    is_training: false,
    progress: 0,
    status: "idle",
    message: "待機中",
  });

  const [error, setError] = useState<string | null>(null);

  const { execute: startTrainingApi, loading: startTrainingLoading } =
    useApiCall();
  const { execute: stopTrainingApi, loading: stopTrainingLoading } =
    useApiCall();
  const { execute: checkTrainingStatusApi } = useApiCall<TrainingStatus>();
  const { execute: getActiveProcessesApi } = useApiCall<ProcessListResponse>();
  const { execute: forceStopProcessApi } = useApiCall();

  const checkTrainingStatus = useCallback(() => {
    checkTrainingStatusApi("/api/ml-training/training/status", {
      onSuccess: (status) => {
        if (status) {
          setTrainingStatus(status);
        }
      },
      onError: (err) => {
        console.error("トレーニング状態の確認に失敗:", err);
      },
    });
  }, [checkTrainingStatusApi]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (trainingStatus.is_training) {
        checkTrainingStatus();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [trainingStatus.is_training, checkTrainingStatus]);

  const startTraining = useCallback(
    async (
      optimizationSettings?: OptimizationSettingsConfig,
      automlConfig?: AutoMLFeatureConfig,
      ensembleConfig?: EnsembleSettingsConfig
    ) => {
      setError(null);

      // 最適化設定、AutoML設定、アンサンブル設定を含むconfigを作成
      const trainingConfig = {
        ...config,
        optimization_settings: optimizationSettings?.enabled
          ? optimizationSettings
          : undefined,
        automl_config: automlConfig,
        ensemble_config: ensembleConfig,
      };

      await startTrainingApi("/api/ml-training/train", {
        method: "POST",
        body: trainingConfig,
        onSuccess: () => {
          setTrainingStatus({
            is_training: true,
            progress: 0,
            status: "starting",
            message: "トレーニングを開始しています...",
            start_time: new Date().toISOString(),
          });
        },
        onError: (errorMessage) => {
          setError(errorMessage);
        },
      });
    },
    [startTrainingApi, config]
  );

  const stopTraining = useCallback(
    async (force: boolean = false) => {
      const url = force
        ? "/api/ml-training/stop?force=true"
        : "/api/ml-training/stop";

      await stopTrainingApi(url, {
        method: "POST",
        onSuccess: () => {
          setTrainingStatus((prev) => ({
            ...prev,
            is_training: false,
            status: force ? "force_stopped" : "stopped",
            message: force
              ? "トレーニングが強制停止されました"
              : "トレーニングが停止されました",
            process_id: undefined,
          }));
        },
        onError: (errorMessage) => {
          setError("トレーニングの停止に失敗しました: " + errorMessage);
        },
      });
    },
    [stopTrainingApi]
  );

  const getActiveProcesses = useCallback(async () => {
    return new Promise<ProcessListResponse | null>((resolve) => {
      getActiveProcessesApi("/api/ml-training/processes", {
        onSuccess: (data) => {
          resolve(data);
        },
        onError: (errorMessage) => {
          console.error("プロセス一覧の取得に失敗:", errorMessage);
          resolve(null);
        },
      });
    });
  }, [getActiveProcessesApi]);

  const forceStopProcess = useCallback(
    async (processId: string) => {
      await forceStopProcessApi(
        `/api/ml-training/process/${processId}/force-stop`,
        {
          method: "POST",
          onSuccess: () => {
            // 該当プロセスが現在のトレーニングの場合、状態を更新
            if (trainingStatus.process_id === processId) {
              setTrainingStatus((prev) => ({
                ...prev,
                is_training: false,
                status: "force_stopped",
                message: "プロセスが強制停止されました",
                process_id: undefined,
              }));
            }
          },
          onError: (errorMessage) => {
            setError("プロセスの強制停止に失敗しました: " + errorMessage);
          },
        }
      );
    },
    [forceStopProcessApi, trainingStatus.process_id]
  );

  return {
    config,
    setConfig,
    trainingStatus,
    error,
    setError,
    startTrainingLoading,
    stopTrainingLoading,
    startTraining,
    stopTraining,
    getActiveProcesses,
    forceStopProcess,
  };
};
