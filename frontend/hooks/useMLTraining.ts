import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";
import { OptimizationProfile } from "@/types/bayesian-optimization";

export interface TrainingConfig {
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  save_model: boolean;
  train_test_split: number;
  random_state: number;
  use_profile: boolean;
  profile_id?: number;
  profile_name?: string;
}

export interface TrainingStatus {
  is_training: boolean;
  progress: number;
  status: string;
  message: string;
  start_time?: string;
  end_time?: string;
  error?: string;
  model_info?: {
    accuracy: number;
    feature_count: number;
    training_samples: number;
    test_samples: number;
  };
}

export const useMLTraining = () => {
  const [config, setConfig] = useState<TrainingConfig>({
    symbol: "BTC/USDT:USDT",
    timeframe: "1h",
    start_date: "2020-03-05",
    end_date: "2024-12-31",
    save_model: true,
    train_test_split: 0.8,
    random_state: 42,
    use_profile: false,
  });

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    is_training: false,
    progress: 0,
    status: "idle",
    message: "待機中",
  });

  const [error, setError] = useState<string | null>(null);
  const [selectedProfile, setSelectedProfile] =
    useState<OptimizationProfile | null>(null);

  const { execute: startTrainingApi, loading: startTrainingLoading } =
    useApiCall();
  const { execute: stopTrainingApi, loading: stopTrainingLoading } =
    useApiCall();
  const { execute: checkTrainingStatusApi } = useApiCall<TrainingStatus>();

  const checkTrainingStatus = useCallback(() => {
    checkTrainingStatusApi("/api/ml/training/status", {
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

  const startTraining = useCallback(async () => {
    setError(null);

    await startTrainingApi("/api/ml/training/start", {
      method: "POST",
      body: config,
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
  }, [startTrainingApi, config]);

  const stopTraining = useCallback(async () => {
    await stopTrainingApi("/api/ml/training/stop", {
      method: "POST",
      onSuccess: () => {
        setTrainingStatus((prev) => ({
          ...prev,
          is_training: false,
          status: "stopped",
          message: "トレーニングが停止されました",
        }));
      },
      onError: (errorMessage) => {
        setError("トレーニングの停止に失敗しました: " + errorMessage);
      },
    });
  }, [stopTrainingApi]);

  return {
    config,
    setConfig,
    trainingStatus,
    error,
    setError,
    selectedProfile,
    setSelectedProfile,
    startTrainingLoading,
    stopTrainingLoading,
    startTraining,
    stopTraining,
  };
};
