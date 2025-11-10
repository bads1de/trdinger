import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";
import { EnsembleSettingsConfig } from "@/components/ml/EnsembleSettings";
import type {
  LabelGenerationConfig,
  FeatureProfile,
} from "@/types/ml-config";
// 利用可能モデルはフロント定数で管理
import { ALGORITHMS } from "../constants/algorithms";

/**
 * パラメータ空間設定インターフェース
 *
 * 最適化アルゴリズムが探索するパラメータの範囲とタイプを定義します。
 */
export interface ParameterSpaceConfig {
  /** パラメータのタイプ（実数、整数、カテゴリカル） */
  type: "real" | "integer" | "categorical";
  /** 数値パラメータの下限（real/integerの場合） */
  low?: number;
  /** 数値パラメータの上限（real/integerの場合） */
  high?: number;
  /** カテゴリカルパラメータの選択肢（categoricalの場合） */
  categories?: string[];
}

/**
 * 最適化設定インターフェース
 *
 * ハイパーパラメータ最適化の設定を定義します。
 */
export interface OptimizationSettingsConfig {
  /** 最適化を有効にするかどうか */
  enabled: boolean;
  /** 最適化手法（現在はoptunaのみ対応） */
  method: "optuna";
  /** 最適化の試行回数 */
  n_calls: number;
  /** 最適化対象のパラメータ空間定義 */
  parameter_space: Record<string, ParameterSpaceConfig>;
}

/**
 * 単一モデル設定インターフェース
 *
 * 単一の機械学習モデルの設定を定義します。
 */
export interface SingleModelConfig {
  /** モデルタイプ */
  model_type: string;
}

/**
 * トレーニング設定インターフェース
 *
 * 機械学習モデルのトレーニングに必要な設定を定義します。
 */
export interface TrainingConfig {
  /** 取引シンボル */
  symbol: string;
  /** 時間枠 */
  timeframe: string;
  /** トレーニング開始日 */
  start_date: string;
  /** トレーニング終了日 */
  end_date: string;
  /** モデルを保存するかどうか */
  save_model: boolean;
  /** トレーニングデータとテストデータの分割比率 */
  train_test_split: number;
  /** 乱数シード */
  random_state: number;
  /** 最適化設定（オプション） */
  optimization_settings?: OptimizationSettingsConfig;
  /** 単一モデル設定（オプション） */
  single_model_config?: SingleModelConfig;
  /** ラベル生成設定（オプション） */
  label_generation?: Partial<LabelGenerationConfig>;
  /** 特徴量プロファイル（オプション） */
  feature_profile?: FeatureProfile;
  /** カスタム特徴量allowlist（オプション） */
  custom_allowlist?: string[] | null;
}

/**
 * トレーニング状態インターフェース
 *
 * 現在のトレーニングの状態情報を保持します。
 */
export interface TrainingStatus {
  /** トレーニング中かどうか */
  is_training: boolean;
  /** トレーニングの進捗（0-100） */
  progress: number;
  /** トレーニングの状態 */
  status: string;
  /** 状態メッセージ */
  message: string;
  /** トレーニング開始時刻 */
  start_time?: string;
  /** トレーニング終了時刻 */
  end_time?: string;
  /** エラーメッセージ */
  error?: string;
  /** プロセスID */
  process_id?: string;
  /** モデル情報 */
  model_info?: {
    /** モデルの精度 */
    accuracy: number;
    /** 特徴量数 */
    feature_count: number;
    /** トレーニングサンプル数 */
    training_samples: number;
    /** テストサンプル数 */
    test_samples: number;
  };
}

/**
 * プロセス情報インターフェース
 *
 * 実行中のトレーニングプロセスの情報を保持します。
 */
export interface ProcessInfo {
  /** プロセスID */
  process_id: string;
  /** タスク名 */
  task_name: string;
  /** プロセス状態 */
  status: string;
  /** 開始時刻 */
  start_time: string;
  /** 終了時刻 */
  end_time?: string;
  /** メタデータ */
  metadata: Record<string, any>;
  /** プロセスが生存しているかどうか */
  is_alive: boolean;
}

/**
 * プロセス一覧レスポンスインターフェース
 *
 * 実行中の全プロセスの一覧情報を保持します。
 */
export interface ProcessListResponse {
  /** プロセス情報のマップ（キー：プロセスID、値：プロセス情報） */
  processes: Record<string, ProcessInfo>;
  /** プロセス数 */
  count: number;
}

/**
 * MLトレーニング管理フック
 *
 * 機械学習モデルのトレーニングを管理します。
 * トレーニングの開始、停止、状態監視、プロセス管理などの機能を提供します。
 *
 * @example
 * ```tsx
 * const {
 *   config,
 *   trainingStatus,
 *   startTraining,
 *   stopTraining,
 *   availableModels
 * } = useMLTraining();
 *
 * // トレーニングを開始
 * startTraining(optimizationSettings);
 *
 * // トレーニングを停止
 * stopTraining();
 *
 * // 利用可能なモデルを取得
 * fetchAvailableModels();
 * ```
 *
 * @returns {{
 *   config: TrainingConfig,
 *   setConfig: (config: TrainingConfig) => void,
 *   trainingStatus: TrainingStatus,
 *   error: string | null,
 *   setError: (error: string | null) => void,
 *   startTrainingLoading: boolean,
 *   stopTrainingLoading: boolean,
 *   startTraining: (optimizationSettings?: OptimizationSettingsConfig, ensembleConfig?: EnsembleSettingsConfig, singleModelConfig?: SingleModelConfig) => Promise<void>,
 *   stopTraining: (force?: boolean) => Promise<void>,
 *   getActiveProcesses: () => Promise<ProcessListResponse | null>,
 *   forceStopProcess: (processId: string) => Promise<void>,
 *   availableModels: string[],
 *   fetchAvailableModels: () => void
 * }} MLトレーニング管理関連の状態と操作関数
 */
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
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  const { execute: startTrainingApi, loading: startTrainingLoading } =
    useApiCall();
  const { execute: stopTrainingApi, loading: stopTrainingLoading } =
    useApiCall();
  const { execute: checkTrainingStatusApi } = useApiCall<TrainingStatus>();
  const { execute: getActiveProcessesApi } = useApiCall<ProcessListResponse>();
  const { execute: forceStopProcessApi } = useApiCall();
  // バックエンドAPI廃止に伴い、利用可能モデルはフロント定数から取得
  // ここでのimportは構文上不正のため、ファイル先頭レベルへ移動しました

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

  const fetchAvailableModels = useCallback(() => {
    try {
      // フロント定数（constants/algorithms）からキー一覧を使用
      const models = Object.keys(ALGORITHMS);
      setAvailableModels(models);
    } catch (error) {
      console.error("利用可能なモデルの取得に失敗:", error);
      setAvailableModels([]);
    }
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (trainingStatus.is_training) {
        checkTrainingStatus();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [trainingStatus.is_training, checkTrainingStatus]);

  // 初期化時に利用可能なモデルを取得
  useEffect(() => {
    fetchAvailableModels();
  }, [fetchAvailableModels]);

  const startTraining = useCallback(
    async (
      optimizationSettings?: OptimizationSettingsConfig,
      ensembleConfig?: EnsembleSettingsConfig,
      singleModelConfig?: SingleModelConfig
    ) => {
      setError(null);

      // 最適化設定、アンサンブル設定、単一モデル設定、ラベル生成設定、特徴量設定を含むconfigを作成
      const trainingConfig = {
        ...config,
        optimization_settings: optimizationSettings?.enabled
          ? optimizationSettings
          : undefined,
        ensemble_config: ensembleConfig,
        single_model_config: singleModelConfig,
        // ラベル生成設定を含める（オプション）
        label_generation: config.label_generation,
        // 特徴量プロファイルとカスタムallowlistを含める（オプション）
        feature_profile: config.feature_profile,
        custom_allowlist: config.custom_allowlist,
      };

      // 送信データをログ出力
      console.log("🚀 フロントエンドから送信するトレーニング設定:");
      console.log("📋 ensemble_config:", ensembleConfig);
      console.log("📋 ensemble_config.enabled:", ensembleConfig?.enabled);
      console.log("📋 single_model_config:", singleModelConfig);
      console.log("📋 label_generation:", config.label_generation);
      console.log("📋 feature_profile:", config.feature_profile);
      console.log("📋 custom_allowlist:", config.custom_allowlist);
      console.log("📋 trainingConfig全体:", trainingConfig);

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
    /** トレーニング設定 */
    config,
    /** トレーニング設定を更新する関数 */
    setConfig,
    /** トレーニング状態 */
    trainingStatus,
    /** エラーメッセージ */
    error,
    /** エラーメッセージを設定する関数 */
    setError,
    /** トレーニング開始中かどうか */
    startTrainingLoading,
    /** トレーニング停止中かどうか */
    stopTrainingLoading,
    /** トレーニングを開始する関数 */
    startTraining,
    /** トレーニングを停止する関数 */
    stopTraining,
    /** アクティブなプロセス一覧を取得する関数 */
    getActiveProcesses,
    /** プロセスを強制停止する関数 */
    forceStopProcess,
    /** 利用可能なモデル一覧 */
    availableModels,
    /** 利用可能なモデル一覧を取得する関数 */
    fetchAvailableModels,
  };
};
