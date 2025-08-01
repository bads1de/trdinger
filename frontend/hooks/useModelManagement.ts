import { useState, useCallback } from "react";
import { useApiCall } from "./useApiCall";

/**
 * モデル情報インターフェース
 *
 * 機械学習モデルの基本情報を保持します。
 */
export interface ModelInfo {
  /** モデル名 */
  name: string;
  /** モデルファイルパス */
  path: string;
  /** ファイルサイズ（MB） */
  size_mb: number;
  /** 最終更新日時 */
  modified_at: string;
  /** モデルタイプ */
  model_type: string;
  /** トレーナータイプ */
  trainer_type: string;
  /** 特徴量数 */
  feature_count: number;
  /** 特徴量重要度を持っているかどうか */
  has_feature_importance: boolean;
  /** 特徴量重要度の数 */
  feature_importance_count: number;
}

/**
 * 現在のモデル情報インターフェース
 *
 * 現在ロードされているモデルの状態情報を保持します。
 */
export interface CurrentModelInfo {
  /** モデルがロードされているかどうか */
  loaded: boolean;
  /** トレーナータイプ */
  trainer_type?: string;
  /** トレーニング済みかどうか */
  is_trained?: boolean;
  /** モデルタイプ */
  model_type?: string;
  /** 特徴量重要度を持っているかどうか */
  has_feature_importance?: boolean;
  /** 特徴量重要度の数 */
  feature_importance_count?: number;
  /** メッセージ */
  message?: string;
  /** エラーメッセージ */
  error?: string;
}

/**
 * モデル一覧レスポンスインターフェース
 *
 * モデル一覧取得APIのレスポンス形式を定義します。
 */
export interface ModelsListResponse {
  /** モデル情報の配列 */
  models: ModelInfo[];
  /** モデルの総数 */
  total_count: number;
  /** エラーメッセージ */
  error?: string;
}

/**
 * モデル読み込みレスポンスインターフェース
 *
 * モデル読み込みAPIのレスポンス形式を定義します。
 */
export interface LoadModelResponse {
  /** 読み込みが成功したかどうか */
  success: boolean;
  /** メッセージ */
  message?: string;
  /** エラーメッセージ */
  error?: string;
  /** 現在のモデル情報 */
  current_model?: CurrentModelInfo;
}

/**
 * モデル管理フック
 *
 * 機械学習モデルの管理機能を提供します。
 * モデル一覧の取得、現在のモデル情報の取得、モデルの読み込みなどの機能を提供します。
 *
 * @example
 * ```tsx
 * const {
 *   models,
 *   currentModel,
 *   loading,
 *   error,
 *   fetchModels,
 *   fetchCurrentModel,
 *   loadModel,
 *   refreshModels
 * } = useModelManagement();
 *
 * // モデル一覧を取得
 * fetchModels();
 *
 * // 現在のモデル情報を取得
 * fetchCurrentModel();
 *
 * // 特定のモデルを読み込み
 * loadModel('my_model');
 *
 * // モデル情報をリフレッシュ
 * refreshModels();
 * ```
 *
 * @returns {{
 *   models: ModelInfo[],
 *   currentModel: CurrentModelInfo | null,
 *   loading: boolean,
 *   error: string | null,
 *   fetchModels: () => Promise<void>,
 *   fetchCurrentModel: () => Promise<void>,
 *   loadModel: (modelName: string) => Promise<void>,
 *   refreshModels: () => Promise<void>
 * }} モデル管理関連の状態と操作関数
 */
export const useModelManagement = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [currentModel, setCurrentModel] = useState<CurrentModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { execute: fetchData } = useApiCall();

  /**
   * モデル一覧を取得
   *
   * 利用可能な全モデルの情報を取得します。
   */
  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);

    await fetchData("/api/ml/models", {
      method: "GET",
      onSuccess: (data) => {
        setModels(data.models || []);
        if (data.error) {
          setError(data.error);
        }
      },
      onError: (errorMessage: string) => {
        setError(errorMessage);
        setModels([]);
      },
      onFinally: () => {
        setLoading(false);
      },
    });
  }, [fetchData]);

  /**
   * 現在のモデル情報を取得
   *
   * 現在ロードされているモデルの状態情報を取得します。
   */
  const fetchCurrentModel = useCallback(async () => {
    await fetchData("/api/ml/models/current", {
      method: "GET",
      onSuccess: (data) => {
        setCurrentModel(data);
      },
      onError: (errorMessage: string) => {
        setCurrentModel({ loaded: false, error: errorMessage });
      },
    });
  }, [fetchData]);

  /**
   * 指定されたモデルを読み込み
   *
   * 指定された名前のモデルをメモリにロードします。
   *
   * @param {string} modelName - 読み込むモデル名
   */
  const loadModel = useCallback(
    async (modelName: string) => {
      setLoading(true);
      setError(null);

      await fetchData(`/api/ml/models/${encodeURIComponent(modelName)}/load`, {
        method: "POST",
        onSuccess: (data) => {
          if (data.success) {
            setCurrentModel(data.current_model || null);
            // モデル一覧を再取得（状態が変わった可能性があるため）
            fetchModels();
          } else {
            setError(data.error || "モデル読み込みに失敗しました");
          }
        },
        onError: (errorMessage: string) => {
          setError(errorMessage);
        },
        onFinally: () => {
          setLoading(false);
        },
      });
    },
    [fetchData, fetchModels]
  );

  /**
   * モデル情報をリフレッシュ
   *
   * モデル一覧と現在のモデル情報を同時に更新します。
   */
  const refreshModels = useCallback(async () => {
    await Promise.all([fetchModels(), fetchCurrentModel()]);
  }, [fetchModels, fetchCurrentModel]);

  return {
    /** モデル一覧 */
    models,
    /** 現在のモデル情報 */
    currentModel,
    /** ローディング状態 */
    loading,
    /** エラーメッセージ */
    error,
    /** モデル一覧を取得する関数 */
    fetchModels,
    /** 現在のモデル情報を取得する関数 */
    fetchCurrentModel,
    /** モデルを読み込む関数 */
    loadModel,
    /** モデル情報をリフレッシュする関数 */
    refreshModels,
  };
};
