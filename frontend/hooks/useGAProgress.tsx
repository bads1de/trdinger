/**
 * GA進捗監視カスタムフック
 * 
 * 遺伝的アルゴリズムの進捗をリアルタイムで監視し、
 * 状態管理とコールバック処理を提供します。
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useApiCall } from "./useApiCall";

interface GAProgress {
  experiment_id: string;
  current_generation: number;
  total_generations: number;
  best_fitness: number;
  average_fitness: number;
  execution_time: number;
  estimated_remaining_time: number;
  progress_percentage: number;
  status: "running" | "completed" | "error";
  best_strategy_preview?: {
    indicators: string[];
    entry_condition: string;
    exit_condition: string;
  };
}

interface GAResult {
  experiment_id: string;
  best_strategy: any;
  best_fitness: number;
  execution_time: number;
  generations_completed: number;
  final_population_size: number;
}

interface UseGAProgressOptions {
  /** ポーリング間隔（ミリ秒）デフォルト: 5000 */
  pollingInterval?: number;
  /** 自動ポーリング開始フラグ デフォルト: true */
  autoStart?: boolean;
  /** 完了時のコールバック */
  onComplete?: (result: GAResult) => void;
  /** エラー時のコールバック */
  onError?: (error: string) => void;
  /** 進捗更新時のコールバック */
  onProgress?: (progress: GAProgress) => void;
}

interface UseGAProgressReturn {
  /** 現在の進捗情報 */
  progress: GAProgress | null;
  /** 最終結果 */
  result: GAResult | null;
  /** エラー情報 */
  error: string | null;
  /** ポーリング中かどうか */
  isPolling: boolean;
  /** ローディング状態 */
  isLoading: boolean;
  /** ポーリング開始 */
  startPolling: (experimentId: string) => void;
  /** ポーリング停止 */
  stopPolling: () => void;
  /** 実験停止 */
  stopExperiment: () => Promise<boolean>;
  /** 進捗リセット */
  reset: () => void;
}

/**
 * GA進捗監視フック
 */
export const useGAProgress = (options: UseGAProgressOptions = {}): UseGAProgressReturn => {
  const {
    pollingInterval = 5000,
    autoStart = true,
    onComplete,
    onError,
    onProgress,
  } = options;

  // 状態管理
  const [progress, setProgress] = useState<GAProgress | null>(null);
  const [result, setResult] = useState<GAResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentExperimentId, setCurrentExperimentId] = useState<string | null>(null);

  // ポーリング制御用のref
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isPollingRef = useRef(false);

  // API呼び出しフック
  const { execute: fetchProgress } = useApiCall();
  const { execute: fetchResult } = useApiCall();
  const { execute: stopExperimentApi } = useApiCall();

  // ポーリング停止
  const stopPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    isPollingRef.current = false;
    setIsPolling(false);
  }, []);

  // 進捗取得
  const fetchProgressData = useCallback(async (experimentId: string) => {
    if (!isPollingRef.current) return;

    try {
      setIsLoading(true);
      const response = await fetchProgress(`/api/auto-strategy/experiments/${experimentId}/progress`);
      
      if (response?.success && response.progress) {
        const progressData = response.progress as GAProgress;
        setProgress(progressData);
        setError(null);

        // 進捗コールバック
        if (onProgress) {
          onProgress(progressData);
        }

        // 完了時の処理
        if (progressData.status === "completed") {
          stopPolling();
          
          try {
            const resultResponse = await fetchResult(`/api/auto-strategy/experiments/${experimentId}/results`);
            if (resultResponse?.success && resultResponse.result) {
              const resultData = resultResponse.result as GAResult;
              setResult(resultData);
              
              if (onComplete) {
                onComplete(resultData);
              }
            }
          } catch (resultError) {
            console.error("Failed to fetch result:", resultError);
            const errorMsg = "結果取得に失敗しました";
            setError(errorMsg);
            if (onError) {
              onError(errorMsg);
            }
          }
        }
        // エラー時の処理
        else if (progressData.status === "error") {
          stopPolling();
          const errorMsg = "GA実行中にエラーが発生しました";
          setError(errorMsg);
          if (onError) {
            onError(errorMsg);
          }
        }
      } else {
        throw new Error("Invalid progress response");
      }
    } catch (err) {
      console.error("Failed to fetch progress:", err);
      const errorMsg = "進捗取得に失敗しました";
      setError(errorMsg);
      stopPolling();
      
      if (onError) {
        onError(errorMsg);
      }
    } finally {
      setIsLoading(false);
    }
  }, [fetchProgress, fetchResult, onProgress, onComplete, onError, stopPolling]);

  // ポーリング開始
  const startPolling = useCallback((experimentId: string) => {
    // 既存のポーリングを停止
    stopPolling();
    
    // 状態リセット
    setProgress(null);
    setResult(null);
    setError(null);
    setCurrentExperimentId(experimentId);
    
    // ポーリング開始
    isPollingRef.current = true;
    setIsPolling(true);
    
    // 初回実行
    fetchProgressData(experimentId);
    
    // 定期ポーリング
    pollingIntervalRef.current = setInterval(() => {
      fetchProgressData(experimentId);
    }, pollingInterval);
  }, [fetchProgressData, pollingInterval, stopPolling]);

  // 実験停止
  const stopExperiment = useCallback(async (): Promise<boolean> => {
    if (!currentExperimentId) return false;

    try {
      const response = await stopExperimentApi(
        `/api/auto-strategy/experiments/${currentExperimentId}/stop`,
        { method: "POST" }
      );
      
      if (response?.success) {
        stopPolling();
        setProgress(prev => prev ? { ...prev, status: "error" } : null);
        return true;
      }
      return false;
    } catch (err) {
      console.error("Failed to stop experiment:", err);
      return false;
    }
  }, [currentExperimentId, stopExperimentApi, stopPolling]);

  // リセット
  const reset = useCallback(() => {
    stopPolling();
    setProgress(null);
    setResult(null);
    setError(null);
    setCurrentExperimentId(null);
  }, [stopPolling]);

  // クリーンアップ
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  return {
    progress,
    result,
    error,
    isPolling,
    isLoading,
    startPolling,
    stopPolling,
    stopExperiment,
    reset,
  };
};

/**
 * GA実行管理フック
 * 
 * GA実行の開始から完了まで一連の流れを管理します。
 */
export const useGAExecution = () => {
  const [isExecuting, setIsExecuting] = useState(false);
  const [experimentId, setExperimentId] = useState<string | null>(null);
  
  const { execute: startGA } = useApiCall();
  
  const gaProgress = useGAProgress({
    onComplete: (result) => {
      setIsExecuting(false);
      console.log("GA execution completed:", result);
    },
    onError: (error) => {
      setIsExecuting(false);
      console.error("GA execution error:", error);
    },
  });

  // GA実行開始
  const executeGA = useCallback(async (config: any) => {
    try {
      setIsExecuting(true);
      
      const response = await startGA("/api/auto-strategy/generate", {
        method: "POST",
        body: config,
      });
      
      if (response?.success && response.experiment_id) {
        const expId = response.experiment_id;
        setExperimentId(expId);
        gaProgress.startPolling(expId);
        return expId;
      } else {
        throw new Error(response?.message || "GA実行開始に失敗しました");
      }
    } catch (error) {
      setIsExecuting(false);
      throw error;
    }
  }, [startGA, gaProgress]);

  // リセット
  const reset = useCallback(() => {
    setIsExecuting(false);
    setExperimentId(null);
    gaProgress.reset();
  }, [gaProgress]);

  return {
    isExecuting,
    experimentId,
    executeGA,
    reset,
    ...gaProgress,
  };
};
