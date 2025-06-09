/**
 * GA進捗表示コンポーネント
 * 
 * 遺伝的アルゴリズムの実行進捗をリアルタイムで表示します。
 */

"use client";

import React, { useState, useEffect } from "react";
import { useApiCall } from "@/hooks/useApiCall";

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

interface GAProgressDisplayProps {
  experimentId: string;
  onComplete?: (result: any) => void;
  onError?: (error: string) => void;
}

const GAProgressDisplay: React.FC<GAProgressDisplayProps> = ({
  experimentId,
  onComplete,
  onError,
}) => {
  const [progress, setProgress] = useState<GAProgress | null>(null);
  const [isPolling, setIsPolling] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { execute: fetchProgress } = useApiCall();
  const { execute: fetchResult } = useApiCall();
  const { execute: stopExperiment } = useApiCall();

  // 進捗ポーリング
  useEffect(() => {
    if (!isPolling || !experimentId) return;

    const pollProgress = async () => {
      try {
        const response = await fetchProgress(`/api/auto-strategy/experiments/${experimentId}/progress`);
        
        if (response?.success && response.progress) {
          const progressData = response.progress as GAProgress;
          setProgress(progressData);

          // 完了またはエラー時の処理
          if (progressData.status === "completed") {
            setIsPolling(false);
            
            // 結果を取得
            try {
              const resultResponse = await fetchResult(`/api/auto-strategy/experiments/${experimentId}/results`);
              if (resultResponse?.success && onComplete) {
                onComplete(resultResponse.result);
              }
            } catch (resultError) {
              console.error("Failed to fetch result:", resultError);
            }
            
          } else if (progressData.status === "error") {
            setIsPolling(false);
            const errorMsg = "GA実行中にエラーが発生しました";
            setError(errorMsg);
            if (onError) {
              onError(errorMsg);
            }
          }
        }
      } catch (err) {
        console.error("Failed to fetch progress:", err);
        setError("進捗取得に失敗しました");
        setIsPolling(false);
      }
    };

    // 初回実行
    pollProgress();

    // 定期ポーリング（5秒間隔）
    const interval = setInterval(pollProgress, 5000);

    return () => clearInterval(interval);
  }, [experimentId, isPolling, fetchProgress, fetchResult, onComplete, onError]);

  // 実験停止ハンドラー
  const handleStop = async () => {
    try {
      const response = await stopExperiment(`/api/auto-strategy/experiments/${experimentId}/stop`, {
        method: "POST",
      });
      
      if (response?.success) {
        setIsPolling(false);
        setProgress(prev => prev ? { ...prev, status: "error" } : null);
      }
    } catch (err) {
      console.error("Failed to stop experiment:", err);
    }
  };

  // 時間フォーマット関数
  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}時間${minutes}分${secs}秒`;
    } else if (minutes > 0) {
      return `${minutes}分${secs}秒`;
    } else {
      return `${secs}秒`;
    }
  };

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">エラー</h3>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!progress) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">進捗情報を読み込み中...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 基本進捗情報 */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">GA実行進捗</h3>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            progress.status === "running" ? "bg-blue-100 text-blue-800" :
            progress.status === "completed" ? "bg-green-100 text-green-800" :
            "bg-red-100 text-red-800"
          }`}>
            {progress.status === "running" ? "実行中" :
             progress.status === "completed" ? "完了" : "エラー"}
          </span>
        </div>

        {/* 進捗バー */}
        <div className="mb-4">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>世代 {progress.current_generation} / {progress.total_generations}</span>
            <span>{progress.progress_percentage.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress.progress_percentage}%` }}
            ></div>
          </div>
        </div>

        {/* 統計情報 */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-sm text-gray-600">最高フィットネス</p>
            <p className="text-lg font-semibold text-gray-900">
              {progress.best_fitness.toFixed(4)}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">平均フィットネス</p>
            <p className="text-lg font-semibold text-gray-900">
              {progress.average_fitness.toFixed(4)}
            </p>
          </div>
        </div>

        {/* 時間情報 */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-sm text-gray-600">実行時間</p>
            <p className="text-sm font-medium text-gray-900">
              {formatTime(progress.execution_time)}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">推定残り時間</p>
            <p className="text-sm font-medium text-gray-900">
              {progress.status === "running" ? formatTime(progress.estimated_remaining_time) : "-"}
            </p>
          </div>
        </div>

        {/* 停止ボタン */}
        {progress.status === "running" && (
          <div className="flex justify-end">
            <button
              onClick={handleStop}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 text-sm"
            >
              実行停止
            </button>
          </div>
        )}
      </div>

      {/* 最良戦略プレビュー */}
      {progress.best_strategy_preview && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">現在の最良戦略</h4>
          
          <div className="space-y-3">
            <div>
              <p className="text-sm text-gray-600">使用指標</p>
              <div className="flex flex-wrap gap-2 mt-1">
                {progress.best_strategy_preview.indicators.map((indicator, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-medium"
                  >
                    {indicator}
                  </span>
                ))}
              </div>
            </div>
            
            <div>
              <p className="text-sm text-gray-600">エントリー条件</p>
              <p className="text-sm font-mono bg-white p-2 rounded border">
                {progress.best_strategy_preview.entry_condition}
              </p>
            </div>
            
            <div>
              <p className="text-sm text-gray-600">イグジット条件</p>
              <p className="text-sm font-mono bg-white p-2 rounded border">
                {progress.best_strategy_preview.exit_condition}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GAProgressDisplay;
