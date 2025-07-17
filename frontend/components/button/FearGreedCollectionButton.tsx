/**
 * Fear & Greed Index データ収集ボタンコンポーネント
 *
 * Fear & Greed Index データの収集機能を提供します。
 */

import React, { useState } from "react";
import { FearGreedCollectionResult } from "@/hooks/useFearGreedData";

interface FearGreedCollectionButtonProps {
  onCollectionStart: (result: FearGreedCollectionResult) => void;
  onCollectionError: (errorMessage: string) => void;
  disabled?: boolean;
  className?: string;
}

const FearGreedCollectionButton: React.FC<FearGreedCollectionButtonProps> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  const [isCollecting, setIsCollecting] = useState(false);

  /**
   * 通常のデータ収集を実行
   */
  const handleCollectData = async (limit: number = 30) => {
    if (isCollecting) return;

    setIsCollecting(true);
    try {
      const response = await fetch(`/api/data/fear-greed/collect?limit=${limit}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`データ収集に失敗しました: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        onCollectionStart(result.data);
      } else {
        throw new Error(result.message || "データ収集に失敗しました");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "データ収集中にエラーが発生しました";
      onCollectionError(errorMessage);
      console.error("Fear & Greed Index データ収集エラー:", error);
    } finally {
      setIsCollecting(false);
    }
  };

  /**
   * 履歴データ収集を実行（全期間）
   */
  const handleCollectHistoricalData = async () => {
    if (isCollecting) return;

    setIsCollecting(true);
    try {
      const response = await fetch("/api/data/fear-greed/collect-historical?limit=1000", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`履歴データ収集に失敗しました: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        onCollectionStart(result.data);
      } else {
        throw new Error(result.message || "履歴データ収集に失敗しました");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "履歴データ収集中にエラーが発生しました";
      onCollectionError(errorMessage);
      console.error("Fear & Greed Index 履歴データ収集エラー:", error);
    } finally {
      setIsCollecting(false);
    }
  };

  /**
   * 差分データ収集を実行
   */
  const handleCollectIncrementalData = async () => {
    if (isCollecting) return;

    setIsCollecting(true);
    try {
      const response = await fetch("/api/data/fear-greed/collect-incremental", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`差分データ収集に失敗しました: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        onCollectionStart(result.data);
      } else {
        throw new Error(result.message || "差分データ収集に失敗しました");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "差分データ収集中にエラーが発生しました";
      onCollectionError(errorMessage);
      console.error("Fear & Greed Index 差分データ収集エラー:", error);
    } finally {
      setIsCollecting(false);
    }
  };

  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-lg font-medium text-secondary-900 dark:text-secondary-100">
          😨 Fear & Greed Index
        </h4>
        <span className="text-xs text-secondary-500 dark:text-secondary-400">
          Alternative.me API
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* 最新データ収集 */}
        <button
          onClick={() => handleCollectData(30)}
          disabled={disabled || isCollecting}
          className="btn-primary text-sm py-2 px-4 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isCollecting ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              収集中...
            </div>
          ) : (
            "最新データ収集"
          )}
        </button>

        {/* 全期間データ収集 */}
        <button
          onClick={handleCollectHistoricalData}
          disabled={disabled || isCollecting}
          className="btn-secondary text-sm py-2 px-4 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isCollecting ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
              収集中...
            </div>
          ) : (
            "全期間データ収集"
          )}
        </button>

        {/* 差分データ収集 */}
        <button
          onClick={handleCollectIncrementalData}
          disabled={disabled || isCollecting}
          className="btn-outline text-sm py-2 px-4 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isCollecting ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
              収集中...
            </div>
          ) : (
            "差分データ収集"
          )}
        </button>
      </div>

      <div className="text-xs text-secondary-500 dark:text-secondary-400 mt-2">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-center">
          <span>最新30件</span>
          <span>全履歴（最大1000件）</span>
          <span>未収集データのみ</span>
        </div>
      </div>
    </div>
  );
};

export default FearGreedCollectionButton;
