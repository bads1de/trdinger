/**
 * Fear & Greed Index 状態表示カードコンポーネント
 *
 * Fear & Greed Index データの状態情報を表示します。
 */

import React from "react";
import { FearGreedDataStatus } from "@/hooks/useFearGreedData";

interface FearGreedStatusCardProps {
  status: FearGreedDataStatus | null;
  loading?: boolean;
}

/**
 * 日時をフォーマット
 */
const formatDateTime = (dateString: string | null): string => {
  if (!dateString) return "データなし";
  
  try {
    const date = new Date(dateString);
    return date.toLocaleString("ja-JP", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "Asia/Tokyo",
    });
  } catch {
    return dateString;
  }
};

/**
 * 日付のみをフォーマット
 */
const formatDate = (dateString: string | null): string => {
  if (!dateString) return "データなし";
  
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString("ja-JP", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      timeZone: "Asia/Tokyo",
    });
  } catch {
    return dateString;
  }
};

/**
 * データの新しさを判定
 */
const getDataFreshness = (latestTimestamp: string | null): {
  status: "fresh" | "stale" | "old" | "none";
  message: string;
  color: string;
} => {
  if (!latestTimestamp) {
    return {
      status: "none",
      message: "データなし",
      color: "text-gray-500 dark:text-gray-400",
    };
  }

  try {
    const latest = new Date(latestTimestamp);
    const now = new Date();
    const diffHours = (now.getTime() - latest.getTime()) / (1000 * 60 * 60);

    if (diffHours <= 24) {
      return {
        status: "fresh",
        message: "最新",
        color: "text-green-600 dark:text-green-400",
      };
    } else if (diffHours <= 48) {
      return {
        status: "stale",
        message: "やや古い",
        color: "text-yellow-600 dark:text-yellow-400",
      };
    } else {
      return {
        status: "old",
        message: "古い",
        color: "text-red-600 dark:text-red-400",
      };
    }
  } catch {
    return {
      status: "none",
      message: "不明",
      color: "text-gray-500 dark:text-gray-400",
    };
  }
};

const FearGreedStatusCard: React.FC<FearGreedStatusCardProps> = ({
  status,
  loading = false,
}) => {
  if (loading) {
    return (
      <div className="enterprise-card animate-pulse">
        <div className="p-4">
          <div className="h-5 bg-secondary-200 dark:bg-secondary-700 rounded mb-3"></div>
          <div className="space-y-2">
            <div className="h-4 bg-secondary-200 dark:bg-secondary-700 rounded"></div>
            <div className="h-4 bg-secondary-200 dark:bg-secondary-700 rounded w-3/4"></div>
            <div className="h-4 bg-secondary-200 dark:bg-secondary-700 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!status || !status.success) {
    return (
      <div className="enterprise-card">
        <div className="p-4">
          <div className="text-center text-red-600 dark:text-red-400">
            <p className="text-sm font-medium">❌ 状態取得エラー</p>
            <p className="text-xs mt-1">{status?.error || "状態情報を取得できませんでした"}</p>
          </div>
        </div>
      </div>
    );
  }

  const freshness = getDataFreshness(status.latest_timestamp);

  return (
    <div className="enterprise-card">
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-lg font-medium text-secondary-900 dark:text-secondary-100">
            😨 Fear & Greed Index
          </h4>
          <span className={`text-xs font-medium ${freshness.color}`}>
            {freshness.message}
          </span>
        </div>

        <div className="space-y-3">
          {/* データ件数 */}
          <div className="flex justify-between items-center">
            <span className="text-sm text-secondary-600 dark:text-secondary-400">
              総データ件数:
            </span>
            <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
              {status.data_range.total_count.toLocaleString()}件
            </span>
          </div>

          {/* データ範囲 */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600 dark:text-secondary-400">
                最古データ:
              </span>
              <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                {formatDate(status.data_range.oldest_data)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600 dark:text-secondary-400">
                最新データ:
              </span>
              <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                {formatDate(status.data_range.newest_data)}
              </span>
            </div>
          </div>

          {/* 最終更新時刻 */}
          <div className="flex justify-between items-center">
            <span className="text-sm text-secondary-600 dark:text-secondary-400">
              最終更新:
            </span>
            <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
              {formatDateTime(status.latest_timestamp)}
            </span>
          </div>

          {/* データ期間 */}
          {status.data_range.oldest_data && status.data_range.newest_data && (
            <div className="pt-2 border-t border-secondary-200 dark:border-secondary-700">
              <div className="flex justify-between items-center">
                <span className="text-sm text-secondary-600 dark:text-secondary-400">
                  データ期間:
                </span>
                <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                  {(() => {
                    try {
                      const oldest = new Date(status.data_range.oldest_data);
                      const newest = new Date(status.data_range.newest_data);
                      const diffDays = Math.ceil((newest.getTime() - oldest.getTime()) / (1000 * 60 * 60 * 24));
                      return `${diffDays}日間`;
                    } catch {
                      return "計算不可";
                    }
                  })()}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* データ収集推奨メッセージ */}
        {status.data_range.total_count === 0 && (
          <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <p className="text-xs text-blue-700 dark:text-blue-300 text-center">
              💡 データがありません。「全期間データ収集」を実行してください。
            </p>
          </div>
        )}

        {freshness.status === "old" && (
          <div className="mt-3 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <p className="text-xs text-yellow-700 dark:text-yellow-300 text-center">
              ⚠️ データが古くなっています。「差分データ収集」を実行してください。
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default FearGreedStatusCard;
