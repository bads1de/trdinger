/**
 * Fear & Greed Index データテーブルコンポーネント
 *
 * Fear & Greed Index データを表形式で表示します。
 */

import React from "react";
import { FearGreedIndexData } from "@/app/api/data/fear-greed/route";

interface FearGreedDataTableProps {
  data: FearGreedIndexData[];
  loading: boolean;
  error: string | null;
}

/**
 * Fear & Greed Index 値に基づく色を取得
 */
const getValueColor = (value: number): string => {
  if (value <= 20) return "text-red-600 dark:text-red-400"; // Extreme Fear
  if (value <= 40) return "text-orange-600 dark:text-orange-400"; // Fear
  if (value <= 60) return "text-yellow-600 dark:text-yellow-400"; // Neutral
  if (value <= 80) return "text-green-600 dark:text-green-400"; // Greed
  return "text-emerald-600 dark:text-emerald-400"; // Extreme Greed
};

/**
 * Fear & Greed Index 分類に基づく背景色を取得
 */
const getClassificationBadge = (classification: string): string => {
  switch (classification) {
    case "Extreme Fear":
      return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
    case "Fear":
      return "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200";
    case "Neutral":
      return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
    case "Greed":
      return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    case "Extreme Greed":
      return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200";
    default:
      return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
  }
};

/**
 * 日時をフォーマット
 */
const formatDateTime = (dateString: string): string => {
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
const formatDate = (dateString: string): string => {
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

const FearGreedDataTable: React.FC<FearGreedDataTableProps> = ({
  data,
  loading,
  error,
}) => {
  if (loading) {
    return (
      <div className="enterprise-card animate-pulse">
        <div className="p-6">
          <div className="h-6 bg-secondary-200 dark:bg-secondary-700 rounded mb-4"></div>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="h-4 bg-secondary-200 dark:bg-secondary-700 rounded"
              ></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="enterprise-card">
        <div className="p-6">
          <div className="text-center text-red-600 dark:text-red-400">
            <p className="text-lg font-medium mb-2">❌ エラーが発生しました</p>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="enterprise-card">
        <div className="p-6">
          <div className="text-center text-secondary-600 dark:text-secondary-400">
            <p className="text-lg font-medium mb-2">📊 データがありません</p>
            <p className="text-sm">
              Fear & Greed Index データを収集してください
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="enterprise-card">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
            Fear & Greed Index データ
          </h3>
          <span className="badge-primary">
            {data.length.toLocaleString()}件
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-secondary-200 dark:divide-secondary-700">
            <thead className="bg-secondary-50 dark:bg-secondary-800">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                  日付
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                  値
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                  分類
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                  取得時刻
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-secondary-900 divide-y divide-secondary-200 dark:divide-secondary-700">
              {data.map((item, index) => (
                <tr
                  key={item.id || index}
                  className="hover:bg-secondary-50 dark:hover:bg-secondary-800 transition-colors duration-150"
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-secondary-900 dark:text-secondary-100">
                    {formatDate(item.data_timestamp)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span
                        className={`text-2xl font-bold ${getValueColor(
                          item.value
                        )}`}
                      >
                        {item.value}
                      </span>
                      <span className="ml-2 text-sm text-secondary-500 dark:text-secondary-400">
                        / 100
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getClassificationBadge(
                        item.value_classification
                      )}`}
                    >
                      {item.value_classification}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-500 dark:text-secondary-400">
                    {formatDateTime(item.timestamp)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {data.length > 0 && (
          <div className="mt-4 text-sm text-secondary-500 dark:text-secondary-400 text-center">
            最新データ: {formatDate(data[0]?.data_timestamp)} | データ範囲:{" "}
            {formatDate(data[data.length - 1]?.data_timestamp)} ～{" "}
            {formatDate(data[0]?.data_timestamp)}
          </div>
        )}
      </div>
    </div>
  );
};

export default FearGreedDataTable;
