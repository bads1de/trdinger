/**
 * ローソク足チャートコンポーネント
 *
 * Rechartsを使用してローソク足チャートを表示するコンポーネントです。
 * シンプルなラインチャートとして実装し、後でローソク足に拡張可能です。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { CandlestickData } from "@/types/strategy";

/**
 * ローソク足チャートのプロパティ
 */
interface CandlestickChartProps {
  /** ローソク足データ */
  data: CandlestickData[];
  /** チャートの高さ（ピクセル） */
  height?: number;
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
}

/**
 * カスタムツールチップコンポーネント
 */
interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({
  active,
  payload,
  label,
}) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as CandlestickData;
    const isPositive = data.close >= data.open;
    const change = data.close - data.open;
    const changePercent = ((change / data.open) * 100).toFixed(2);

    return (
      <div className="bg-white dark:bg-gray-800 p-4 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg">
        <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
          {new Date(label || "").toLocaleString("ja-JP")}
        </p>
        <div className="mt-2 space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">始値:</span>
            <span className="font-medium">${data.open.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">高値:</span>
            <span className="font-medium">${data.high.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">安値:</span>
            <span className="font-medium">${data.low.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">終値:</span>
            <span className="font-medium">${data.close.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">変動:</span>
            <span
              className={`font-medium ${
                isPositive ? "text-green-600" : "text-red-600"
              }`}
            >
              {isPositive ? "+" : ""}${change.toFixed(2)} ({changePercent}%)
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">出来高:</span>
            <span className="font-medium">{data.volume.toLocaleString()}</span>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

/**
 * ローソク足チャートコンポーネント
 */
const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  height = 400,
  loading = false,
  error,
}) => {
  // エラー状態（ローディングより優先）
  if (error) {
    return (
      <div
        className="flex items-center justify-center bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800"
        style={{ height }}
      >
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400 font-medium">
            チャートの読み込みに失敗しました
          </p>
          <p className="mt-1 text-sm text-red-500 dark:text-red-300">{error}</p>
        </div>
      </div>
    );
  }

  // ローディング状態
  if (loading) {
    return (
      <div
        className="flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded-lg"
        style={{ height }}
      >
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            チャートデータを読み込み中...
          </p>
        </div>
      </div>
    );
  }

  // データが空の場合
  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded-lg"
        style={{ height }}
      >
        <p className="text-gray-600 dark:text-gray-400">
          表示するデータがありません
        </p>
      </div>
    );
  }

  // チャートデータの変換（Rechartsで使用するため）
  const chartData = data.map((item, index) => ({
    ...item,
    // インデックスを使用してX軸の値とする
    index,
    // 日時の表示用
    dateLabel: new Date(item.timestamp).toLocaleDateString("ja-JP"),
  }));

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={chartData}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
          <XAxis
            dataKey="index"
            tickFormatter={(value) => {
              const item = chartData[value];
              return item ? item.dateLabel : "";
            }}
            className="text-xs"
          />
          <YAxis
            domain={["dataMin - 50", "dataMax + 50"]}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
            className="text-xs"
          />
          <Tooltip content={<CustomTooltip />} />

          {/* 終値のライン */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#2563eb" }}
          />

          {/* 高値のライン（薄い色） */}
          <Line
            type="monotone"
            dataKey="high"
            stroke="#10b981"
            strokeWidth={1}
            strokeOpacity={0.5}
            dot={false}
          />

          {/* 安値のライン（薄い色） */}
          <Line
            type="monotone"
            dataKey="low"
            stroke="#ef4444"
            strokeWidth={1}
            strokeOpacity={0.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CandlestickChart;
