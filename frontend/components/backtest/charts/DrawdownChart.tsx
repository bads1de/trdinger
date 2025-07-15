/**
 * ドローダウンチャートコンポーネント
 *
 * バックテスト結果のドローダウン期間を可視化するチャート
 */

"use client";

import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import ChartContainer from "./ChartContainer";
import { chartColors, chartStyles } from "./ChartTheme";
import { sampleData } from "@/utils/chartDataTransformers";
import { formatDateTime } from "@/utils/formatters";
import { formatCurrency } from "@/utils/financialFormatters";
import { ChartEquityPoint } from "@/types/backtest";

interface DrawdownChartProps {
  /** チャート表示用の資産曲線データ（ドローダウン情報を含む） */
  data: ChartEquityPoint[];
  /** 最大ドローダウン率（パーセンテージ） */
  maxDrawdown?: number;
  /** 最大ドローダウンの参照線を表示するか */
  showMaxDrawdown?: boolean;
  /** 最大データポイント数（パフォーマンス最適化用） */
  maxDataPoints?: number;
  /** チャートタイトル */
  title: string;
  /** サブタイトル（オプション） */
  subtitle?: string;
  /** アクションボタン（オプション） */
  actions?: React.ReactNode;
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** チャートの高さ */
  height?: number;
  /** 追加のCSSクラス */
  className?: string;
  /** テーマ */
  theme?: "light" | "dark";
}

/**
 * カスタムツールチップコンポーネント
 */
const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) {
    return null;
  }

  const data = payload[0].payload;
  const date = new Date(label).toLocaleDateString("ja-JP");
  const drawdown = data.drawdown || 0;

  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
      <p className="text-white font-semibold mb-2">{date}</p>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm text-red-400 mr-3">ドローダウン:</span>
        <span className="text-red-400 font-medium">
          -{drawdown.toFixed(2)}%
        </span>
      </div>
      {data.equity && (
        <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-600">
          <span className="text-sm text-gray-400">資産額:</span>
          <span className="text-white font-medium">
            {formatCurrency(data.equity)}
          </span>
        </div>
      )}
    </div>
  );
};

/**
 * ドローダウンチャートメインコンポーネント
 */
const DrawdownChart: React.FC<DrawdownChartProps> = ({
  data,
  maxDrawdown,
  showMaxDrawdown = true,
  maxDataPoints = 1000,
  title,
  subtitle,
  actions,
  loading = false,
  error,
  height = 350,
  className = "",
  theme = "dark",
}) => {
  // データの前処理とサンプリング
  const processedData = useMemo(() => {
    if (!data || data.length === 0) {
      return [];
    }

    // パフォーマンス最適化のためのサンプリング
    const sampledData = sampleData(data, maxDataPoints);

    // ドローダウンを負の値として表示（チャートでは下向きに表示）
    return sampledData.map((point) => ({
      ...point,
      drawdown: -(point.drawdown || 0), // 負の値に変換
    }));
  }, [data, maxDataPoints]);

  // Y軸のドメインを計算
  const yAxisDomain = useMemo(() => {
    if (!processedData || processedData.length === 0) {
      return [-20, 0]; // デフォルトで-20%から0%
    }

    const drawdownValues = processedData.map((d) => d.drawdown);
    const minDrawdown = Math.min(...drawdownValues);

    // 最小値に10%のマージンを追加（より深いドローダウンを表示）
    const margin = Math.abs(minDrawdown) * 0.1;

    return [
      minDrawdown - margin,
      0, // 最大値は常に0%
    ];
  }, [processedData]);

  // 最大ドローダウンの値を計算
  const calculatedMaxDrawdown = useMemo(() => {
    if (maxDrawdown !== undefined) {
      return -Math.abs(maxDrawdown); // 負の値に変換
    }

    if (!processedData || processedData.length === 0) {
      return 0;
    }

    return Math.min(...processedData.map((d) => d.drawdown));
  }, [maxDrawdown, processedData]);

  return (
    <ChartContainer
      title={title}
      subtitle={subtitle}
      data={processedData}
      loading={loading}
      error={error}
      height={height}
      className={className}
      actions={actions}
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={processedData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid {...chartStyles.grid} />

          <XAxis
            dataKey="date"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(timestamp) => formatDateTime(timestamp)}
            {...chartStyles.axis}
          />

          <YAxis
            domain={yAxisDomain}
            tickFormatter={(value) => `${Math.abs(value).toFixed(1)}%`}
            {...chartStyles.axis}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* 最大ドローダウンの参照線 */}
          {showMaxDrawdown && calculatedMaxDrawdown < 0 && (
            <ReferenceLine
              y={calculatedMaxDrawdown}
              stroke={chartColors.maxDrawdown}
              strokeDasharray="5 5"
              strokeWidth={2}
              label={{
                value: `最大DD: ${Math.abs(calculatedMaxDrawdown).toFixed(1)}%`,
                position: "topLeft" as any,
                style: { fill: chartColors.maxDrawdown },
              }}
            />
          )}

          {/* ゼロライン（ドローダウンなし） */}
          <ReferenceLine
            y={0}
            stroke={chartColors.neutral}
            strokeWidth={1}
            strokeOpacity={0.5}
          />

          {/* ドローダウンエリア */}
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke={chartColors.drawdown}
            fill={chartColors.drawdownFill}
            strokeWidth={2}
            fillOpacity={0.3}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
};

export default DrawdownChart;
