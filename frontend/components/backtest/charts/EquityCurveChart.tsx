/**
 * 資産曲線チャートコンポーネント
 *
 * バックテスト結果の資産推移を時系列で表示するチャート
 */

"use client";

import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import ChartContainer from "./ChartContainer";
import { chartColors, chartStyles } from "./ChartTheme";
import { formatDateTime } from "@/utils/formatters";
import { formatCurrency } from "@/utils/financialFormatters";
import { sampleData } from "@/utils/chartDataTransformers";
import { ChartEquityPoint } from "@/types/backtest";

interface EquityCurveChartProps {
  /** チャート表示用の資産曲線データ */
  data: ChartEquityPoint[];
  /** 初期資金 */
  initialCapital: number;
  /** Buy & Hold リターン率（オプション） */
  buyHoldReturn?: number;
  /** Buy & Hold 比較線を表示するか */
  showBuyHold?: boolean;
  /** 初期資本の参照線を表示するか */
  showInitialCapital?: boolean;
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

  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
      <p className="text-white font-semibold mb-2">{date}</p>
      {payload.map((entry: any, index: number) => (
        <div key={index} className="flex items-center justify-between mb-1">
          <span className="text-sm mr-3" style={{ color: entry.color }}>
            {entry.name}:
          </span>
          <span className="text-white font-medium">
            {entry.dataKey === "equity" || entry.dataKey === "buyHold"
              ? formatCurrency(entry.value)
              : `${entry.value.toFixed(2)}%`}
          </span>
        </div>
      ))}
      {data.drawdown > 0 && (
        <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-600">
          <span className="text-sm text-red-400">ドローダウン:</span>
          <span className="text-red-400 font-medium">
            -{data.drawdown.toFixed(2)}%
          </span>
        </div>
      )}
    </div>
  );
};

/**
 * 資産曲線チャートメインコンポーネント
 */
const EquityCurveChart: React.FC<EquityCurveChartProps> = ({
  data,
  initialCapital,
  buyHoldReturn,
  showBuyHold = false,
  showInitialCapital = true,
  maxDataPoints = 1000,
  title,
  subtitle,
  actions,
  loading = false,
  error,
  height = 400,
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

    // Buy & Hold データの追加
    if (showBuyHold && buyHoldReturn !== undefined) {
      return sampledData.map((point, index) => {
        const progress = index / (sampledData.length - 1);
        // buyHoldReturnはパーセンテージ形式なので100で割って小数に変換
        const buyHoldEquity =
          initialCapital * (1 + (buyHoldReturn / 100) * progress);

        return {
          ...point,
          buyHold: buyHoldEquity,
        };
      });
    }

    return sampledData;
  }, [data, maxDataPoints, showBuyHold, buyHoldReturn]);

  // Y軸のドメインを計算
  const yAxisDomain = useMemo(() => {
    if (!processedData || processedData.length === 0) {
      return ["auto", "auto"];
    }

    const equityValues = processedData.map((d) => d.equity);
    const buyHoldValues =
      showBuyHold && buyHoldReturn !== undefined
        ? processedData.map((d) => d.buyHold || 0)
        : [];

    const allValues = [...equityValues, ...buyHoldValues];
    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);

    // 5%のマージンを追加
    const margin = (maxValue - minValue) * 0.05;

    // Y軸の最小値が0未満になることを許容するように修正
    // これにより、総リターンがマイナスの場合でも正確に表示される
    const yMin = minValue - margin;
    const yMax = maxValue + margin;

    // グラフが扁平になるのを防ぐため、最小の表示範囲を設けることも検討できる
    // 例: if (yMax - yMin < initialCapital * 0.1) { ... }

    return [yMin, yMax];
  }, [processedData, showBuyHold, buyHoldReturn, initialCapital]);

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
        <LineChart
          data={processedData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid {...chartStyles.grid} />

          <XAxis
            dataKey="date"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(timestamp) => formatDateTime(timestamp).dateTime}
            {...chartStyles.axis}
          />

          <YAxis
            domain={yAxisDomain}
            tickFormatter={(value) => formatCurrency(value)}
            {...chartStyles.axis}
          />

          <Tooltip content={<CustomTooltip />} />

          {(showBuyHold || showInitialCapital) && (
            <Legend {...chartStyles.legend} />
          )}

          {/* 初期資本の参照線 */}
          {showInitialCapital && (
            <ReferenceLine
              y={initialCapital}
              stroke={chartColors.neutral}
              strokeDasharray="5 5"
              strokeOpacity={0.6}
              label={{ value: "初期資本", position: "topRight" as any }}
            />
          )}

          {/* 資産曲線 */}
          <Line
            type="monotone"
            dataKey="equity"
            stroke={chartColors.equity}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0 }}
            name="資産額"
          />

          {/* Buy & Hold 比較線 */}
          {showBuyHold && buyHoldReturn !== undefined && (
            <Line
              type="monotone"
              dataKey="buyHold"
              stroke={chartColors.buyHold}
              strokeWidth={2}
              strokeDasharray="8 4"
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
              name="Buy & Hold"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
};

export default EquityCurveChart;
