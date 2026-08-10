/**
 * リターン分布チャートコンポーネント
 *
 * 取引リターンの統計分布をヒストグラムで表示するチャート
 */

"use client";

import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import ChartContainer from "./ChartContainer";
import { chartColors, chartStyles } from "./ChartTheme";
import { ChartTooltipContainer, TooltipRow } from "./ChartTooltip";
import { calculateReturnDistribution } from "@/utils/chartDataTransformers";
import { formatPercentage, formatNumber } from "@/utils/formatters";
import { Trade } from "@/types/backtest";

interface ReturnsDistributionChartProps {
  /** 取引履歴データ */
  data: Trade[];
  /** ヒストグラムのビン数 */
  bins?: number;
  /** ゼロライン（損益分岐点）を表示するか */
  showZeroLine?: boolean;
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
const CustomTooltip: React.FC<any> = ({ active, payload }) => {
  if (!active || !payload || !payload.length) {
    return null;
  }

  const data = payload[0].payload;

  return (
    <ChartTooltipContainer active={active} payload={payload} title="リターン範囲">
      <div className="space-y-1 text-sm">
        <TooltipRow
          label="範囲"
          value={`${formatPercentage(data.rangeStart, 1)} ～ ${formatPercentage(
            data.rangeEnd,
            1
          )}`}
        />
        <TooltipRow
          label="取引数"
          value={`${data.count}件`}
          valueClassName="text-blue-400 font-medium"
        />
        <TooltipRow
          label="頻度"
          value={formatPercentage(data.frequency, 1)}
          valueColor="green"
        />
      </div>
    </ChartTooltipContainer>
  );
};

/**
 * リターン分布チャートメインコンポーネント
 */
const ReturnsDistributionChart: React.FC<ReturnsDistributionChartProps> = ({
  data,
  bins = 20,
  showZeroLine = true,
  title,
  subtitle,
  actions,
  loading = false,
  error,
  height = 400,
  className = "",
}) => {
  // 分布データの計算
  const distributionData = useMemo(() => {
    if (!data || data.length === 0) {
      return [];
    }

    const distribution = calculateReturnDistribution(data, bins);

    // チャート表示用にラベルを追加
    return distribution.map((bin) => ({
      ...bin,
      rangeLabel: formatPercentage(bin.rangeStart, 1),
      color: bin.rangeStart >= 0 ? chartColors.winTrade : chartColors.lossTrade,
    }));
  }, [data, bins]);

  // Y軸のドメインを計算
  const yAxisDomain = useMemo(() => {
    if (!distributionData || distributionData.length === 0) {
      return [0, 100];
    }

    const maxFrequency = Math.max(...distributionData.map((d) => d.frequency));
    return [0, Math.ceil(maxFrequency * 1.1)]; // 10%のマージンを追加
  }, [distributionData]);

  // 統計情報の計算
  const statistics = useMemo(() => {
    if (!data || data.length === 0) {
      return null;
    }

    const returns = data.map((trade) => trade.return_pct * 100);
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance =
      returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) /
      returns.length;
    const stdDev = Math.sqrt(variance);
    const skewness =
      returns.reduce(
        (sum, ret) => sum + Math.pow((ret - mean) / stdDev, 3),
        0
      ) / returns.length;

    return {
      mean,
      stdDev,
      skewness,
      min: Math.min(...returns),
      max: Math.max(...returns),
    };
  }, [data]);

  return (
    <ChartContainer
      title={title}
      subtitle={subtitle}
      data={distributionData}
      loading={loading}
      error={error}
      height={height}
      className={className}
      actions={actions}
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={distributionData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid {...chartStyles.grid} />

          <XAxis
            dataKey="rangeLabel"
            {...chartStyles.axis}
            angle={-45}
            textAnchor="end"
            height={80}
          />

          <YAxis
            domain={yAxisDomain}
            tickFormatter={(value) => formatPercentage(value, 1)}
            {...chartStyles.axis}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* ゼロライン（損益分岐点） */}
          {showZeroLine && (
            <ReferenceLine
              x={0}
              stroke={chartColors.neutral}
              strokeDasharray="3 3"
              strokeOpacity={0.6}
              label={{ value: "損益分岐点", position: "top" }}
            />
          )}

          {/* 頻度バー */}
          <Bar
            dataKey="frequency"
            fill={chartColors.primary}
            radius={[2, 2, 0, 0]}
            name="頻度"
          />
        </BarChart>
      </ResponsiveContainer>

      {/* 統計情報の表示 */}
      {statistics && (
        <div className="mt-4 p-4 bg-gray-800/50 rounded-lg">
          <h4 className="text-sm font-semibold text-white mb-3">統計情報</h4>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-xs">
            <div className="text-center">
              <div className="text-gray-400">平均</div>
              <div className="text-white font-medium">
                {formatPercentage(statistics.mean)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">標準偏差</div>
              <div className="text-white font-medium">
                {formatPercentage(statistics.stdDev)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">歪度</div>
              <div className="text-white font-medium">
                {formatNumber(statistics.skewness, 2, 2)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">最小値</div>
              <div className="text-red-400 font-medium">
                {formatPercentage(statistics.min)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">最大値</div>
              <div className="text-green-400 font-medium">
                {formatPercentage(statistics.max)}
              </div>
            </div>
          </div>
        </div>
      )}
    </ChartContainer>
  );
};

export default ReturnsDistributionChart;
