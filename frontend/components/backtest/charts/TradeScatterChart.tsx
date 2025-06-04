/**
 * 取引散布図チャートコンポーネント
 *
 * 取引の利益/損失分布と取引サイズとの相関を可視化するチャート
 */

"use client";

import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import ChartContainer from "./ChartContainer";
import { chartColors, chartStyles, formatters } from "./ChartTheme";
import { sampleData } from "@/utils/chartDataTransformers";
import { ChartTradePoint } from "@/types/backtest";

interface TradeScatterChartProps {
  /** チャート表示用の取引データ */
  data: ChartTradePoint[];
  /** ゼロライン（損益分岐点）を表示するか */
  showZeroLine?: boolean;
  /** 取引タイプ（ロング/ショート）別に分けて表示するか */
  separateByType?: boolean;
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
const CustomTooltip: React.FC<any> = ({ active, payload }) => {
  if (!active || !payload || !payload.length) {
    return null;
  }

  const data = payload[0].payload;
  const entryDate = new Date(data.entryDate).toLocaleDateString("ja-JP");
  const exitDate = new Date(data.exitDate).toLocaleDateString("ja-JP");

  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
      <p className="text-white font-semibold mb-2">取引詳細</p>

      <div className="space-y-1 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">エントリー:</span>
          <span className="text-white">{entryDate}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">イグジット:</span>
          <span className="text-white">{exitDate}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">タイプ:</span>
          <span
            className={
              data.type === "long" ? "text-green-400" : "text-yellow-400"
            }
          >
            {data.type === "long" ? "ロング" : "ショート"}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">リターン:</span>
          <span className={data.isWin ? "text-green-400" : "text-red-400"}>
            {data.returnPct > 0 ? "+" : ""}
            {data.returnPct.toFixed(2)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">損益:</span>
          <span className={data.isWin ? "text-green-400" : "text-red-400"}>
            {formatters.currency(data.pnl)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">サイズ:</span>
          <span className="text-white">{data.size.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
};

/**
 * 取引散布図チャートメインコンポーネント
 */
const TradeScatterChart: React.FC<TradeScatterChartProps> = ({
  data,
  showZeroLine = true,
  separateByType = false,
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
      return { all: [], long: [], short: [] };
    }

    // パフォーマンス最適化のためのサンプリング
    const sampledData = sampleData(data, maxDataPoints);

    if (separateByType) {
      return {
        all: sampledData,
        long: sampledData.filter((trade) => trade.type === "long"),
        short: sampledData.filter((trade) => trade.type === "short"),
      };
    }

    return {
      all: sampledData,
      long: [],
      short: [],
    };
  }, [data, maxDataPoints, separateByType]);

  // Y軸のドメインを計算
  const yAxisDomain = useMemo(() => {
    if (!processedData.all || processedData.all.length === 0) {
      return [-10, 10]; // デフォルトで-10%から+10%
    }

    const returnValues = processedData.all.map((d) => d.returnPct);
    const minReturn = Math.min(...returnValues);
    const maxReturn = Math.max(...returnValues);

    // 10%のマージンを追加
    const margin = Math.max(Math.abs(minReturn), Math.abs(maxReturn)) * 0.1;

    return [minReturn - margin, maxReturn + margin];
  }, [processedData]);

  return (
    <ChartContainer
      title={title}
      subtitle={subtitle}
      data={processedData.all}
      loading={loading}
      error={error}
      height={height}
      className={className}
      actions={actions}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid {...chartStyles.grid} />

          <XAxis
            dataKey="entryDate"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(timestamp) => formatters.date(timestamp)}
            {...chartStyles.axis}
          />

          <YAxis
            dataKey="returnPct"
            domain={yAxisDomain}
            tickFormatter={(value) => `${value.toFixed(1)}%`}
            {...chartStyles.axis}
          />

          <Tooltip content={<CustomTooltip />} />

          {(separateByType || showZeroLine) && (
            <Legend {...chartStyles.legend} />
          )}

          {/* ゼロライン（損益分岐点） */}
          {showZeroLine && (
            <ReferenceLine
              y={0}
              stroke={chartColors.neutral}
              strokeDasharray="3 3"
              strokeOpacity={0.6}
              label={{ value: "損益分岐点", position: "topRight" as any }}
            />
          )}

          {/* 取引タイプ別表示 */}
          {separateByType ? (
            <>
              {/* ロング取引 */}
              {processedData.long.length > 0 && (
                <Scatter
                  data={processedData.long}
                  fill={chartColors.longTrade}
                  name="ロング取引"
                />
              )}

              {/* ショート取引 */}
              {processedData.short.length > 0 && (
                <Scatter
                  data={processedData.short}
                  fill={chartColors.shortTrade}
                  name="ショート取引"
                />
              )}
            </>
          ) : (
            /* 勝敗別表示 */
            <Scatter
              data={processedData.all}
              fill={chartColors.primary}
              name="取引"
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
};

export default TradeScatterChart;
