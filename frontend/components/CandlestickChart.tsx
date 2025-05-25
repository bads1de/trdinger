/**
 * ローソク足チャートコンポーネント
 *
 * ApexChartsを使用してローソク足チャートを表示するコンポーネントです。
 * 真のローソク足（キャンドルスティック）チャートを実装しています。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React, { useMemo } from "react";
import dynamic from "next/dynamic";
import { CandlestickData } from "@/types/strategy";

// ApexChartsを動的インポート（SSR対応）
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

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
 * OHLCデータをApexCharts形式に変換する関数
 */
const convertToApexChartsData = (data: CandlestickData[]) => {
  return data.map((item) => ({
    x: new Date(item.timestamp).getTime(),
    y: [item.open, item.high, item.low, item.close],
  }));
};

/**
 * ApexChartsのオプション設定を生成する関数
 */
const createChartOptions = (height: number) => {
  return {
    chart: {
      type: "candlestick" as const,
      height,
      toolbar: {
        show: true,
        tools: {
          download: true,
          selection: true,
          zoom: true,
          zoomin: true,
          zoomout: true,
          pan: true,
          reset: true,
        },
      },
      background: "transparent",
    },
    title: {
      text: "",
      align: "left" as const,
    },
    xaxis: {
      type: "datetime" as const,
      labels: {
        style: {
          colors: "#6B7280",
          fontSize: "12px",
        },
      },
    },
    yaxis: {
      tooltip: {
        enabled: true,
      },
      labels: {
        style: {
          colors: "#6B7280",
          fontSize: "12px",
        },
        formatter: (value: number) => `$${value.toFixed(2)}`,
      },
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: "#ffffff", // 陽線: 白色
          downward: "#808080", // 陰線: グレー色
        },
        wick: {
          useFillColor: true,
        },
      },
    },
    grid: {
      borderColor: "#E5E7EB",
      strokeDashArray: 3,
    },
    tooltip: {
      enabled: true,
      theme: "light" as const,
    },
  };
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
  // ApexChartsのオプションとシリーズデータをメモ化
  const chartOptions = useMemo(() => createChartOptions(height), [height]);

  const chartSeries = useMemo(() => {
    if (!data || data.length === 0) return [];
    return [
      {
        name: "ローソク足",
        data: convertToApexChartsData(data),
      },
    ];
  }, [data]);

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

  return (
    <div className="w-full">
      <Chart
        options={chartOptions}
        series={chartSeries}
        type="candlestick"
        height={height}
      />
    </div>
  );
};

export default CandlestickChart;
