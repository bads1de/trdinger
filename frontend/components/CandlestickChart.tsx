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
        className="flex items-center justify-center bg-error-50 dark:bg-error-900/20 rounded-enterprise-lg border border-error-200 dark:border-error-800"
        style={{ height }}
      >
        <div className="text-center p-8">
          <div className="w-16 h-16 mx-auto mb-4 bg-error-100 dark:bg-error-900/50 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-error-600 dark:text-error-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-error-800 dark:text-error-200 mb-2">
            📊 チャートの読み込みに失敗しました
          </h3>
          <p className="text-sm text-error-600 dark:text-error-400 max-w-md">{error}</p>
          <div className="mt-4">
            <span className="badge-error">エラー</span>
          </div>
        </div>
      </div>
    );
  }

  // ローディング状態
  if (loading) {
    return (
      <div
        className="flex items-center justify-center bg-secondary-50 dark:bg-secondary-900/50 rounded-enterprise-lg border border-secondary-200 dark:border-secondary-700"
        style={{ height }}
      >
        <div className="text-center p-8">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 border-4 border-secondary-200 dark:border-secondary-700 rounded-full"></div>
            <div className="absolute inset-0 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
            <div className="absolute inset-2 bg-primary-100 dark:bg-primary-900/50 rounded-full flex items-center justify-center">
              <svg className="w-6 h-6 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
          </div>
          <h3 className="text-lg font-semibold text-secondary-800 dark:text-secondary-200 mb-2">
            📈 チャートデータを読み込み中
          </h3>
          <p className="text-sm text-secondary-600 dark:text-secondary-400">
            高精度なローソク足データを取得しています...
          </p>
          <div className="mt-4">
            <span className="badge-primary animate-pulse">読み込み中</span>
          </div>
        </div>
      </div>
    );
  }

  // データが空の場合
  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-secondary-50 dark:bg-secondary-900/50 rounded-enterprise-lg border border-secondary-200 dark:border-secondary-700"
        style={{ height }}
      >
        <div className="text-center p-8">
          <div className="w-16 h-16 mx-auto mb-4 bg-secondary-100 dark:bg-secondary-800 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-secondary-500 dark:text-secondary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-secondary-700 dark:text-secondary-300 mb-2">
            📊 データがありません
          </h3>
          <p className="text-sm text-secondary-600 dark:text-secondary-400">
            選択した通貨ペアと時間軸の組み合わせでは、表示できるデータがありません
          </p>
          <div className="mt-4">
            <span className="badge-warning">データなし</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full relative">
      {/* チャートコンテナ */}
      <div className="bg-white dark:bg-secondary-900 rounded-enterprise-lg border border-secondary-200 dark:border-secondary-700 overflow-hidden">
        <Chart
          options={chartOptions}
          series={chartSeries}
          type="candlestick"
          height={height}
        />
      </div>

      {/* チャート情報オーバーレイ */}
      <div className="absolute top-4 right-4 flex items-center gap-2">
        <span className="badge-success text-xs">
          📊 ライブデータ
        </span>
        <span className="badge-primary text-xs">
          {data.length} ポイント
        </span>
      </div>
    </div>
  );
};

export default CandlestickChart;
