/**
 * 月次リターンヒートマップコンポーネント
 * 
 * 月別パフォーマンスの季節性を可視化するヒートマップ
 */

"use client";

import React, { useMemo } from 'react';
import ChartContainer from './ChartContainer';
import { chartColors } from './ChartTheme';
import { calculateMonthlyReturns } from '@/utils/chartDataTransformers';
import { EquityPoint } from '@/types/backtest';

interface MonthlyReturnsHeatmapProps {
  /** 資産曲線データ */
  data: EquityPoint[];
  /** カラースケールの凡例を表示するか */
  showLegend?: boolean;
  /** 統計情報を表示するか */
  showStatistics?: boolean;
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
  theme?: 'light' | 'dark';
}

const monthNames = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
];

/**
 * リターン値に基づいて色を取得
 */
const getReturnColor = (returnValue: number): string => {
  const intensity = Math.min(Math.abs(returnValue) / 10, 1); // 10%を最大強度とする
  
  if (returnValue > 0) {
    // 正のリターン: 緑系
    const alpha = Math.round(intensity * 255).toString(16).padStart(2, '0');
    return `#10B981${alpha}`; // emerald-500 with alpha
  } else if (returnValue < 0) {
    // 負のリターン: 赤系
    const alpha = Math.round(intensity * 255).toString(16).padStart(2, '0');
    return `#EF4444${alpha}`; // red-500 with alpha
  } else {
    // ゼロ: グレー
    return '#6B7280'; // gray-500
  }
};

/**
 * ヒートマップセルコンポーネント
 */
const HeatmapCell: React.FC<{
  year: number;
  month: number;
  returnValue: number | null;
  onClick?: () => void;
}> = ({ year, month, returnValue, onClick }) => {
  const backgroundColor = returnValue !== null ? getReturnColor(returnValue) : '#374151';
  const textColor = returnValue !== null && Math.abs(returnValue) > 5 ? '#FFFFFF' : '#9CA3AF';

  return (
    <div
      className="relative group cursor-pointer transition-all duration-200 hover:scale-105 hover:z-10"
      style={{ backgroundColor }}
      onClick={onClick}
      title={`${year}年${month}月: ${returnValue !== null ? `${returnValue.toFixed(2)}%` : 'データなし'}`}
    >
      <div 
        className="w-full h-full flex items-center justify-center text-xs font-medium border border-gray-600/30"
        style={{ color: textColor }}
      >
        {returnValue !== null ? `${returnValue.toFixed(1)}%` : '-'}
      </div>
      
      {/* ツールチップ */}
      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-20">
        {year}年{month}月: {returnValue !== null ? `${returnValue.toFixed(2)}%` : 'データなし'}
      </div>
    </div>
  );
};

/**
 * カラースケール凡例コンポーネント
 */
const ColorLegend: React.FC = () => {
  const legendValues = [-10, -5, 0, 5, 10];
  
  return (
    <div data-testid="color-legend" className="flex items-center justify-center space-x-2 mt-4">
      <span className="text-xs text-gray-400 mr-2">リターン:</span>
      {legendValues.map((value, index) => (
        <div key={index} className="flex flex-col items-center">
          <div
            className="w-6 h-4 border border-gray-600"
            style={{ backgroundColor: getReturnColor(value) }}
          />
          <span className="text-xs text-gray-400 mt-1">{value}%</span>
        </div>
      ))}
    </div>
  );
};

/**
 * 月次リターンヒートマップメインコンポーネント
 */
const MonthlyReturnsHeatmap: React.FC<MonthlyReturnsHeatmapProps> = ({
  data,
  showLegend = true,
  showStatistics = true,
  title,
  subtitle,
  actions,
  loading = false,
  error,
  height = 300,
  className = '',
  theme = 'dark'
}) => {
  // 月次リターンデータの計算
  const monthlyData = useMemo(() => {
    if (!data || data.length === 0) {
      return { heatmapData: [], years: [], statistics: null };
    }

    const monthlyReturns = calculateMonthlyReturns(data);
    
    // 年ごとにグループ化
    const yearGroups = monthlyReturns.reduce((acc, item) => {
      if (!acc[item.year]) {
        acc[item.year] = {};
      }
      acc[item.year][item.month] = item.return;
      return acc;
    }, {} as Record<number, Record<number, number>>);

    const years = Object.keys(yearGroups).map(Number).sort();
    
    // ヒートマップ用のデータ構造を作成
    const heatmapData = years.map(year => {
      const yearData = Array.from({ length: 12 }, (_, index) => {
        const month = index + 1;
        return {
          year,
          month,
          returnValue: yearGroups[year][month] || null,
        };
      });
      return { year, months: yearData };
    });

    // 統計情報の計算
    const allReturns = monthlyReturns.map(item => item.return);
    const statistics = allReturns.length > 0 ? {
      avgReturn: allReturns.reduce((sum, ret) => sum + ret, 0) / allReturns.length,
      bestMonth: Math.max(...allReturns),
      worstMonth: Math.min(...allReturns),
      positiveMonths: allReturns.filter(ret => ret > 0).length,
      totalMonths: allReturns.length,
    } : null;

    return { heatmapData, years, statistics };
  }, [data]);

  return (
    <ChartContainer
      title={title}
      subtitle={subtitle}
      data={monthlyData.heatmapData}
      loading={loading}
      error={error}
      height={height}
      className={className}
      actions={actions}
    >
      <div data-testid="heatmap-container" className="space-y-4">
        {/* 月のヘッダー */}
        <div className="grid grid-cols-13 gap-1">
          <div className="text-xs text-gray-400 font-medium"></div>
          {monthNames.map(month => (
            <div key={month} className="text-xs text-gray-400 font-medium text-center">
              {month}
            </div>
          ))}
        </div>

        {/* ヒートマップ本体 */}
        <div className="space-y-1">
          {monthlyData.heatmapData.map(({ year, months }) => (
            <div key={year} className="grid grid-cols-13 gap-1">
              {/* 年のラベル */}
              <div className="text-xs text-gray-400 font-medium flex items-center">
                {year}
              </div>
              
              {/* 月のセル */}
              {months.map(({ month, returnValue }) => (
                <div key={month} className="aspect-square">
                  <HeatmapCell
                    year={year}
                    month={month}
                    returnValue={returnValue}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* カラースケール凡例 */}
        {showLegend && <ColorLegend />}

        {/* 統計情報 */}
        {showStatistics && monthlyData.statistics && (
          <div data-testid="monthly-statistics" className="mt-4 p-4 bg-gray-800/50 rounded-lg">
            <h4 className="text-sm font-semibold text-white mb-3">月次統計</h4>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-xs">
              <div className="text-center">
                <div className="text-gray-400">平均月次リターン</div>
                <div className="text-white font-medium">
                  {monthlyData.statistics.avgReturn.toFixed(2)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-gray-400">最高月</div>
                <div className="text-green-400 font-medium">
                  {monthlyData.statistics.bestMonth.toFixed(2)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-gray-400">最低月</div>
                <div className="text-red-400 font-medium">
                  {monthlyData.statistics.worstMonth.toFixed(2)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-gray-400">プラス月数</div>
                <div className="text-blue-400 font-medium">
                  {monthlyData.statistics.positiveMonths}/{monthlyData.statistics.totalMonths}
                </div>
              </div>
              <div className="text-center">
                <div className="text-gray-400">勝率</div>
                <div className="text-purple-400 font-medium">
                  {((monthlyData.statistics.positiveMonths / monthlyData.statistics.totalMonths) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </ChartContainer>
  );
};

export default MonthlyReturnsHeatmap;
