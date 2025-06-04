/**
 * チャート表示用データ変換ユーティリティ
 *
 * バックテストデータをチャート表示用の形式に変換する関数群
 */

import { format } from "date-fns";
import {
  EquityPoint,
  Trade,
  ChartEquityPoint,
  ChartTradePoint,
  MonthlyReturn,
  ReturnDistribution,
} from "@/types/backtest";

/**
 * ドローダウンを計算する
 *
 * @param equityCurve 資産曲線データ
 * @returns ドローダウンが追加された資産曲線データ
 */
export const calculateDrawdown = (
  equityCurve: EquityPoint[]
): EquityPoint[] => {
  if (!equityCurve || equityCurve.length === 0) {
    return [];
  }

  let peak = equityCurve[0].equity;

  return equityCurve.map((point) => {
    // 新しい最高値を更新
    if (point.equity > peak) {
      peak = point.equity;
    }

    // ドローダウン率を計算（0-1の範囲）
    const drawdown_pct = peak > 0 ? (peak - point.equity) / peak : 0;

    return {
      ...point,
      drawdown_pct,
    };
  });
};

/**
 * 資産曲線データをチャート表示用に変換する
 *
 * @param equityCurve 資産曲線データ
 * @returns チャート表示用の資産曲線データ
 */
export const transformEquityCurve = (
  equityCurve: EquityPoint[]
): ChartEquityPoint[] => {
  if (!equityCurve || equityCurve.length === 0) {
    return [];
  }

  // ドローダウンを計算
  const equityWithDrawdown = calculateDrawdown(equityCurve);

  return equityWithDrawdown.map((point) => ({
    date: new Date(point.timestamp).getTime(),
    equity: point.equity,
    drawdown: (point.drawdown_pct || 0) * 100, // パーセンテージに変換
    formattedDate: format(new Date(point.timestamp), "yyyy-MM-dd HH:mm"),
  }));
};

/**
 * 取引履歴をチャート表示用に変換する
 *
 * @param trades 取引履歴データ
 * @returns チャート表示用の取引データ
 */
export const transformTradeHistory = (trades: Trade[]): ChartTradePoint[] => {
  if (!trades || trades.length === 0) {
    return [];
  }

  return trades.map((trade) => ({
    entryDate: new Date(trade.entry_time).getTime(),
    exitDate: new Date(trade.exit_time).getTime(),
    pnl: trade.pnl,
    returnPct: trade.return_pct * 100, // パーセンテージに変換
    size: Math.abs(trade.size),
    type: trade.size > 0 ? "long" : "short",
    isWin: trade.pnl > 0,
  }));
};

/**
 * 月次リターンデータを生成する
 *
 * @param equityCurve 資産曲線データ
 * @returns 月次リターンデータ
 */
export const calculateMonthlyReturns = (
  equityCurve: EquityPoint[]
): MonthlyReturn[] => {
  if (!equityCurve || equityCurve.length === 0) {
    return [];
  }

  const monthlyData = new Map<string, { start: number; end: number }>();

  // 各月の開始と終了の資産額を記録
  equityCurve.forEach((point) => {
    const date = new Date(point.timestamp);
    const yearMonth = `${date.getFullYear()}-${String(
      date.getMonth() + 1
    ).padStart(2, "0")}`;

    if (!monthlyData.has(yearMonth)) {
      monthlyData.set(yearMonth, { start: point.equity, end: point.equity });
    } else {
      const existing = monthlyData.get(yearMonth)!;
      existing.end = point.equity; // 最後の値で更新
    }
  });

  // 月次リターンを計算
  const monthNames = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];

  return Array.from(monthlyData.entries())
    .map(([yearMonth, data]) => {
      const [year, month] = yearMonth.split("-").map(Number);
      const monthlyReturn =
        data.start > 0 ? (data.end - data.start) / data.start : 0;

      return {
        year,
        month,
        return: monthlyReturn * 100, // パーセンテージに変換
        monthName: monthNames[month - 1],
      };
    })
    .sort((a, b) => a.year - b.year || a.month - b.month);
};

/**
 * リターン分布を計算する
 *
 * @param trades 取引履歴データ
 * @param bins ヒストグラムのビン数（デフォルト: 20）
 * @returns リターン分布データ
 */
export const calculateReturnDistribution = (
  trades: Trade[],
  bins: number = 20
): ReturnDistribution[] => {
  if (!trades || trades.length === 0) {
    return [];
  }

  const returns = trades.map((trade) => trade.return_pct * 100); // パーセンテージに変換
  const minReturn = Math.min(...returns);
  const maxReturn = Math.max(...returns);
  const binWidth = (maxReturn - minReturn) / bins;

  const distribution: ReturnDistribution[] = [];

  for (let i = 0; i < bins; i++) {
    const rangeStart = minReturn + i * binWidth;
    const rangeEnd = minReturn + (i + 1) * binWidth;

    const count = returns.filter(
      (ret) =>
        ret >= rangeStart && (i === bins - 1 ? ret <= rangeEnd : ret < rangeEnd)
    ).length;

    const frequency = (count / trades.length) * 100;

    distribution.push({
      rangeStart,
      rangeEnd,
      count,
      frequency,
    });
  }

  return distribution;
};

/**
 * データをサンプリングする（パフォーマンス最適化用）
 *
 * @param data データ配列
 * @param maxPoints 最大ポイント数（デフォルト: 1000）
 * @returns サンプリングされたデータ
 */
export const sampleData = <T>(data: T[], maxPoints: number = 1000): T[] => {
  if (!data || data.length <= maxPoints) {
    return data;
  }

  const step = Math.ceil(data.length / maxPoints);
  return data.filter((_, index) => index % step === 0);
};

/**
 * Buy & Hold リターンを計算する
 *
 * @param equityCurve 資産曲線データ
 * @returns Buy & Hold リターン率
 */
export const calculateBuyHoldReturn = (equityCurve: EquityPoint[]): number => {
  if (!equityCurve || equityCurve.length < 2) {
    return 0;
  }

  const startEquity = equityCurve[0].equity;
  const endEquity = equityCurve[equityCurve.length - 1].equity;

  return startEquity > 0 ? (endEquity - startEquity) / startEquity : 0;
};

/**
 * 最大ドローダウンを計算する
 *
 * @param equityCurve 資産曲線データ
 * @returns 最大ドローダウン率（0-1の範囲）
 */
export const calculateMaxDrawdown = (equityCurve: EquityPoint[]): number => {
  if (!equityCurve || equityCurve.length === 0) {
    return 0;
  }

  const equityWithDrawdown = calculateDrawdown(equityCurve);
  return Math.max(
    ...equityWithDrawdown.map((point) => point.drawdown_pct || 0)
  );
};

/**
 * 日付範囲でデータをフィルタリングする
 *
 * @param data データ配列
 * @param startDate 開始日
 * @param endDate 終了日
 * @param dateField 日付フィールド名
 * @returns フィルタリングされたデータ
 */
export const filterByDateRange = <T extends Record<string, any>>(
  data: T[],
  startDate: Date,
  endDate: Date,
  dateField: keyof T
): T[] => {
  return data.filter((item) => {
    const itemDate = new Date(item[dateField]);
    return itemDate >= startDate && itemDate <= endDate;
  });
};
