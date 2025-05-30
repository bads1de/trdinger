/**
 * データテーブルのカラム定義設定
 *
 * 各データテーブルのカラム定義を統一的に管理します。
 * フォーマッター関数とスタイリングを含む完全なカラム設定を提供します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import React from "react";
import { TableColumn } from "@/components/DataTable";
import {
  FundingRateData,
  PriceData,
  OpenInterestData,
  TechnicalIndicatorData,
} from "@/types/strategy";
import {
  formatDateTime,
  formatPrice,
  formatSymbol,
  formatFundingRate,
  getFundingRateColor,
  formatCurrency,
  formatVolume,
  getPriceChangeColor,
  formatCompactNumber,
} from "@/utils/formatters";

/**
 * ファンディングレートデータテーブルのカラム定義
 */
export const fundingRateColumns: TableColumn<FundingRateData>[] = [
  {
    key: "symbol",
    header: "通貨ペア",
    width: "120px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-semibold text-primary-400">
        {formatSymbol(value)}
      </span>
    ),
  },
  {
    key: "funding_rate",
    header: "FR",
    width: "150px",
    sortable: true,
    formatter: (value: number) => (
      <span
        className={`font-mono text-sm font-semibold ${getFundingRateColor(
          value
        )}`}
      >
        {formatFundingRate(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "funding_timestamp",
    header: "ファンディング時刻",
    width: "180px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-mono text-sm">{formatDateTime(value)}</span>
    ),
  },
  {
    key: "mark_price",
    header: "マーク価格",
    width: "120px",
    sortable: true,
    formatter: (value: number | null) => (
      <span className="font-mono text-sm text-blue-400">
        {formatPrice(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "index_price",
    header: "インデックス価格",
    width: "120px",
    sortable: true,
    formatter: (value: number | null) => (
      <span className="font-mono text-sm text-purple-400">
        {formatPrice(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "next_funding_timestamp",
    header: "次回ファンディング",
    width: "180px",
    sortable: true,
    formatter: (value: string | null) => (
      <span className="font-mono text-sm text-gray-400">
        {value ? formatDateTime(value) : "-"}
      </span>
    ),
  },
  {
    key: "timestamp",
    header: "取得時刻",
    width: "180px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-mono text-xs text-gray-500">
        {formatDateTime(value)}
      </span>
    ),
  },
];

/**
 * OHLCVデータテーブルのカラム定義
 */
export const ohlcvColumns: TableColumn<PriceData>[] = [
  {
    key: "timestamp",
    header: "日時",
    width: "180px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-mono text-sm">{formatDateTime(value)}</span>
    ),
  },
  {
    key: "open",
    header: "始値",
    width: "120px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm">{formatCurrency(value)}</span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "high",
    header: "高値",
    width: "120px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm text-green-400">
        {formatCurrency(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "low",
    header: "安値",
    width: "120px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm text-red-400">
        {formatCurrency(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "close",
    header: "終値",
    width: "120px",
    sortable: true,
    formatter: (value: number, row: PriceData) => (
      <span
        className={`font-mono text-sm font-semibold ${getPriceChangeColor(
          row.open,
          value
        )}`}
      >
        {formatCurrency(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "volume",
    header: "出来高",
    width: "100px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm text-blue-400">
        {formatVolume(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
];

/**
 * オープンインタレストデータテーブルのカラム定義
 */
export const openInterestColumns: TableColumn<OpenInterestData>[] = [
  {
    key: "symbol",
    header: "通貨ペア",
    width: "120px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-semibold text-primary-400">
        {formatSymbol(value)}
      </span>
    ),
  },
  {
    key: "open_interest_value",
    header: "OI値 (USD)",
    width: "150px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm font-semibold text-green-400">
        {formatCurrency(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "open_interest_amount",
    header: "OI量",
    width: "120px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm text-blue-400">
        {formatCompactNumber(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "data_timestamp",
    header: "データ時刻",
    width: "180px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-mono text-sm text-gray-400">
        {formatDateTime(value)}
      </span>
    ),
  },
  {
    key: "timestamp",
    header: "取得時刻",
    width: "180px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-mono text-xs text-gray-500">
        {formatDateTime(value)}
      </span>
    ),
  },
];

/**
 * テクニカル指標値のフォーマット
 */
const formatIndicatorValue = (value: number, indicatorType: string): string => {
  if (indicatorType === "RSI") {
    // RSIは0-100の範囲で小数点2桁
    return value.toFixed(2);
  } else {
    // SMA, EMAは価格なので適切な桁数で表示
    if (value >= 1000) {
      return value.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });
    } else {
      return value.toFixed(4);
    }
  }
};

/**
 * 指標タイプの色を取得
 */
const getIndicatorTypeColor = (indicatorType: string): string => {
  switch (indicatorType) {
    case "SMA":
      return "text-blue-400";
    case "EMA":
      return "text-green-400";
    case "RSI":
      return "text-purple-400";
    case "MACD":
      return "text-orange-400";
    case "BB":
      return "text-cyan-400";
    case "ATR":
      return "text-yellow-400";
    case "STOCH":
      return "text-pink-400";
    case "CCI":
      return "text-indigo-400";
    case "WILLR":
      return "text-red-400";
    case "MOM":
      return "text-lime-400";
    case "ROC":
      return "text-emerald-400";
    case "PSAR":
      return "text-violet-400";
    default:
      return "text-gray-400";
  }
};

/**
 * RSI値の色を取得（買われすぎ・売られすぎ判定）
 */
const getRSIColor = (value: number): string => {
  if (value >= 70) {
    return "text-red-400"; // 買われすぎ
  } else if (value <= 30) {
    return "text-green-400"; // 売られすぎ
  } else {
    return "text-gray-300"; // 中立
  }
};

/**
 * テクニカル指標データテーブルのカラム定義
 */
export const technicalIndicatorColumns: TableColumn<TechnicalIndicatorData>[] = [
  {
    key: "symbol",
    header: "通貨ペア",
    width: "100px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-semibold text-primary-400">
        {formatSymbol(value)}
      </span>
    ),
  },
  {
    key: "timeframe",
    header: "時間枠",
    width: "80px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-mono text-sm text-gray-300">
        {value}
      </span>
    ),
  },
  {
    key: "indicator_type",
    header: "指標",
    width: "80px",
    sortable: true,
    formatter: (value: string) => (
      <span className={`font-semibold text-sm ${getIndicatorTypeColor(value)}`}>
        {value}
      </span>
    ),
  },
  {
    key: "period",
    header: "期間",
    width: "60px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm text-gray-300">
        {value}
      </span>
    ),
    cellClassName: "text-center",
  },
  {
    key: "value",
    header: "値",
    width: "120px",
    sortable: true,
    formatter: (value: number, row: TechnicalIndicatorData) => {
      const formattedValue = formatIndicatorValue(value, row.indicator_type);
      const colorClass = row.indicator_type === "RSI"
        ? getRSIColor(value)
        : "text-yellow-400";

      return (
        <span className={`font-mono text-sm font-semibold ${colorClass}`}>
          {formattedValue}
        </span>
      );
    },
    cellClassName: "text-right",
  },
  {
    key: "signal_value",
    header: "シグナル",
    width: "100px",
    sortable: true,
    formatter: (value: number | null, row: TechnicalIndicatorData) => {
      if (value === null || value === undefined) {
        return <span className="text-gray-500">-</span>;
      }
      const formattedValue = formatIndicatorValue(value, row.indicator_type);
      return (
        <span className="font-mono text-sm text-cyan-400">
          {formattedValue}
        </span>
      );
    },
    cellClassName: "text-right",
  },
  {
    key: "histogram_value",
    header: "ヒストグラム",
    width: "100px",
    sortable: true,
    formatter: (value: number | null, row: TechnicalIndicatorData) => {
      if (value === null || value === undefined) {
        return <span className="text-gray-500">-</span>;
      }
      const formattedValue = formatIndicatorValue(value, row.indicator_type);
      const colorClass = value >= 0 ? "text-green-400" : "text-red-400";
      return (
        <span className={`font-mono text-sm ${colorClass}`}>
          {formattedValue}
        </span>
      );
    },
    cellClassName: "text-right",
  },
  {
    key: "upper_band",
    header: "上限",
    width: "100px",
    sortable: true,
    formatter: (value: number | null, row: TechnicalIndicatorData) => {
      if (value === null || value === undefined) {
        return <span className="text-gray-500">-</span>;
      }
      const formattedValue = formatIndicatorValue(value, row.indicator_type);
      return (
        <span className="font-mono text-sm text-red-400">
          {formattedValue}
        </span>
      );
    },
    cellClassName: "text-right",
  },
  {
    key: "lower_band",
    header: "下限",
    width: "100px",
    sortable: true,
    formatter: (value: number | null, row: TechnicalIndicatorData) => {
      if (value === null || value === undefined) {
        return <span className="text-gray-500">-</span>;
      }
      const formattedValue = formatIndicatorValue(value, row.indicator_type);
      return (
        <span className="font-mono text-sm text-blue-400">
          {formattedValue}
        </span>
      );
    },
    cellClassName: "text-right",
  },
  {
    key: "timestamp",
    header: "データ時刻",
    width: "180px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-mono text-sm text-gray-400">
        {formatDateTime(value)}
      </span>
    ),
  },
];
