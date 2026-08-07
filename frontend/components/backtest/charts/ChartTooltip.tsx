/**
 * チャート用カスタムツールチップ共通コンポーネント
 *
 * Rechartsのカスタムツールチップに共通のコンテナ・ガード・行レイアウトを提供する
 */

"use client";

import React from "react";

/**
 * Rechartsがツールチップのcontentプロパティへ渡すprops
 */
export interface ChartTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string | number;
}

/**
 * ツールチップ行のラベル・値カラー
 */
export type TooltipTextColor = "default" | "red" | "green" | "yellow";

/**
 * 共通のカスタムツールチップコンテナ
 *
 * active/payloadのガードと外枠スタイルを提供し、
 * ヘッダーと本体の表示は呼び出し側がchildrenで定義する
 */
export const ChartTooltipContainer: React.FC<
  ChartTooltipProps & { title?: string; children?: React.ReactNode }
> = ({ active, payload, title, children }) => {
  if (!active || !payload || !payload.length) {
    return null;
  }

  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
      {title && <p className="text-white font-semibold mb-2">{title}</p>}
      {children}
    </div>
  );
};

const textColorClass = (color: TooltipTextColor): string => {
  switch (color) {
    case "red":
      return "text-red-400";
    case "green":
      return "text-green-400";
    case "yellow":
      return "text-yellow-400";
    default:
      return "";
  }
};

/**
 * ツールチップ内のラベル・値の1行を表示する
 */
export const TooltipRow: React.FC<{
  label: string;
  value: React.ReactNode;
  labelColor?: TooltipTextColor;
  valueColor?: TooltipTextColor;
  labelClassName?: string;
  valueClassName?: string;
  labelStyle?: React.CSSProperties;
  className?: string;
}> = ({
  label,
  value,
  labelColor = "default",
  valueColor = "default",
  labelClassName,
  valueClassName,
  labelStyle,
  className = "flex justify-between",
}) => {
  return (
    <div className={className}>
      <span
        className={`${labelClassName ?? (labelColor === "default" ? "text-gray-400 mr-3" : "mr-3")} ${textColorClass(labelColor)}`}
        style={labelStyle}
      >
        {label}:
      </span>
      <span
        className={`${valueClassName ?? "text-white font-medium"} ${textColorClass(valueColor)}`}
      >
        {value}
      </span>
    </div>
  );
};
