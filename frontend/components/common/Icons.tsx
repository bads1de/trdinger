/**
 * 共通アイコンコンポーネント
 *
 * アプリケーション全体で使用されるSVGアイコンを統一的に管理します。
 * 一貫性のあるアイコンサイズとスタイルを提供します。
 *
 */

import React from "react";

/**
 * アイコンの基本プロパティ
 */
interface IconProps {
  size?: "xs" | "sm" | "md" | "lg" | "xl";
  className?: string;
}

/**
 * サイズマッピング
 */
const sizeMap = {
  xs: "w-3 h-3",
  sm: "w-4 h-4",
  md: "w-5 h-5",
  lg: "w-6 h-6",
  xl: "w-8 h-8",
};

/**
 * 検索アイコン
 */
export const SearchIcon: React.FC<IconProps> = ({
  size = "sm",
  className = "",
}) => (
  <svg
    className={`${sizeMap[size]} ${className}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
    />
  </svg>
);

/**
 * エクスポートアイコン
 */
export const ExportIcon: React.FC<IconProps> = ({
  size = "sm",
  className = "",
}) => (
  <svg
    className={`${sizeMap[size]} ${className}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
    />
  </svg>
);

/**
 * エラーアイコン
 */
export const ErrorIcon: React.FC<IconProps> = ({
  size = "sm",
  className = "",
}) => (
  <svg
    className={`${sizeMap[size]} ${className}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
);

/**
 * ソートアイコン（昇順）
 */
export const SortAscIcon: React.FC<IconProps> = ({
  size = "sm",
  className = "",
}) => (
  <svg
    className={`${sizeMap[size]} ${className}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M5 15l7-7 7 7"
    />
  </svg>
);

/**
 * ソートアイコン（降順）
 */
export const SortDescIcon: React.FC<IconProps> = ({
  size = "sm",
  className = "",
}) => (
  <svg
    className={`${sizeMap[size]} ${className}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M19 9l-7 7-7-7"
    />
  </svg>
);

/**
 * ソートアイコン（ニュートラル）
 */
export const SortNeutralIcon: React.FC<IconProps> = ({
  size = "sm",
  className = "",
}) => (
  <svg
    className={`${sizeMap[size]} ${className}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"
    />
  </svg>
);
