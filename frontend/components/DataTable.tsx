/**
 * 共通データテーブルコンポーネント
 *
 * ソート、ページネーション、CSVエクスポート機能を持つ汎用テーブルコンポーネントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React, { useState, useMemo } from "react";

/**
 * テーブルカラムの定義
 */
export interface TableColumn<T> {
  /** カラムのキー */
  key: keyof T;
  /** カラムのヘッダー表示名 */
  header: string;
  /** カラムの幅（CSS値） */
  width?: string;
  /** ソート可能かどうか */
  sortable?: boolean;
  /** セルの値をフォーマットする関数 */
  formatter?: (value: any, row: T) => React.ReactNode;
  /** セルのクラス名 */
  cellClassName?: string;
}

/**
 * データテーブルのプロパティ
 */
export interface DataTableProps<T> {
  /** テーブルデータ */
  data: T[];
  /** カラム定義 */
  columns: TableColumn<T>[];
  /** テーブルのタイトル */
  title?: string;
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** 1ページあたりの表示件数 */
  pageSize?: number;
  /** CSVエクスポート機能を有効にするか */
  enableExport?: boolean;
  /** 検索機能を有効にするか */
  enableSearch?: boolean;
  /** 検索対象のキー */
  searchKeys?: (keyof T)[];
  /** テーブルのクラス名 */
  className?: string;
}

/**
 * ソート方向
 */
type SortDirection = "asc" | "desc" | null;

/**
 * 共通データテーブルコンポーネント
 */
const DataTable = <T extends Record<string, any>>({
  data,
  columns,
  title,
  loading = false,
  error,
  pageSize = 50,
  enableExport = true,
  enableSearch = true,
  searchKeys = [],
  className = "",
}: DataTableProps<T>) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [sortKey, setSortKey] = useState<keyof T | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(null);
  const [searchTerm, setSearchTerm] = useState("");

  // データのフィルタリング（検索）
  const filteredData = useMemo(() => {
    if (!searchTerm || !enableSearch || searchKeys.length === 0) {
      return data;
    }

    return data.filter((row) =>
      searchKeys.some((key) => {
        const value = row[key];
        if (value == null) return false;
        return String(value).toLowerCase().includes(searchTerm.toLowerCase());
      })
    );
  }, [data, searchTerm, enableSearch, searchKeys]);

  // データのソート
  const sortedData = useMemo(() => {
    if (!sortKey || !sortDirection) {
      return filteredData;
    }

    return [...filteredData].sort((a, b) => {
      const aValue = a[sortKey];
      const bValue = b[sortKey];

      if (aValue == null && bValue == null) return 0;
      if (aValue == null) return sortDirection === "asc" ? -1 : 1;
      if (bValue == null) return sortDirection === "asc" ? 1 : -1;

      if (typeof aValue === "number" && typeof bValue === "number") {
        return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
      }

      const aStr = String(aValue);
      const bStr = String(bValue);
      const comparison = aStr.localeCompare(bStr);
      return sortDirection === "asc" ? comparison : -comparison;
    });
  }, [filteredData, sortKey, sortDirection]);

  // ページネーション
  const totalPages = Math.ceil(sortedData.length / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = startIndex + pageSize;
  const paginatedData = sortedData.slice(startIndex, endIndex);

  // ソートハンドラ
  const handleSort = (key: keyof T) => {
    const column = columns.find((col) => col.key === key);
    if (!column?.sortable) return;

    if (sortKey === key) {
      if (sortDirection === "asc") {
        setSortDirection("desc");
      } else if (sortDirection === "desc") {
        setSortKey(null);
        setSortDirection(null);
      } else {
        setSortDirection("asc");
      }
    } else {
      setSortKey(key);
      setSortDirection("asc");
    }
    setCurrentPage(1);
  };

  // CSVエクスポート
  const handleExportCSV = () => {
    const headers = columns.map((col) => col.header);
    const csvContent = [
      headers.join(","),
      ...sortedData.map((row) =>
        columns
          .map((col) => {
            const value = row[col.key];
            const formattedValue = col.formatter
              ? String(col.formatter(value, row)).replace(/,/g, ";")
              : String(value || "").replace(/,/g, ";");
            return `"${formattedValue}"`;
          })
          .join(",")
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `${title || "data"}_${new Date().toISOString().split("T")[0]}.csv`);
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // ソートアイコン
  const getSortIcon = (key: keyof T) => {
    if (sortKey !== key) {
      return (
        <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
        </svg>
      );
    }

    return sortDirection === "asc" ? (
      <svg className="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
      </svg>
    ) : (
      <svg className="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    );
  };

  if (error) {
    return (
      <div className="bg-gray-900 dark:bg-gray-900 rounded-enterprise-lg border border-gray-700 dark:border-gray-700 p-8">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-gray-800 dark:bg-gray-800 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-error-600 dark:text-error-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-error-800 dark:text-error-200 mb-2">
            📊 データの読み込みに失敗しました
          </h3>
          <p className="text-sm text-error-600 dark:text-error-400">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-900 dark:bg-gray-900 rounded-enterprise-lg border border-gray-700 dark:border-gray-700 ${className}`}>
      {/* ヘッダー */}
      <div className="p-6 border-b border-gray-700 dark:border-gray-700">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            {title && (
              <h3 className="text-lg font-semibold text-gray-100 dark:text-gray-100">
                {title}
              </h3>
            )}
            <p className="text-sm text-gray-400 dark:text-gray-400 mt-1">
              {loading ? "データを読み込み中..." : `${sortedData.length}件のデータを表示中`}
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* 検索 */}
            {enableSearch && searchKeys.length > 0 && (
              <div className="relative">
                <input
                  type="text"
                  placeholder="検索..."
                  value={searchTerm}
                  onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setCurrentPage(1);
                  }}
                  className="w-48 px-3 py-2 bg-gray-800 dark:bg-gray-800 border border-gray-600 dark:border-gray-600 rounded-lg text-gray-100 dark:text-gray-100 text-sm focus:outline-none focus:ring-2 focus:ring-primary-600 focus:border-transparent"
                />
                <svg className="absolute right-3 top-2.5 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
            )}

            {/* CSVエクスポート */}
            {enableExport && (
              <button
                onClick={handleExportCSV}
                disabled={loading || sortedData.length === 0}
                className="px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors duration-200 flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                CSV出力
              </button>
            )}
          </div>
        </div>
      </div>

      {/* テーブル */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-800 dark:bg-gray-800">
            <tr>
              {columns.map((column) => (
                <th
                  key={String(column.key)}
                  className={`px-6 py-3 text-left text-xs font-medium text-gray-300 dark:text-gray-300 uppercase tracking-wider ${
                    column.sortable ? "cursor-pointer hover:bg-gray-700 dark:hover:bg-gray-700" : ""
                  }`}
                  style={{ width: column.width }}
                  onClick={() => column.sortable && handleSort(column.key)}
                >
                  <div className="flex items-center gap-2">
                    {column.header}
                    {column.sortable && getSortIcon(column.key)}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700 dark:divide-gray-700">
            {loading ? (
              <tr>
                <td colSpan={columns.length} className="px-6 py-12 text-center">
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
                    <span className="ml-3 text-gray-400 dark:text-gray-400">データを読み込み中...</span>
                  </div>
                </td>
              </tr>
            ) : paginatedData.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-6 py-12 text-center text-gray-400 dark:text-gray-400">
                  データがありません
                </td>
              </tr>
            ) : (
              paginatedData.map((row, index) => (
                <tr key={index} className="hover:bg-gray-800 dark:hover:bg-gray-800">
                  {columns.map((column) => (
                    <td
                      key={String(column.key)}
                      className={`px-6 py-4 whitespace-nowrap text-sm text-gray-100 dark:text-gray-100 ${column.cellClassName || ""}`}
                    >
                      {column.formatter
                        ? column.formatter(row[column.key], row)
                        : String(row[column.key] || "")}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* ページネーション */}
      {totalPages > 1 && (
        <div className="px-6 py-4 border-t border-gray-700 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-400 dark:text-gray-400">
              {startIndex + 1} - {Math.min(endIndex, sortedData.length)} / {sortedData.length}件
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 bg-gray-800 hover:bg-gray-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-gray-100 text-sm rounded border border-gray-600"
              >
                前へ
              </button>
              <span className="px-3 py-1 text-sm text-gray-100">
                {currentPage} / {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 bg-gray-800 hover:bg-gray-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-gray-100 text-sm rounded border border-gray-600"
              >
                次へ
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataTable;
