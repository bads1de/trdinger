/**
 * 折りたたみ可能なJSON表示コンポーネント
 *
 * 大きなJSONデータを初期状態では折りたたんで表示し、
 * ユーザーが必要に応じて展開できるコンポーネントです。
 */

"use client";

import React, { useState } from "react";
import { ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/24/outline";

interface CollapsibleJsonProps {
  /** 表示するJSONデータ */
  data: any;
  /** タイトル */
  title: string;
  /** 初期状態で展開するかどうか（デフォルト: false） */
  defaultExpanded?: boolean;
  /** 最大高さ（展開時、デフォルト: 400px） */
  maxHeight?: string;
  /** カスタムクラス名 */
  className?: string;
  /** テーマ（デフォルト: 'matrix'） */
  theme?: "matrix" | "dark" | "light";
}

const CollapsibleJson: React.FC<CollapsibleJsonProps> = ({
  data,
  title,
  defaultExpanded = false,
  maxHeight = "400px",
  className = "",
  theme = "matrix",
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  // JSONサイズを計算
  const jsonString = JSON.stringify(data, null, 2);
  const jsonSize = jsonString.length;
  const isLarge = jsonSize > 1000;

  // テーマに応じたスタイル
  const getThemeStyles = () => {
    switch (theme) {
      case "matrix":
        return {
          container: "bg-black/80 border-green-500/30",
          header: "text-green-400",
          indicator: "bg-green-400",
          content: "text-green-300 bg-gray-900/50 border-green-500/20",
          overlay: "bg-green-400/5",
          button: "hover:bg-green-500/10 text-green-400",
        };
      case "dark":
        return {
          container: "bg-gray-900 border-gray-600",
          header: "text-gray-200",
          indicator: "bg-blue-400",
          content: "text-gray-300 bg-gray-800 border-gray-600",
          overlay: "bg-blue-400/5",
          button: "hover:bg-gray-700 text-gray-300",
        };
      case "light":
        return {
          container: "bg-white border-gray-300",
          header: "text-gray-800",
          indicator: "bg-blue-500",
          content: "text-gray-700 bg-gray-50 border-gray-300",
          overlay: "bg-blue-500/5",
          button: "hover:bg-gray-100 text-gray-700",
        };
      default:
        return getThemeStyles(); // デフォルトはmatrix
    }
  };

  const styles = getThemeStyles();

  return (
    <div
      className={`p-4 rounded-lg border shadow-lg ${styles.container} ${className}`}
    >
      {/* ヘッダー */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center">
          <div
            className={`w-2 h-2 rounded-full mr-2 ${styles.indicator} ${
              theme === "matrix" ? "animate-pulse" : ""
            }`}
          ></div>
          <h4
            className={`font-mono text-sm font-semibold tracking-wide ${styles.header}`}
          >
            {title}
          </h4>
          {isLarge && (
            <span className={`ml-2 text-xs px-2 py-1 rounded ${styles.button}`}>
              {(jsonSize / 1024).toFixed(1)}KB
            </span>
          )}
        </div>

        {/* 展開/折りたたみボタン */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className={`flex items-center px-3 py-1 rounded transition-colors ${styles.button}`}
          aria-label={isExpanded ? "折りたたむ" : "展開する"}
        >
          {isExpanded ? (
            <>
              <ChevronDownIcon className="w-4 h-4 mr-1" />
              <span className="text-xs">折りたたむ</span>
            </>
          ) : (
            <>
              <ChevronRightIcon className="w-4 h-4 mr-1" />
              <span className="text-xs">展開する</span>
            </>
          )}
        </button>
      </div>

      {/* プレビュー（折りたたみ時） */}
      {!isExpanded && (
        <div className="relative">
          <div className={`p-3 rounded border ${styles.content}`}>
            <div className="flex items-center justify-between">
              <span className="text-xs opacity-70">
                {Object.keys(data || {}).length} 個のプロパティ
              </span>
            </div>
            {/* 主要なプロパティのプレビュー */}
            {data && typeof data === "object" && (
              <div className="mt-2 text-xs opacity-60">
                {Object.keys(data)
                  .slice(0, 3)
                  .map((key, index) => (
                    <span key={key}>
                      {key}
                      {index < Math.min(Object.keys(data).length, 3) - 1 &&
                        ", "}
                    </span>
                  ))}
                {Object.keys(data).length > 3 && "..."}
              </div>
            )}
          </div>
        </div>
      )}

      {/* 展開されたJSON */}
      {isExpanded && (
        <div className="relative">
          <div
            className={`overflow-auto ${styles.content} p-3 rounded border shadow-inner`}
            style={{ maxHeight }}
          >
            <pre className="font-mono text-xs leading-relaxed whitespace-pre-wrap">
              <code className={styles.content.split(" ")[0]}>{jsonString}</code>
            </pre>
          </div>
          {/* テーマに応じたオーバーレイ効果 */}
          <div
            className={`absolute inset-0 rounded pointer-events-none ${styles.overlay}`}
          ></div>
        </div>
      )}

      {/* フッター情報 */}
      {isExpanded && isLarge && (
        <div className="mt-2 flex justify-between items-center text-xs opacity-60">
          <span>{jsonString.split("\n").length} 行</span>
          <span>{jsonSize.toLocaleString()} 文字</span>
        </div>
      )}
    </div>
  );
};

export default CollapsibleJson;
