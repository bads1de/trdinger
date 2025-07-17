/**
 * 外部市場データ収集ボタンコンポーネント
 *
 * 外部市場データ（SP500、NASDAQ、DXY、VIX）の収集を実行するボタンです。
 */

import React, { useState } from "react";
import { useExternalMarketData, ExternalMarketCollectionResult } from "@/hooks/useExternalMarketData";

interface ExternalMarketCollectionButtonProps {
  onCollectionStart: (result: ExternalMarketCollectionResult) => void;
  onCollectionError: (errorMessage: string) => void;
  disabled?: boolean;
}

/**
 * 外部市場データ収集ボタンコンポーネント
 */
const ExternalMarketCollectionButton: React.FC<ExternalMarketCollectionButtonProps> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
}) => {
  const [isCollecting, setIsCollecting] = useState(false);
  const { collectIncrementalData } = useExternalMarketData();

  /**
   * 外部市場データ収集を実行
   */
  const handleCollectData = async () => {
    if (isCollecting || disabled) return;

    setIsCollecting(true);
    try {
      // 差分収集を実行（全シンボル）
      const result = await collectIncrementalData();
      onCollectionStart(result);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "外部市場データ収集に失敗しました";
      onCollectionError(errorMessage);
    } finally {
      setIsCollecting(false);
    }
  };

  return (
    <button
      onClick={handleCollectData}
      disabled={disabled || isCollecting}
      className={`
        inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md
        transition-all duration-200 ease-in-out transform hover:scale-105 active:scale-95
        ${
          disabled || isCollecting
            ? "bg-gray-300 text-gray-500 cursor-not-allowed"
            : "bg-purple-600 hover:bg-purple-700 text-white shadow-lg hover:shadow-xl"
        }
      `}
      title="外部市場データ（SP500、NASDAQ、DXY、VIX）を収集します"
    >
      {isCollecting ? (
        <>
          <svg
            className="animate-spin -ml-1 mr-3 h-4 w-4 text-white"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
          収集中...
        </>
      ) : (
        <>
          <svg
            className="w-4 h-4 mr-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 10V3L4 14h7v7l9-11h-7z"
            />
          </svg>
          外部市場データ収集
        </>
      )}
    </button>
  );
};

export default ExternalMarketCollectionButton;
