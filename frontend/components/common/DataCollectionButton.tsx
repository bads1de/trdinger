/**
 * 汎用データ収集ボタンコンポーネント
 *
 * 各種データ収集機能を統合した汎用ボタンコンポーネントです。
 * 設定オブジェクトにより動作をカスタマイズできます。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React, { useState } from "react";

/**
 * ボタンの状態を表す列挙型
 */
export type ButtonState = "idle" | "loading" | "success" | "error";

/**
 * データ収集の設定
 */
export interface DataCollectionConfig {
  /** API エンドポイント */
  apiEndpoint: string;
  /** HTTP メソッド */
  method?: "GET" | "POST" | "PUT" | "PATCH";
  /** 確認ダイアログのメッセージ */
  confirmMessage?: string;
  /** ボタンのテキスト（状態別） */
  buttonText: {
    idle: string;
    loading: string;
    success: string;
    error: string;
  };
  /** ボタンのアイコン（状態別） */
  buttonIcon?: {
    idle: React.ReactNode;
    loading?: React.ReactNode;
    success?: React.ReactNode;
    error?: React.ReactNode;
  };
  /** 説明テキスト */
  description?: string;
  /** 成功時のリセット時間（ミリ秒） */
  successResetTime?: number;
  /** エラー時のリセット時間（ミリ秒） */
  errorResetTime?: number;
}

/**
 * データ収集ボタンコンポーネントのプロパティ
 */
export interface DataCollectionButtonProps {
  /** データ収集の設定 */
  config: DataCollectionConfig;
  /** データ収集開始時のコールバック */
  onCollectionStart?: (result: any) => void;
  /** データ収集エラー時のコールバック */
  onCollectionError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * 汎用データ収集ボタンコンポーネント
 */
const DataCollectionButton: React.FC<DataCollectionButtonProps> = ({
  config,
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  const [buttonState, setButtonState] = useState<ButtonState>("idle");
  const [lastResult, setLastResult] = useState<any>(null);

  /**
   * データ収集を実行
   */
  const handleCollectData = async () => {
    // 確認ダイアログ
    if (config.confirmMessage) {
      const confirmed = window.confirm(config.confirmMessage);
      if (!confirmed) {
        return;
      }
    }

    try {
      setButtonState("loading");
      setLastResult(null);

      const response = await fetch(config.apiEndpoint, {
        method: config.method || "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const result = await response.json();

      if (result.success) {
        setButtonState("success");
        setLastResult(result.data || result);
        onCollectionStart?.(result.data || result);

        // 指定時間後にアイドル状態に戻す
        setTimeout(() => {
          setButtonState("idle");
        }, config.successResetTime || 3000);
      } else {
        throw new Error(result.message || "データ収集に失敗しました");
      }
    } catch (error) {
      console.error("データ収集エラー:", error);
      setButtonState("error");
      const errorMessage =
        error instanceof Error
          ? error.message
          : "データ収集中にエラーが発生しました";
      onCollectionError?.(errorMessage);

      // 指定時間後にアイドル状態に戻す
      setTimeout(() => {
        setButtonState("idle");
      }, config.errorResetTime || 5000);
    }
  };

  /**
   * ボタンが無効かどうかを判定
   */
  const isButtonDisabled = disabled || buttonState === "loading";

  /**
   * ボタンのアイコンを取得
   */
  const getButtonIcon = () => {
    const icons = config.buttonIcon;

    switch (buttonState) {
      case "loading":
        return (
          icons?.loading || (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
          )
        );
      case "success":
        return icons?.success || <span className="text-green-400">✅</span>;
      case "error":
        return icons?.error || <span className="text-red-400">❌</span>;
      default:
        return icons?.idle || null;
    }
  };

  /**
   * ボタンのテキストを取得
   */
  const getButtonText = () => {
    return config.buttonText[buttonState];
  };

  /**
   * ボタンのスタイルクラスを取得
   */
  const getButtonClasses = () => {
    // classNameにh-10が含まれているかチェック
    const hasHeightClass =
      className.includes("h-10") || className.includes("h-12");
    const heightClass = hasHeightClass ? "" : "h-12";
    const baseClasses = `flex items-center justify-center gap-2 px-4 text-sm font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 min-w-[120px] ${heightClass}`;

    if (isButtonDisabled) {
      return `${baseClasses} bg-gray-600 text-gray-400 cursor-not-allowed`;
    }

    switch (buttonState) {
      case "loading":
        return `${baseClasses} bg-blue-600 text-white cursor-not-allowed opacity-75`;
      case "success":
        return `${baseClasses} bg-green-600 text-white`;
      case "error":
        return `${baseClasses} bg-red-600 text-white`;
      default:
        return `${baseClasses} bg-primary-600 hover:bg-primary-700 text-white focus:ring-primary-500`;
    }
  };

  return (
    <button
      onClick={handleCollectData}
      disabled={isButtonDisabled}
      className={getButtonClasses()}
      title={config.description}
    >
      {getButtonIcon()}
      <span>{getButtonText()}</span>
    </button>
  );
};

export default DataCollectionButton;
