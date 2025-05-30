/**
 * API呼び出し用汎用ボタンコンポーネント
 *
 * ローディング状態、エラーハンドリング、アイコン表示などを統一的に管理する
 * 汎用的なAPIボタンコンポーネントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";
import { BUTTON_STYLES, combineStyles } from "@/constants/styles";

/**
 * ボタンのサイズ
 */
export type ButtonSize = "xs" | "sm" | "md" | "lg" | "xl";

/**
 * ボタンのバリアント
 */
export type ButtonVariant = "primary" | "secondary" | "success" | "warning" | "error" | "outline" | "ghost";

/**
 * APIボタンのプロパティ
 */
export interface ApiButtonProps {
  /** ボタンのテキスト */
  children: React.ReactNode;
  /** クリック時のハンドラー */
  onClick: () => void | Promise<void>;
  /** ローディング状態 */
  loading?: boolean;
  /** 無効状態 */
  disabled?: boolean;
  /** ボタンのサイズ */
  size?: ButtonSize;
  /** ボタンのバリアント */
  variant?: ButtonVariant;
  /** アイコン（SVGパス） */
  icon?: React.ReactNode;
  /** ローディング時のテキスト */
  loadingText?: string;
  /** 追加のクラス名 */
  className?: string;
  /** 最小幅を固定するか */
  fixedWidth?: boolean;
}

/**
 * デフォルトのローディングアイコン
 */
const LoadingIcon: React.FC<{ className?: string }> = ({ className = "w-4 h-4" }) => (
  <div className={`animate-spin rounded-full border-b-2 border-current ${className}`} />
);

/**
 * API呼び出し用汎用ボタンコンポーネント
 */
const ApiButton: React.FC<ApiButtonProps> = ({
  children,
  onClick,
  loading = false,
  disabled = false,
  size = "md",
  variant = "primary",
  icon,
  loadingText,
  className = "",
  fixedWidth = true,
}) => {
  const isDisabled = disabled || loading;
  
  const buttonClasses = combineStyles(
    BUTTON_STYLES.base,
    BUTTON_STYLES.sizes[size],
    loading ? BUTTON_STYLES.states.loading : BUTTON_STYLES.variants[variant],
    isDisabled && "opacity-50 cursor-not-allowed",
    !fixedWidth && "min-w-0",
    className
  );

  const handleClick = async () => {
    if (isDisabled) return;
    await onClick();
  };

  return (
    <button
      onClick={handleClick}
      disabled={isDisabled}
      className={buttonClasses}
    >
      {loading ? (
        <>
          <LoadingIcon />
          <span>{loadingText || "処理中..."}</span>
        </>
      ) : (
        <>
          {icon && <span className="flex-shrink-0">{icon}</span>}
          <span>{children}</span>
        </>
      )}
    </button>
  );
};

export default ApiButton;
