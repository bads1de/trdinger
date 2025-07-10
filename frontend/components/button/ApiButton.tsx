/**
 * API呼び出し用汎用ボタンコンポーネント
 *
 * ローディング状態、エラーハンドリングなどを統一的に管理する
 * 汎用的なAPIボタンコンポーネントです。
 * ActionButtonをラップし、API呼び出しに特化した機能を提供します。
 */

"use client";

import React from "react";
import ActionButton from "../common/ActionButton";

export type ButtonSize = "sm" | "md" | "lg";
export type ButtonVariant = "primary" | "secondary" | "success" | "warning" | "danger";

export interface ApiButtonProps {
  children: React.ReactNode;
  onClick: () => void | Promise<void>;
  loading?: boolean;
  disabled?: boolean;
  size?: ButtonSize;
  variant?: ButtonVariant;
  icon?: React.ReactNode;
  loadingText?: string;
  className?: string;
}

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
}) => {
  const handleClick = async () => {
    if (disabled || loading) return;
    await onClick();
  };

  return (
    <ActionButton
      onClick={handleClick}
      loading={loading}
      disabled={disabled}
      size={size}
      variant={variant}
      icon={icon}
      loadingText={loadingText}
      className={className}
    >
      {children}
    </ActionButton>
  );
};

export default ApiButton;
