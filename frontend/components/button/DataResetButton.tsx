/**
 * データリセットボタンコンポーネント
 *
 * OHLCV、ファンディングレート、オープンインタレストデータのリセット機能を提供します。
 *
 */

"use client";

import React, { useState } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import ApiButton from "./ApiButton";
import { RESET_CONFIGS } from "@/constants/data-reset-constants";
import ConfirmDialog from "@/components/common/ConfirmDialog";

/**
 * データリセットの種類
 */
export type ResetType =
  | "all"
  | "ohlcv"
  | "funding-rates"
  | "open-interest"
  | "symbol";

/**
 * データリセット結果の型
 */
export interface DataResetResult {
  success: boolean;
  deleted_counts?: {
    ohlcv: number;
    funding_rates: number;
    open_interest: number;
  };
  deleted_count?: number;
  total_deleted?: number;
  data_type?: string;
  symbol?: string;
  message: string;
  errors?: string[];
  timestamp: string;
}

/**
 * データリセットボタンコンポーネントのプロパティ
 */
interface DataResetButtonProps {
  /** リセットの種類 */
  resetType: ResetType;
  /** 特定シンボルリセット時のシンボル名 */
  symbol?: string;
  /** リセット完了時のコールバック */
  onResetComplete?: (result: DataResetResult) => void;
  /** リセットエラー時のコールバック */
  onResetError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
  /** ボタンサイズ */
  size?: "sm" | "md" | "lg";
  /** ボタンバリアント */
  variant?: "primary" | "secondary" | "success" | "warning" | "danger";
}

/**
 * データリセットボタンコンポーネント
 */
const DataResetButton: React.FC<DataResetButtonProps> = ({
  resetType,
  symbol,
  onResetComplete,
  onResetError,
  disabled = false,
  className = "",
  size = "sm",
  variant,
}) => {
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const { execute, loading: isLoading } = useApiCall<DataResetResult>();

  const config = RESET_CONFIGS[resetType];
  const buttonVariant = variant || config.variant;

  // エンドポイントと確認メッセージの構築
  let endpoint = config.endpoint;
  let confirmDescription = config.confirmMessage;

  if (resetType === "symbol" && symbol) {
    endpoint = `${endpoint}/${encodeURIComponent(symbol)}`;
    confirmDescription = confirmDescription.replace(
      "指定されたシンボル",
      `シンボル「${symbol}」`
    );
  }

  // 確認メッセージをカスタマイズ
  if (resetType === "symbol" && symbol) {
    confirmDescription =
      `シンボル「${symbol}」の全データ（OHLCV・ファンディングレート・オープンインタレスト）を削除します。\n\n` +
      "この操作は取り消すことができません。";
  }

  const handleResetClick = () => {
    if (isLoading) return;
    setIsConfirmOpen(true);
  };

  const handleConfirmReset = async () => {
    await execute(endpoint, {
      method: "DELETE",
      // confirmMessage: confirmDescription, // useApiCallの内部confirmは使用せず、独自ダイアログを使用
      successMessage:
        resetType === "symbol" && symbol
          ? `${symbol} のデータリセットが完了しました`
          : "データリセットが完了しました",
      onSuccess: (data) => {
        onResetComplete?.(data);
      },
      onError: (error) => {
        console.error("データリセットエラー:", error);
        onResetError?.(error || "データリセット中にエラーが発生しました");
      },
    });
  };

  // ボタンラベルを動的に生成
  const getButtonLabel = () => {
    if (resetType === "symbol" && symbol) {
      return `${symbol} リセット`;
    }
    return config.label;
  };

  return (
    <>
      <ApiButton
        onClick={handleResetClick}
        loading={isLoading}
        disabled={disabled || isLoading}
        variant={buttonVariant}
        size={size}
        loadingText="削除中..."
        className={className}
        icon={<span>{config.icon}</span>}
      >
        {getButtonLabel()}
      </ApiButton>

      <ConfirmDialog
        open={isConfirmOpen}
        onOpenChange={setIsConfirmOpen}
        title={`${getButtonLabel()}の確認`}
        description={confirmDescription}
        confirmText="削除を実行"
        cancelText="キャンセル"
        variant="destructive"
        onConfirm={handleConfirmReset}
      />
    </>
  );
};

export default DataResetButton;

export default DataResetButton;
