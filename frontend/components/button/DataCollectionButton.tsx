/**
 * 汎用データ収集ボタンコンポーネントと設定の統合ファイル
 *
 * 設定オブジェクトベースで動作する汎用的なデータ収集ボタンです。
 * API エンドポイント、確認メッセージ、ボタンテキストなどを外部から注入可能で、
 * useApiCall と useDataCollection の両方に対応します。
 *
 * データ収集ボタンの設定を統一的に管理します。各データ収集ボタンの設定をここで定義します。
 */

"use client";

import React from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { useDataCollection } from "@/hooks/useDataCollection";
import ApiButton from "./ApiButton";
import { ButtonVariant } from "./ApiButton";

/**
 * データ収集の設定
 */
export interface DataCollectionConfig {
  /** API エンドポイント */
  endpoint?: string;
  /** ボタンテキスト */
  buttonText: string;
  /** ボタンのバリアント */
  variant: ButtonVariant;
  /** 確認メッセージ（オプション） */
  confirmMessage?: string;
  /** 成功時のメッセージ（オプション） */
  successMessage?: string;
  /** ローディング中のテキスト */
  loadingText?: string;
  /** useDataCollection を使用するかどうか */
  useDataCollection?: boolean;
  /** useDataCollection 使用時のメソッド名 */
  dataCollectionMethod?: string;
  /** 追加のクエリパラメータ */
  queryParams?: Record<string, string>;
}

/**
 * 全データ一括収集ボタンの設定
 */
export const allDataCollectionConfig: DataCollectionConfig = {
  endpoint: "/api/data-collection/all/bulk-collect",
  buttonText: "全データ取得",
  variant: "primary",
  confirmMessage:
    "全データ（OHLCV・FR・OI）を一括取得します。\n\n" +
    "この処理には数分から十数分かかる場合があります。\n" +
    "テクニカル指標も自動計算されます。続行しますか？",
  successMessage: "全データの収集を開始しました",
  loadingText: "収集中...",
};

/**
 * OHLCV収集ボタンの設定
 */
export const ohlcvCollectionConfig: DataCollectionConfig = {
  buttonText: "OHLCV収集",
  variant: "primary",
  useDataCollection: true,
  dataCollectionMethod: "ohlcv.collect",
  successMessage: "OHLCVデータの収集を開始しました",
  loadingText: "収集中...",
};

/**
 * Funding Rate収集ボタンの設定
 */
export const fundingRateCollectionConfig: DataCollectionConfig = {
  buttonText: "FR収集",
  variant: "success",
  useDataCollection: true,
  dataCollectionMethod: "fundingRate.collect",
  successMessage: "Funding Rateデータの収集を開始しました",
  loadingText: "収集中...",
};

/**
 * Open Interest収集ボタンの設定
 */
export const openInterestCollectionConfig: DataCollectionConfig = {
  buttonText: "OI収集",
  variant: "warning",
  endpoint: "/api/open-interest/bulk-collect",
  confirmMessage:
    "BTCの全期間OIデータを取得します。\n\n" +
    "この処理には数分かかる場合があります。続行しますか？",
  successMessage: "Open Interestデータの収集を開始しました",
  loadingText: "収集中...",
};

/**
 * 単一シンボル Open Interest収集ボタンの設定を生成
 */
export const createSingleOpenInterestConfig = (
  symbol: string
): DataCollectionConfig => ({
  endpoint: "/api/open-interest/collect",
  buttonText: `OI収集 (${symbol})`,
  variant: "warning",
  queryParams: {
    symbol: symbol,
    fetch_all: "true",
  },
  successMessage: `Open Interestデータ (${symbol}) の収集を開始しました`,
  loadingText: "収集中...",
});

/**
 * データ収集ボタンのプロパティ
 */
interface DataCollectionButtonProps {
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
  /** 追加のプロパティ（モードやシンボルなど） */
  additionalProps?: Record<string, any>;
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
  additionalProps = {},
}) => {
  const { execute, loading: postLoading } = useApiCall();
  const dataCollection = useDataCollection();

  const handleClick = async () => {
    try {
      if (config.confirmMessage) {
        if (!window.confirm(config.confirmMessage)) {
          return;
        }
      }

      if (config.useDataCollection && config.dataCollectionMethod) {
        // useDataCollection を使用する場合
        await handleDataCollectionMethod();
      } else if (config.endpoint) {
        // useApiCall を使用する場合
        await handlePostRequest();
      } else {
        throw new Error(
          "設定が不正です: endpoint または dataCollectionMethod が必要です"
        );
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "データ収集に失敗しました";
      onCollectionError?.(errorMessage);
    }
  };

  const handlePostRequest = async () => {
    if (!config.endpoint) return;

    let url = config.endpoint;

    // クエリパラメータの追加
    if (config.queryParams || additionalProps) {
      const params = new URLSearchParams();

      // 設定からのクエリパラメータ
      if (config.queryParams) {
        Object.entries(config.queryParams).forEach(([key, value]) => {
          params.append(key, value);
        });
      }

      // 追加プロパティからのクエリパラメータ
      Object.entries(additionalProps).forEach(([key, value]) => {
        if (typeof value === "string" || typeof value === "number") {
          params.append(key, String(value));
        }
      });

      if (params.toString()) {
        url += `?${params.toString()}`;
      }
    }

    await execute(url, {
      method: "POST",
      successMessage: config.successMessage,
      onSuccess: (data) => {
        onCollectionStart?.(data);
      },
      onError: (error) => {
        throw new Error(error || "データ収集に失敗しました");
      },
    });
  };

  const handleDataCollectionMethod = async () => {
    if (!config.dataCollectionMethod) return;

    // dataCollectionMethod の形式: "ohlcv.collect" や "fundingRate.collect"
    const [category, method] = config.dataCollectionMethod.split(".");

    if (!category || !method) {
      throw new Error(`不正なメソッド形式: ${config.dataCollectionMethod}`);
    }

    const categoryObject = (dataCollection as any)[category];
    if (!categoryObject || typeof categoryObject[method] !== "function") {
      throw new Error(
        `メソッドが見つかりません: ${config.dataCollectionMethod}`
      );
    }

    await categoryObject[method](
      onCollectionStart,
      onCollectionError,
      config.successMessage
    );
  };

  const getLoadingState = () => {
    if (config.useDataCollection && config.dataCollectionMethod) {
      const [category] = config.dataCollectionMethod.split(".");
      const categoryObject = (dataCollection as any)[category];
      return categoryObject?.loading || dataCollection.isAnyLoading;
    }
    return postLoading;
  };

  const getDisabledState = () => {
    if (config.useDataCollection) {
      return disabled || dataCollection.isAnyLoading;
    }
    return disabled || postLoading;
  };

  return (
    <ApiButton
      onClick={handleClick}
      loading={getLoadingState()}
      disabled={getDisabledState()}
      variant={config.variant}
      size="sm"
      loadingText={config.loadingText || "収集中..."}
      className={className}
    >
      {config.buttonText}
    </ApiButton>
  );
};

export default DataCollectionButton;