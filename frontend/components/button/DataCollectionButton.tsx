/**
 * 汎用データ収集ボタンコンポーネント
 *
 * 設定オブジェクトベースで動作する汎用的なデータ収集ボタンです。
 * API エンドポイント、確認メッセージ、ボタンテキストなどを外部から注入可能で、
 * usePostRequest と useDataCollection の両方に対応します。
 */

"use client";

import React from "react";
import { usePostRequest } from "@/hooks/usePostRequest";
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
  const { sendPostRequest, isLoading: postLoading } = usePostRequest();
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
        // usePostRequest を使用する場合
        await handlePostRequest();
      } else {
        throw new Error("設定が不正です: endpoint または dataCollectionMethod が必要です");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "データ収集に失敗しました";
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
        if (typeof value === 'string' || typeof value === 'number') {
          params.append(key, String(value));
        }
      });
      
      if (params.toString()) {
        url += `?${params.toString()}`;
      }
    }

    const { success, data, error } = await sendPostRequest(url);
    
    if (success) {
      onCollectionStart?.(data);
    } else {
      throw new Error(error || "データ収集に失敗しました");
    }
  };

  const handleDataCollectionMethod = async () => {
    if (!config.dataCollectionMethod) return;

    // dataCollectionMethod の形式: "ohlcv.collect" や "fundingRate.collect"
    const [category, method] = config.dataCollectionMethod.split('.');
    
    if (!category || !method) {
      throw new Error(`不正なメソッド形式: ${config.dataCollectionMethod}`);
    }

    const categoryObject = (dataCollection as any)[category];
    if (!categoryObject || typeof categoryObject[method] !== 'function') {
      throw new Error(`メソッドが見つかりません: ${config.dataCollectionMethod}`);
    }

    await categoryObject[method](onCollectionStart, onCollectionError);
  };

  const getLoadingState = () => {
    if (config.useDataCollection && config.dataCollectionMethod) {
      const [category] = config.dataCollectionMethod.split('.');
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
