/**
 * Fear & Greed Index データ状態取得API
 *
 * Fear & Greed Indexデータの状態を取得するAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * データ状態の型定義
 */
export interface FearGreedDataStatus {
  /** 成功フラグ */
  success: boolean;
  /** データ範囲情報 */
  data_range: {
    /** 最古のデータ日時 */
    oldest_data: string | null;
    /** 最新のデータ日時 */
    newest_data: string | null;
    /** 総件数 */
    total_count: number;
  };
  /** 最新タイムスタンプ */
  latest_timestamp: string | null;
  /** 現在時刻 */
  current_time: string;
  /** エラー情報 */
  error?: string;
}

/**
 * データ状態レスポンスの型定義
 */
export interface FearGreedStatusResponse {
  /** 成功フラグ */
  success: boolean;
  /** メッセージ */
  message: string;
  /** データ */
  data: FearGreedDataStatus;
}

/**
 * バックエンドAPIからFear & Greed Indexデータの状態を取得する関数
 *
 * @returns データ状態
 */
async function fetchFearGreedDataStatus(): Promise<FearGreedDataStatus> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/fear-greed/status", BACKEND_API_URL);

  // バックエンドAPIを呼び出し
  const response = await fetch(apiUrl.toString(), {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(
      `Fear & Greed Index データ状態取得エラー: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(
      `Fear & Greed Index データ状態取得に失敗しました: ${response.status}`
    );
  }

  const result = await response.json();

  if (!result.success) {
    console.error("Fear & Greed Index データ状態取得失敗:", result.message);
    throw new Error(
      result.message || "Fear & Greed Index データ状態取得に失敗しました"
    );
  }

  return result.data;
}

/**
 * GET /api/data/fear-greed/status
 *
 * Fear & Greed Index データの状態を取得します。
 */
export async function GET(request: NextRequest) {
  try {
    console.log("Fear & Greed Index データ状態取得リクエスト");

    const status = await fetchFearGreedDataStatus();

    const response: FearGreedStatusResponse = {
      success: true,
      message: "Fear & Greed Index データ状態を取得しました",
      data: status,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Fear & Greed Index データ状態取得エラー:", error);

    const errorResponse: FearGreedStatusResponse = {
      success: false,
      message:
        error instanceof Error
          ? error.message
          : "Fear & Greed Index データ状態取得中に予期せぬエラーが発生しました",
      data: {
        success: false,
        data_range: {
          oldest_data: null,
          newest_data: null,
          total_count: 0,
        },
        latest_timestamp: null,
        current_time: new Date().toISOString(),
        error: error instanceof Error ? error.message : "Unknown error",
      },
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
