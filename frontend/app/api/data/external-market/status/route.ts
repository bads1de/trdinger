/**
 * 外部市場データ状態確認API
 *
 * バックエンドAPIを通じて外部市場データの状態を確認するAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * 外部市場データ統計情報の型定義
 */
export interface ExternalMarketDataStatistics {
  /** データ件数 */
  count: number;
  /** シンボル一覧 */
  symbols: string[];
  /** シンボル数 */
  symbol_count: number;
  /** 日付範囲 */
  date_range?: {
    oldest: string;
    newest: string;
  };
}

/**
 * 外部市場データ状態の型定義
 */
export interface ExternalMarketDataStatus {
  /** 成功フラグ */
  success: boolean;
  /** 統計情報 */
  statistics: ExternalMarketDataStatistics;
  /** 最新タイムスタンプ */
  latest_timestamp?: string;
  /** 現在時刻 */
  current_time: string;
  /** エラー（失敗時） */
  error?: string;
}

/**
 * 外部市場データ状態レスポンスの型定義
 */
export interface ExternalMarketDataStatusResponse {
  /** 成功フラグ */
  success: boolean;
  /** メッセージ */
  message: string;
  /** データ */
  data: ExternalMarketDataStatus;
}

/**
 * バックエンドAPIから外部市場データの状態を取得する関数
 *
 * @returns データ状態情報
 */
async function fetchExternalMarketDataStatus(): Promise<ExternalMarketDataStatus> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/external-market/status", BACKEND_API_URL);

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
      `外部市場データ状態取得エラー: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(`外部市場データ状態取得に失敗しました: ${response.status}`);
  }

  const result = await response.json();

  if (!result.success) {
    console.error("外部市場データ状態取得失敗:", result.message);
    throw new Error(result.message || "外部市場データ状態取得に失敗しました");
  }

  return result.data;
}

/**
 * GET /api/data/external-market/status
 *
 * 外部市場データの状態を取得します。
 */
export async function GET(request: NextRequest) {
  try {
    console.log("外部市場データ状態取得リクエスト");

    const status = await fetchExternalMarketDataStatus();

    const response: ExternalMarketDataStatusResponse = {
      success: true,
      message: "外部市場データの状態を取得しました",
      data: status,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("外部市場データ状態取得エラー:", error);

    const errorResponse: ExternalMarketDataStatusResponse = {
      success: false,
      message: error instanceof Error ? error.message : "外部市場データ状態取得に失敗しました",
      data: {
        success: false,
        statistics: {
          count: 0,
          symbols: [],
          symbol_count: 0,
        },
        current_time: new Date().toISOString(),
        error: error instanceof Error ? error.message : "Unknown error",
      },
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
