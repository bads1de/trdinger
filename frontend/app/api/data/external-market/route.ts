/**
 * 外部市場データ取得API
 *
 * データベースに保存された外部市場データ（SP500、NASDAQ、DXY、VIXなど）を取得するAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * 外部市場データの型定義
 */
export interface ExternalMarketData {
  /** ID */
  id: number;
  /** シンボル（例: ^GSPC, ^IXIC, DX-Y.NYB, ^VIX） */
  symbol: string;
  /** 始値 */
  open: number;
  /** 高値 */
  high: number;
  /** 安値 */
  low: number;
  /** 終値 */
  close: number;
  /** 出来高（VIXなどでは null の場合がある） */
  volume: number | null;
  /** データの日付（ISO形式） */
  data_timestamp: string;
  /** データ取得時刻（ISO形式） */
  timestamp: string;
  /** 作成日時（ISO形式） */
  created_at: string;
  /** 更新日時（ISO形式） */
  updated_at: string;
}

/**
 * 外部市場データレスポンスの型定義
 */
export interface ExternalMarketDataResponse {
  /** 成功フラグ */
  success: boolean;
  /** メッセージ */
  message: string;
  /** データ */
  data: ExternalMarketData[];
  /** メタデータ */
  metadata?: {
    count: number;
    symbol?: string;
    start_time?: string;
    end_time?: string;
    limit: number;
  };
}

/**
 * バックエンドAPIから外部市場データを取得する関数
 *
 * @param symbol シンボル
 * @param limit データ件数
 * @param startTime 開始時刻
 * @param endTime 終了時刻
 * @returns 外部市場データの配列
 */
async function fetchDatabaseExternalMarketData(
  symbol?: string,
  limit: number = 100,
  startTime?: string,
  endTime?: string
): Promise<ExternalMarketData[]> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/external-market/", BACKEND_API_URL);
  apiUrl.searchParams.set("limit", limit.toString());

  if (symbol) {
    apiUrl.searchParams.set("symbol", symbol);
  }
  if (startTime) {
    apiUrl.searchParams.set("start_time", startTime);
  }
  if (endTime) {
    apiUrl.searchParams.set("end_time", endTime);
  }

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
      `外部市場データ取得エラー: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(`外部市場データ取得に失敗しました: ${response.status}`);
  }

  const result = await response.json();

  if (!result.success) {
    console.error("外部市場データ取得失敗:", result.message);
    throw new Error(result.message || "外部市場データ取得に失敗しました");
  }

  return result.data;
}

/**
 * GET /api/data/external-market
 *
 * 外部市場データを取得します。
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || undefined;
    const limit = parseInt(searchParams.get("limit") || "100");
    const startTime = searchParams.get("start_time") || undefined;
    const endTime = searchParams.get("end_time") || undefined;

    console.log(
      `外部市場データ取得リクエスト: symbol=${symbol}, limit=${limit}, start_time=${startTime}, end_time=${endTime}`
    );

    const data = await fetchDatabaseExternalMarketData(symbol, limit, startTime, endTime);

    const response: ExternalMarketDataResponse = {
      success: true,
      message: `外部市場データを ${data.length} 件取得しました`,
      data: data,
      metadata: {
        count: data.length,
        symbol: symbol,
        start_time: startTime,
        end_time: endTime,
        limit: limit,
      },
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("外部市場データ取得エラー:", error);

    const errorResponse: ExternalMarketDataResponse = {
      success: false,
      message: error instanceof Error ? error.message : "外部市場データ取得に失敗しました",
      data: [],
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
