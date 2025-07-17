/**
 * Fear & Greed Index データ取得API
 *
 * データベースに保存されたFear & Greed Indexデータを取得するAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * Fear & Greed Index データの型定義
 */
export interface FearGreedIndexData {
  /** ID */
  id: number;
  /** インデックス値 (0-100) */
  value: number;
  /** 分類 (Extreme Fear, Fear, Neutral, Greed, Extreme Greed) */
  value_classification: string;
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
 * Fear & Greed Index レスポンスの型定義
 */
export interface FearGreedIndexResponse {
  /** 成功フラグ */
  success: boolean;
  /** メッセージ */
  message: string;
  /** データ */
  data: FearGreedIndexData[];
  /** メタデータ */
  metadata?: {
    count: number;
    start_date?: string;
    end_date?: string;
    limit?: number;
  };
}

/**
 * バックエンドAPIからFear & Greed Indexデータを取得する関数
 *
 * @param limit データ件数
 * @param startDate 開始日時
 * @param endDate 終了日時
 * @returns Fear & Greed Indexデータの配列
 */
async function fetchDatabaseFearGreedData(
  limit: number = 30,
  startDate?: string,
  endDate?: string
): Promise<FearGreedIndexData[]> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/fear-greed/data", BACKEND_API_URL);
  apiUrl.searchParams.set("limit", limit.toString());

  if (startDate) {
    apiUrl.searchParams.set("start_date", startDate);
  }
  if (endDate) {
    apiUrl.searchParams.set("end_date", endDate);
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
      `Fear & Greed Index データ取得エラー: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(
      `Fear & Greed Index データ取得に失敗しました: ${response.status}`
    );
  }

  const result = await response.json();

  if (!result.success) {
    console.error("Fear & Greed Index データ取得失敗:", result.message);
    throw new Error(
      result.message || "Fear & Greed Index データ取得に失敗しました"
    );
  }

  // バックエンドのレスポンス形式に対応
  return result.data?.data || result.data || [];
}

/**
 * GET /api/data/fear-greed
 *
 * Fear & Greed Index データを取得します。
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get("limit") || "30");
    const startDate = searchParams.get("start_date") || undefined;
    const endDate = searchParams.get("end_date") || undefined;

    console.log(
      `Fear & Greed Index データ取得リクエスト: limit=${limit}, start_date=${startDate}, end_date=${endDate}`
    );

    const data = await fetchDatabaseFearGreedData(limit, startDate, endDate);

    const response: FearGreedIndexResponse = {
      success: true,
      message: `Fear & Greed Index データを ${data.length} 件取得しました`,
      data: data,
      metadata: {
        count: data.length,
        start_date: startDate,
        end_date: endDate,
        limit: limit,
      },
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Fear & Greed Index データ取得エラー:", error);

    const errorResponse: FearGreedIndexResponse = {
      success: false,
      message:
        error instanceof Error
          ? error.message
          : "Fear & Greed Index データ取得中に予期せぬエラーが発生しました",
      data: [],
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
