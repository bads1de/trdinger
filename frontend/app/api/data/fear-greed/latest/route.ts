/**
 * Fear & Greed Index 最新データ取得API
 *
 * 最新のFear & Greed Indexデータを取得するAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";
import { FearGreedIndexData, FearGreedIndexResponse } from "../route";

/**
 * バックエンドAPIから最新のFear & Greed Indexデータを取得する関数
 *
 * @param limit データ件数
 * @returns 最新のFear & Greed Indexデータの配列
 */
async function fetchLatestFearGreedData(
  limit: number = 30
): Promise<FearGreedIndexData[]> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/fear-greed/latest", BACKEND_API_URL);
  apiUrl.searchParams.set("limit", limit.toString());

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
      `最新Fear & Greed Index データ取得エラー: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(
      `最新Fear & Greed Index データ取得に失敗しました: ${response.status}`
    );
  }

  const result = await response.json();

  if (!result.success) {
    console.error("最新Fear & Greed Index データ取得失敗:", result.message);
    throw new Error(
      result.message || "最新Fear & Greed Index データ取得に失敗しました"
    );
  }

  // バックエンドのレスポンス形式に対応
  return result.data?.data || result.data || [];
}

/**
 * GET /api/data/fear-greed/latest
 *
 * 最新のFear & Greed Index データを取得します。
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get("limit") || "30");

    console.log(`最新Fear & Greed Index データ取得リクエスト: limit=${limit}`);

    const data = await fetchLatestFearGreedData(limit);

    const response: FearGreedIndexResponse = {
      success: true,
      message: `最新のFear & Greed Index データを ${data.length} 件取得しました`,
      data: data,
      metadata: {
        count: data.length,
        limit: limit,
      },
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("最新Fear & Greed Index データ取得エラー:", error);

    const errorResponse: FearGreedIndexResponse = {
      success: false,
      message:
        error instanceof Error
          ? error.message
          : "最新Fear & Greed Index データ取得中に予期せぬエラーが発生しました",
      data: [],
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
