/**
 * Fear & Greed Index データ収集API
 *
 * Alternative.me APIからFear & Greed Indexデータを収集するAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * データ収集結果の型定義
 */
export interface FearGreedCollectionResult {
  /** 成功フラグ */
  success: boolean;
  /** メッセージ */
  message: string;
  /** 取得件数 */
  fetched_count: number;
  /** 挿入件数 */
  inserted_count: number;
  /** 収集タイプ */
  collection_type?: string;
  /** エラー情報 */
  error?: string;
}

/**
 * データ収集レスポンスの型定義
 */
export interface FearGreedCollectionResponse {
  /** 成功フラグ */
  success: boolean;
  /** メッセージ */
  message: string;
  /** データ */
  data: FearGreedCollectionResult;
}

/**
 * バックエンドAPIでFear & Greed Indexデータを収集する関数
 *
 * @param limit 取得するデータ数
 * @returns 収集結果
 */
async function collectFearGreedData(
  limit: number = 30
): Promise<FearGreedCollectionResult> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/fear-greed/collect", BACKEND_API_URL);
  apiUrl.searchParams.set("limit", limit.toString());

  // バックエンドAPIを呼び出し
  const response = await fetch(apiUrl.toString(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(
      `Fear & Greed Index データ収集エラー: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(
      `Fear & Greed Index データ収集に失敗しました: ${response.status}`
    );
  }

  const result = await response.json();

  if (!result.success) {
    console.error("Fear & Greed Index データ収集失敗:", result.message);
    throw new Error(
      result.message || "Fear & Greed Index データ収集に失敗しました"
    );
  }

  return result.data;
}

/**
 * POST /api/data/fear-greed/collect
 *
 * Fear & Greed Index データを収集します。
 */
export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get("limit") || "30");

    console.log(`Fear & Greed Index データ収集リクエスト: limit=${limit}`);

    const result = await collectFearGreedData(limit);

    const response: FearGreedCollectionResponse = {
      success: true,
      message: result.message,
      data: result,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Fear & Greed Index データ収集エラー:", error);

    const errorResponse: FearGreedCollectionResponse = {
      success: false,
      message:
        error instanceof Error
          ? error.message
          : "Fear & Greed Index データ収集中に予期せぬエラーが発生しました",
      data: {
        success: false,
        message: "収集に失敗しました",
        fetched_count: 0,
        inserted_count: 0,
        error: error instanceof Error ? error.message : "Unknown error",
      },
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
