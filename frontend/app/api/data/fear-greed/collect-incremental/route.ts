/**
 * Fear & Greed Index 差分データ収集API
 *
 * Fear & Greed Indexデータの差分収集を行うAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * 収集結果の型定義
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
 * 収集レスポンスの型定義
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
 * バックエンドAPIでFear & Greed Indexデータの差分収集を実行する関数
 *
 * @param limit 取得するデータ数
 * @returns 収集結果
 */
async function collectIncrementalFearGreedData(limit: number = 30): Promise<FearGreedCollectionResult> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/fear-greed/collect-incremental", BACKEND_API_URL);
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
      `Fear & Greed Index 差分データ収集エラー: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(
      `Fear & Greed Index 差分データ収集に失敗しました: ${response.status}`
    );
  }

  const result = await response.json();

  if (!result.success) {
    console.error("Fear & Greed Index 差分データ収集失敗:", result.message);
    throw new Error(
      result.message || "Fear & Greed Index 差分データ収集に失敗しました"
    );
  }

  return result.data;
}

/**
 * POST /api/data/fear-greed/collect-incremental
 *
 * Fear & Greed Index データの差分収集を実行します。
 */
export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get("limit") || "30");

    console.log(`Fear & Greed Index 差分データ収集リクエスト: limit=${limit}`);

    const result = await collectIncrementalFearGreedData(limit);

    const response: FearGreedCollectionResponse = {
      success: true,
      message: "Fear & Greed Index 差分データ収集が完了しました",
      data: result,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Fear & Greed Index 差分データ収集エラー:", error);

    const errorResponse: FearGreedCollectionResponse = {
      success: false,
      message:
        error instanceof Error
          ? error.message
          : "Fear & Greed Index 差分データ収集中に予期せぬエラーが発生しました",
      data: {
        success: false,
        message: error instanceof Error ? error.message : "Unknown error",
        fetched_count: 0,
        inserted_count: 0,
        error: error instanceof Error ? error.message : "Unknown error",
      },
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
