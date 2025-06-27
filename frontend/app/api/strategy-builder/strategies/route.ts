/**
 * ストラテジービルダー戦略一覧取得API
 *
 * バックエンドから保存済み戦略の一覧を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const activeOnly = searchParams.get("active_only") || "true";
    const limit = searchParams.get("limit");

    // クエリパラメータを構築
    const queryParams = new URLSearchParams();
    queryParams.append("active_only", activeOnly);
    if (limit) {
      queryParams.append("limit", limit);
    }

    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/strategy-builder/strategies?${queryParams.toString()}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        // キャッシュを無効化して最新データを取得
        cache: "no-store",
      }
    );

    if (!response.ok) {
      console.error(
        `バックエンドAPIエラー: ${response.status} ${response.statusText}`
      );

      return NextResponse.json(
        {
          success: false,
          message: `バックエンドAPIエラー: ${response.status}`,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("戦略一覧取得エラー:", error);

    return NextResponse.json(
      {
        success: false,
        message: "戦略一覧の取得に失敗しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
