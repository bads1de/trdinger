/**
 * ストラテジービルダー指標一覧取得API
 *
 * バックエンドから利用可能なテクニカル指標の一覧を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    // バックエンドAPIを呼び出し
    const response = await fetch(`${BACKEND_API_URL}/api/strategy-builder/indicators`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      // キャッシュを無効化して最新データを取得
      cache: "no-store",
    });

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
    console.error("指標一覧取得エラー:", error);

    return NextResponse.json(
      {
        success: false,
        message: "指標一覧の取得に失敗しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
