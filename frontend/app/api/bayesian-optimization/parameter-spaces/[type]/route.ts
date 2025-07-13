/**
 * デフォルトパラメータ空間取得API
 *
 * バックエンドからデフォルトパラメータ空間を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(
  request: NextRequest,
  { params }: { params: { type: string } }
) {
  try {
    const { type } = params;

    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/bayesian-optimization/parameter-spaces/${type}`,
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
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        {
          success: false,
          error: errorData.detail || "Backend API error",
          message: "デフォルトパラメータ空間の取得に失敗しました",
          timestamp: new Date().toISOString(),
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("デフォルトパラメータ空間取得API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        message: "デフォルトパラメータ空間の取得中にエラーが発生しました",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
