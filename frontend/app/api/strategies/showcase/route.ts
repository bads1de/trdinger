/**
 * 戦略ショーケース一覧取得API
 *
 * バックエンドから戦略ショーケース一覧を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    // クエリパラメータを取得
    const { searchParams } = new URL(request.url);
    
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/strategies/showcase?${searchParams.toString()}`;

    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
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

    return NextResponse.json({
      success: true,
      strategies: data.strategies || [],
      total_count: data.total_count || 0,
      message: data.message || "戦略一覧を取得しました",
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error("戦略一覧取得エラー:", error);

    return NextResponse.json(
      {
        success: false,
        message: "サーバー内部エラーが発生しました",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
