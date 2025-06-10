/**
 * 戦略ショーケース生成API
 *
 * バックエンドで戦略ショーケースを生成します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/strategies/showcase/generate`;

    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
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
      message: data.message || "戦略生成を開始しました",
      generated_count: data.generated_count || 0,
      saved_count: data.saved_count || 0,
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error("戦略生成エラー:", error);

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
