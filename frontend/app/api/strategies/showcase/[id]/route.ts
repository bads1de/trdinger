/**
 * 戦略ショーケース詳細取得API
 *
 * 指定されたIDの戦略詳細を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;
    
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/strategies/showcase/${id}`;

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

      if (response.status === 404) {
        return NextResponse.json(
          {
            success: false,
            message: "戦略が見つかりませんでした",
          },
          { status: 404 }
        );
      }

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
      strategy: data.strategy || null,
      message: data.message || "戦略詳細を取得しました",
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error("戦略詳細取得エラー:", error);

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
