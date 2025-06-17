/**
 * リスクレベル一覧取得API
 *
 * 利用可能なリスクレベル一覧を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/strategies/risk-levels`;

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
      risk_levels: data.risk_levels || {
        low: "低リスク",
        medium: "中リスク",
        high: "高リスク",
      },
      message: data.message || "リスクレベル一覧を取得しました",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("リスクレベル一覧取得エラー:", error);

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
