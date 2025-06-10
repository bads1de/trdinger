/**
 * 戦略カテゴリ一覧取得API
 *
 * 利用可能な戦略カテゴリ一覧を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/strategies/categories`;

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
      categories: data.categories || {
        trend_following: "トレンドフォロー",
        mean_reversion: "逆張り",
        breakout: "ブレイクアウト",
        range_trading: "レンジ取引",
        momentum: "モメンタム"
      },
      message: data.message || "カテゴリ一覧を取得しました",
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error("カテゴリ一覧取得エラー:", error);

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
