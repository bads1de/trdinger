/**
 * 戦略ショーケース統計情報取得API
 *
 * 戦略ショーケースの統計情報を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/strategies/showcase/stats`;

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
      statistics: data.statistics || {
        total_strategies: 0,
        avg_return: 0,
        avg_sharpe_ratio: 0,
        avg_max_drawdown: 0,
        category_distribution: {},
        risk_distribution: {},
      },
      message: data.message || "統計情報を取得しました",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("統計情報取得エラー:", error);

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
