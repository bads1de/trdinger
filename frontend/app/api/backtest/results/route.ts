/**
 * バックテスト結果一覧取得API
 *
 * バックエンドから過去のバックテスト結果一覧を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // クエリパラメータを取得
    const limit = searchParams.get("limit") || "50";
    const offset = searchParams.get("offset") || "0";
    const symbol = searchParams.get("symbol");
    const strategy_name = searchParams.get("strategy_name");

    // バックエンドAPIのURLを構築
    const backendUrl = new URL(`${BACKEND_API_URL}/api/backtest/results`);
    backendUrl.searchParams.set("limit", limit);
    backendUrl.searchParams.set("offset", offset);

    if (symbol) {
      backendUrl.searchParams.set("symbol", symbol);
    }
    if (strategy_name) {
      backendUrl.searchParams.set("strategy_name", strategy_name);
    }

    const response = await fetch(backendUrl.toString(), {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Backend API error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      results: data.results || [],
      total: data.total || 0,
      limit: parseInt(limit),
      offset: parseInt(offset),
      message: "Backtest results retrieved successfully",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error fetching backtest results:", error);

    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
        message: "Failed to fetch backtest results",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
