/**
 * バックテスト実行API
 *
 * バックエンドでバックテストを実行し、結果を返します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // リクエストボディのバリデーション
    const requiredFields = [
      "strategy_name",
      "symbol",
      "timeframe",
      "start_date",
      "end_date",
      "initial_capital",
      "commission_rate",
      "strategy_config",
    ];

    for (const field of requiredFields) {
      if (!body[field]) {
        return NextResponse.json(
          {
            success: false,
            error: `Missing required field: ${field}`,
            message: "Invalid request body",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        );
      }
    }

    // バックエンドAPIを呼び出し
    const response = await fetch(`${BACKEND_API_URL}/api/backtest/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("Backend API error:", {
        status: response.status,
        statusText: response.statusText,
        data: data,
      });
      throw new Error(data.detail || `Backend API error: ${response.status}`);
    }

    return NextResponse.json({
      success: true,
      result: data.result,
      message: "Backtest completed successfully",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error running backtest:", error);

    // より詳細なエラー情報をログに出力
    if (error instanceof Error) {
      console.error("Error details:", {
        name: error.name,
        message: error.message,
        stack: error.stack,
      });
    }

    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
        message: "Failed to run backtest",
        timestamp: new Date().toISOString(),
        details:
          error instanceof Error
            ? {
                name: error.name,
                stack: error.stack,
              }
            : null,
      },
      { status: 500 }
    );
  }
}
