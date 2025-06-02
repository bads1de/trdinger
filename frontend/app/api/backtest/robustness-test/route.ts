/**
 * ロバストネステストAPI
 *
 * バックエンドでロバストネステストを実行し、結果を返します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // リクエストボディのバリデーション
    if (!body.base_config || !body.test_periods || !body.optimization_params) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required fields: base_config, test_periods, or optimization_params",
          message: "Invalid request body",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // バックエンドAPIを呼び出し
    const response = await fetch(`${BACKEND_API_URL}/api/backtest/robustness-test`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Backend API error: ${response.status} ${response.statusText}`, errorText);
      
      return NextResponse.json(
        {
          success: false,
          error: `Backend API error: ${response.status}`,
          message: errorText || "Robustness test failed",
          timestamp: new Date().toISOString(),
        },
        { status: response.status }
      );
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      result: data.result,
      message: "Robustness test completed successfully",
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error("Robustness test API error:", error);
    
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        message: "Robustness test failed",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
