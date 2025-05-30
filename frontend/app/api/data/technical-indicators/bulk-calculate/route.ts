/**
 * テクニカル指標一括計算API
 *
 * バックエンドAPIへのプロキシとしてテクニカル指標を一括計算します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BulkTechnicalIndicatorCalculationResult } from "@/types/strategy";

// バックエンドAPIのベースURL
const BACKEND_API_URL = process.env.BACKEND_API_URL || "http://localhost:8000";

/**
 * POST /api/data/technical-indicators/bulk-calculate
 *
 * 指定された条件でテクニカル指標を一括計算します。
 */
export async function POST(request: NextRequest) {
  try {
    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || "BTC/USDT";
    const timeframe = searchParams.get("timeframe") || "1h";
    const useDefault = searchParams.get("use_default") !== "false";
    const limit = searchParams.get("limit");

    // バックエンドAPIのURLを構築
    const params = new URLSearchParams({
      symbol,
      timeframe,
      use_default: useDefault.toString(),
    });

    if (limit) {
      params.append("limit", limit);
    }

    const backendUrl = `${BACKEND_API_URL}/api/technical-indicators/bulk-calculate?${params}`;

    // バックエンドAPIを呼び出し
    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const result = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        {
          success: false,
          message: result.detail || "テクニカル指標の一括計算に失敗しました",
          timestamp: new Date().toISOString(),
        },
        { status: response.status }
      );
    }

    // レスポンスの返却
    const bulkCalculationResponse = {
      success: true,
      data: result.data as BulkTechnicalIndicatorCalculationResult,
      message: result.message,
    };

    return NextResponse.json(bulkCalculationResponse);
  } catch (error) {
    console.error("テクニカル指標一括計算エラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "テクニカル指標の一括計算中にエラーが発生しました",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
