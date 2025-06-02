/**
 * テクニカル指標計算API
 *
 * バックエンドAPIへのプロキシとしてテクニカル指標を計算します。
 */

import { NextRequest, NextResponse } from "next/server";
import { TechnicalIndicatorCalculationResponse } from "@/types/strategy";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/data/technical-indicators/calculate
 *
 * 指定された条件でテクニカル指標を計算します。
 */
export async function POST(request: NextRequest) {
  try {
    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || "BTC/USDT";
    const timeframe = searchParams.get("timeframe") || "1h";
    const indicatorType = searchParams.get("indicator_type") || "SMA";
    const period = searchParams.get("period") || "20";
    const limit = searchParams.get("limit");

    // バックエンドAPIのURLを構築
    const params = new URLSearchParams({
      symbol,
      timeframe,
      indicator_type: indicatorType,
      period,
    });

    if (limit) {
      params.append("limit", limit);
    }

    const backendUrl = `${BACKEND_API_URL}/api/technical-indicators/calculate?${params}`;

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
          message: result.detail || "テクニカル指標の計算に失敗しました",
          timestamp: new Date().toISOString(),
        },
        { status: response.status }
      );
    }

    // レスポンスの返却
    const calculationResponse: TechnicalIndicatorCalculationResponse = {
      success: true,
      data: result.data,
      message: result.message,
    };

    return NextResponse.json(calculationResponse);
  } catch (error) {
    console.error("テクニカル指標計算エラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "テクニカル指標の計算中にエラーが発生しました",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
