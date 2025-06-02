/**
 * テクニカル指標データ取得API
 *
 * バックエンドAPIへのプロキシとしてテクニカル指標データを取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { TechnicalIndicatorResponse } from "@/types/strategy";
import { BACKEND_API_URL } from "@/constants";

/**
 * GET /api/data/technical-indicators
 *
 * 指定された条件のテクニカル指標データを取得します。
 */
export async function GET(request: NextRequest) {
  try {
    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || "BTC/USDT";
    const timeframe = searchParams.get("timeframe") || "1h";
    const indicatorType = searchParams.get("indicator_type");
    const period = searchParams.get("period");
    const limit = searchParams.get("limit") || "100";
    const startDate = searchParams.get("start_date");
    const endDate = searchParams.get("end_date");

    // バックエンドAPIのURLを構築
    const params = new URLSearchParams({
      symbol,
      timeframe,
      limit,
    });

    if (indicatorType) {
      params.append("indicator_type", indicatorType);
    }
    if (period) {
      params.append("period", period);
    }
    if (startDate) {
      params.append("start_date", startDate);
    }
    if (endDate) {
      params.append("end_date", endDate);
    }

    const backendUrl = `${BACKEND_API_URL}/api/technical-indicators?${params}`;

    // バックエンドAPIを呼び出し
    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const result = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        {
          success: false,
          message: result.detail || "テクニカル指標データの取得に失敗しました",
          timestamp: new Date().toISOString(),
        },
        { status: response.status }
      );
    }

    // レスポンスの返却
    const technicalIndicatorResponse: TechnicalIndicatorResponse = {
      success: true,
      data: {
        symbol: result.data.symbol,
        timeframe: result.data.timeframe,
        indicator_type: result.data.indicator_type,
        period: result.data.period,
        count: result.data.count,
        technical_indicators: result.data.technical_indicators,
      },
      message: result.message,
    };

    return NextResponse.json(technicalIndicatorResponse);
  } catch (error) {
    console.error("テクニカル指標データ取得エラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "テクニカル指標データの取得中にエラーが発生しました",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
