/**
 * OIデータ取得API
 *
 * バックエンドAPIからOIデータを取得し、
 * フロントエンドに返すAPIエンドポイントです。
 *
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * GET /api/data/open-interest
 *
 * 指定されたシンボルのOIデータを取得します。
 */
export async function GET(request: NextRequest) {
  try {
    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || "BTC/USDT";
    const startDate = searchParams.get("start_date");
    const endDate = searchParams.get("end_date");
    const limit = searchParams.get("limit") || "1000";

    // バックエンドAPIに転送
    let backendUrl = `${BACKEND_API_URL}/api/open-interest?symbol=${encodeURIComponent(
      symbol
    )}&limit=${limit}`;

    if (startDate) {
      backendUrl += `&start_date=${encodeURIComponent(startDate)}`;
    }
    if (endDate) {
      backendUrl += `&end_date=${encodeURIComponent(endDate)}`;
    }

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

    return NextResponse.json(data);
  } catch (error) {
    console.error("OIデータ取得エラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "OIデータの取得中にエラーが発生しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
