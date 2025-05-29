/**
 * オープンインタレストデータ取得API
 *
 * バックエンドAPIからオープンインタレストデータを取得し、
 * フロントエンドに返すAPIエンドポイントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * GET /api/data/open-interest
 *
 * 指定されたシンボルのオープンインタレストデータを取得します。
 */
export async function GET(request: NextRequest) {
  try {
    console.log("オープンインタレストデータ取得リクエスト開始");

    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || "BTC/USDT";
    const startDate = searchParams.get("start_date");
    const endDate = searchParams.get("end_date");
    const limit = searchParams.get("limit") || "1000";

    console.log(`取得対象: ${symbol}, 件数: ${limit}`);

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

    console.log(`バックエンドURL: ${backendUrl}`);

    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      console.error(`バックエンドAPIエラー: ${response.status} ${response.statusText}`);
      return NextResponse.json(
        {
          success: false,
          message: `バックエンドAPIエラー: ${response.status}`,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log(`オープンインタレストデータ取得成功: ${data.data?.count || 0}件`);

    return NextResponse.json(data);
  } catch (error) {
    console.error("オープンインタレストデータ取得エラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "オープンインタレストデータの取得中にエラーが発生しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
