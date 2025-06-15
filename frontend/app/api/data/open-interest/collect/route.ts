/**
 * OIデータ収集API
 *
 * バックエンドAPIを呼び出してOIデータを収集し、
 * データベースに保存するAPIエンドポイントです。
 *
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/data/open-interest/collect
 *
 * 指定されたシンボルのOIデータを収集します。
 */
export async function POST(request: NextRequest) {
  try {
    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || "BTC/USDT:USDT";
    const limit = searchParams.get("limit") || "100";
    const fetchAll = searchParams.get("fetch_all") === "true";

    // シンボルバリデーション (簡易版)
    if (typeof symbol !== "string" || symbol.trim() === "") {
      return NextResponse.json(
        {
          success: false,
          message: "無効なシンボルです",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // バックエンドAPIに転送
    let backendUrl = `${BACKEND_API_URL}/api/open-interest/collect?symbol=${encodeURIComponent(
      symbol
    )}&limit=${limit}`;
    if (fetchAll) {
      backendUrl += "&fetch_all=true";
    }

    const response = await fetch(backendUrl, {
      method: "POST",
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
    console.error("OIデータ収集エラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "OIデータの収集中にエラーが発生しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
