/**
 * OIデータ一括収集API
 *
 * バックエンドAPIを呼び出してBTC・ETHのOIデータを
 * 一括で収集し、データベースに保存するAPIエンドポイントです。
 *
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/data/open-interest/bulk-collect
 *
 * BTCのOIデータを一括収集します。
 */
export async function POST(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/open-interest/bulk-collect`;

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
    console.error("OIデータ一括収集エラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "OIデータの一括収集中にエラーが発生しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
