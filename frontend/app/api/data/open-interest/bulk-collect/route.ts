/**
 * オープンインタレストデータ一括収集API
 *
 * バックエンドAPIを呼び出してBTC・ETHのオープンインタレストデータを
 * 一括で収集し、データベースに保存するAPIエンドポイントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/data/open-interest/bulk-collect
 *
 * BTC・ETHシンボルのオープンインタレストデータを一括収集します。
 */
export async function POST(request: NextRequest) {
  try {
    console.log("オープンインタレストデータ一括収集リクエスト開始");

    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/open-interest/bulk-collect`;

    console.log(`バックエンドURL: ${backendUrl}`);

    const response = await fetch(backendUrl, {
      method: "POST",
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
    console.log(
      `オープンインタレストデータ一括収集成功: ${data.data?.summary?.total_saved || 0}件保存`
    );

    return NextResponse.json(data);
  } catch (error) {
    console.error("オープンインタレストデータ一括収集エラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "オープンインタレストデータの一括収集中にエラーが発生しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
