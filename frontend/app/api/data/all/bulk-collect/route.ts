/**
 * 全データ一括収集API
 *
 * フロントエンドからの全データ一括収集リクエストを受け取り、
 * バックエンドAPIに転送して全データ（OHLCV・FR・OI・TI）の一括収集を実行するAPIエンドポイントです。
 *
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/data/all/bulk-collect
 *
 * 全データ（OHLCV・Funding Rate・Open Interest・Technical Indicators）を一括収集します。
 * 既存データをチェックし、データが存在しない組み合わせのみ収集を実行します。
 * OHLCVデータ収集後にテクニカル指標も自動計算します。
 */
export async function POST(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/data-collection/all/bulk-collect`;

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

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("全データ一括収集APIエラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "全データ一括収集中にエラーが発生しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
