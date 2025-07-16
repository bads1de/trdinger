/**
 * MLトレーニング停止API
 *
 * バックエンドのMLトレーニング停止APIへのプロキシエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/ml/stop
 *
 * MLトレーニングを停止します。
 */
export async function POST(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/ml/stop`;
    
    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: data.detail || "トレーニング停止に失敗しました" },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("MLトレーニング停止エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
