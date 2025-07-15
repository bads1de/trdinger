/**
 * MLトレーニング状態取得API
 *
 * バックエンドのMLトレーニング状態APIへのプロキシエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * GET /api/ml/status
 *
 * MLトレーニングの現在の状態を取得します。
 */
export async function GET(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/ml/status`;

    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("ML Status API - エラーレスポンス:", data);
      return NextResponse.json(
        { success: false, message: data.detail || "状態取得に失敗しました" },
        { status: response.status }
      );
    }

    const responseData = { success: true, ...data };
    return NextResponse.json(responseData);
  } catch (error) {
    console.error("MLトレーニング状態取得エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
