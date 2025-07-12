/**
 * MLトレーニング状態取得API
 * 
 * フロントエンドからのMLトレーニング状態確認リクエストをバックエンドに転送します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * GET /api/ml/training/status
 *
 * MLトレーニングの現在の状態を取得します。
 */
export async function GET(request: NextRequest) {
  try {
    // バックエンドAPIに転送（ml_managementの/training/statusエンドポイントを使用）
    const backendUrl = `${BACKEND_API_URL}/api/ml/training/status`;
    
    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: data.detail || "トレーニング状態取得に失敗しました" },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("MLトレーニング状態取得エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
