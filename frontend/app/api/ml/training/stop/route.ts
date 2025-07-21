/**
 * MLトレーニング停止API
 *
 * フロントエンドからのMLトレーニング停止リクエストをバックエンドに転送します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/ml/training/stop
 *
 * MLトレーニングを停止します。
 */
export async function POST(request: NextRequest) {
  try {
    // バックエンドAPIに転送（最適化統合済みのエンドポイントを使用）
    const backendUrl = `${BACKEND_API_URL}/api/ml-training/stop`;

    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        {
          success: false,
          message: data.detail || "トレーニング停止に失敗しました",
        },
        { status: response.status }
      );
    }

    return NextResponse.json({ ...data, success: true });
  } catch (error) {
    console.error("MLトレーニング停止エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
