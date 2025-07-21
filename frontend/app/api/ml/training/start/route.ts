/**
 * MLトレーニング開始API
 *
 * フロントエンドからのMLトレーニング開始リクエストをバックエンドに転送します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";
import { convertSymbolForBackend } from "@/utils/symbolConverter";

/**
 * POST /api/ml/training/start
 *
 * MLモデルのトレーニングを開始します。
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // シンボル形式を変換
    if (body.symbol) {
      body.symbol = convertSymbolForBackend(body.symbol);
    }

    // バックエンドAPIに転送（最適化統合済みのエンドポイントを使用）
    const backendUrl = `${BACKEND_API_URL}/api/ml-training/train`;
    console.log("🔗 Calling backend URL:", backendUrl);
    console.log("📤 Request body:", JSON.stringify(body, null, 2));

    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        {
          success: false,
          message: data.detail || "トレーニング開始に失敗しました",
        },
        { status: response.status }
      );
    }

    return NextResponse.json({ ...data, success: true });
  } catch (error) {
    console.error("MLトレーニング開始エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
