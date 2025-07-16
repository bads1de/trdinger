/**
 * ML特徴量重要度取得API
 *
 * バックエンドのML特徴量重要度APIへのプロキシエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * GET /api/ml/feature-importance
 *
 * MLモデルの特徴量重要度を取得します。
 */
export async function GET(request: NextRequest) {
  try {
    // クエリパラメータを取得
    const { searchParams } = new URL(request.url);
    const topN = searchParams.get("top_n") || "10";
    
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/ml/feature-importance?top_n=${topN}`;
    
    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: data.detail || "特徴量重要度取得に失敗しました" },
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true, ...data });
  } catch (error) {
    console.error("ML特徴量重要度取得エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
