/**
 * ML設定リセットAPI
 * 
 * ML設定のリセットリクエストをバックエンドに転送します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/ml/config/reset
 *
 * ML設定をデフォルト値にリセットします。
 */
export async function POST(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/ml/config/reset`;
    
    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: data.detail || "設定のリセットに失敗しました" },
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true, ...data });
  } catch (error) {
    console.error("ML設定リセットエラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}