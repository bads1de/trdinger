/**
 * MLモデルクリーンアップAPI
 * 
 * 古いMLモデルファイルのクリーンアップリクエストをバックエンドに転送します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/ml/models/cleanup
 *
 * 古いMLモデルファイルをクリーンアップします。
 */
export async function POST(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/ml/models/cleanup`;
    
    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: data.detail || "モデルクリーンアップに失敗しました" },
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true, ...data });
  } catch (error) {
    console.error("MLモデルクリーンアップエラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
