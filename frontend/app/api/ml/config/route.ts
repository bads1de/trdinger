/**
 * ML設定取得API
 *
 * バックエンドのML設定APIへのプロキシエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * GET /api/ml/config
 *
 * ML設定情報を取得します。
 */
export async function GET(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/ml/config`;
    
    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: data.detail || "ML設定取得に失敗しました" },
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true, ...data });
  } catch (error) {
    console.error("ML設定取得エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}

/**
 * PUT /api/ml/config
 *
 * ML設定情報を更新します。
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/ml/config`;
    
    const response = await fetch(backendUrl, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: data.detail || "ML設定の更新に失敗しました" },
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true, ...data });
  } catch (error) {
    console.error("ML設定更新エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
