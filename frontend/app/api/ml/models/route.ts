/**
 * MLモデル一覧取得API
 *
 * フロントエンドからのMLモデル一覧取得リクエストをバックエンドに転送します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * GET /api/ml/models
 *
 * MLモデルの一覧を取得します。
 */
export async function GET(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/ml/models`;

    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("ML Models API - エラーレスポンス:", data);
      return NextResponse.json(
        {
          success: false,
          message: data.detail || "モデル一覧取得に失敗しました",
        },
        { status: response.status }
      );
    }

    // レスポンスデータにsuccessフラグを追加
    const responseData = { success: true, ...data };
    console.log("ML Models API - 成功レスポンス:", {
      modelCount: data.models?.length || 0,
      models: data.models?.map((m: any) => ({
        name: m.name,
        accuracy: m.accuracy,
        f1_score: m.f1_score,
        feature_count: m.feature_count,
      })) || [],
    });

    return NextResponse.json(responseData);
  } catch (error) {
    console.error("MLモデル一覧取得エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
