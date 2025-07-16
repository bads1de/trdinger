/**
 * MLモデルバックアップAPI
 *
 * 特定のMLモデルのバックアップリクエストをバックエンドに転送します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/ml/models/[modelId]/backup
 *
 * 指定されたMLモデルをバックアップします。
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ modelId: string }> }
) {
  try {
    const { modelId } = await params;

    if (!modelId) {
      return NextResponse.json(
        { success: false, message: "モデルIDが指定されていません" },
        { status: 400 }
      );
    }

    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/ml/models/${encodeURIComponent(
      modelId
    )}/backup`;

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
          message: data.detail || "モデルバックアップに失敗しました",
        },
        { status: response.status }
      );
    }

    return NextResponse.json({ ...data, success: true });
  } catch (error) {
    console.error("MLモデルバックアップエラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
