/**
 * MLモデル個別操作API
 * 
 * 特定のMLモデルに対する操作（削除など）をバックエンドに転送します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * DELETE /api/ml/models/[modelId]
 *
 * 指定されたMLモデルを削除します。
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: { modelId: string } }
) {
  try {
    const { modelId } = await params;

    if (!modelId) {
      return NextResponse.json(
        { success: false, message: "モデルIDが指定されていません" },
        { status: 400 }
      );
    }

    console.log("削除要求されたモデルID:", modelId);

    // バックエンドAPIに転送（modelIdは既にURLデコードされているのでそのまま使用）
    const backendUrl = `${BACKEND_API_URL}/api/ml/models/${encodeURIComponent(modelId)}`;

    console.log("バックエンドURL:", backendUrl);

    const response = await fetch(backendUrl, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: data.detail || "モデル削除に失敗しました" },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("MLモデル削除エラー:", error);
    return NextResponse.json(
      { success: false, message: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
