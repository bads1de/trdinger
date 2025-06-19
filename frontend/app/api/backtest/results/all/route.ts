/**
 * バックテスト結果一括削除API
 *
 * すべてのバックテスト結果をバックエンドから削除します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function DELETE(request: NextRequest) {
  try {
    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/backtest/results-all`,
      {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Backend API error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      message: `すべてのバックテスト結果を削除しました (${data.deleted_count}件)`,
      deleted_count: data.deleted_count,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Delete all backtest results error:", error);

    return NextResponse.json(
      {
        success: false,
        message: "バックテスト結果の一括削除に失敗しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
