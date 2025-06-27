/**
 * ストラテジービルダー個別戦略取得API
 *
 * バックエンドから指定されたIDの戦略詳細を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const strategyId = params.id;

    // IDのバリデーション
    if (!strategyId || isNaN(Number(strategyId))) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid strategy ID",
          message: "戦略IDが無効です",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/strategy-builder/strategies/${strategyId}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        // キャッシュを無効化して最新データを取得
        cache: "no-store",
      }
    );

    if (!response.ok) {
      console.error(
        `バックエンドAPIエラー: ${response.status} ${response.statusText}`
      );

      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        {
          success: false,
          message: `バックエンドAPIエラー: ${response.status}`,
          error: errorData.detail || response.statusText,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("戦略詳細取得エラー:", error);

    return NextResponse.json(
      {
        success: false,
        message: "戦略詳細の取得に失敗しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
