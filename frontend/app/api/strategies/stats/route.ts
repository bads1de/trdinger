/**
 * 戦略統計API
 * 
 * 戦略の統計情報を取得するAPIエンドポイント
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/strategies/stats`,
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
      throw new Error(`Backend API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      stats: data.stats || {},
      message: "Strategy statistics retrieved successfully",
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error("Strategy statistics API error:", error);
    
    return NextResponse.json(
      {
        success: false,
        stats: {},
        error: error instanceof Error ? error.message : "Unknown error",
        message: "Failed to retrieve strategy statistics",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
