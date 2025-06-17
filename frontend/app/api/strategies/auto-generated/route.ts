/**
 * オートストラテジー戦略API
 *
 * オートストラテジーで生成された戦略のみを取得するAPIエンドポイント
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    // クエリパラメータを取得
    const { searchParams } = new URL(request.url);

    // バックエンドAPIに転送するパラメータを構築
    const params = new URLSearchParams();

    // ページネーション
    const limit = searchParams.get("limit") || "20";
    const offset = searchParams.get("offset") || "0";
    params.append("limit", limit);
    params.append("offset", offset);

    // ソート
    const sort_by = searchParams.get("sort_by") || "fitness_score";
    const sort_order = searchParams.get("sort_order") || "desc";
    params.append("sort_by", sort_by);
    params.append("sort_order", sort_order);

    // オートストラテジー固有のフィルター
    const experiment_id = searchParams.get("experiment_id");
    if (experiment_id) params.append("experiment_id", experiment_id);

    const min_fitness = searchParams.get("min_fitness");
    if (min_fitness) params.append("min_fitness", min_fitness);

    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/strategies/auto-generated?${params.toString()}`,
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
      throw new Error(
        `Backend API error: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      strategies: data.strategies || [],
      total_count: data.total_count || 0,
      has_more: data.has_more || false,
      message: "Auto-generated strategies retrieved successfully",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Auto-generated strategies API error:", error);

    return NextResponse.json(
      {
        success: false,
        strategies: [],
        total_count: 0,
        has_more: false,
        error: error instanceof Error ? error.message : "Unknown error",
        message: "Failed to retrieve auto-generated strategies",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
