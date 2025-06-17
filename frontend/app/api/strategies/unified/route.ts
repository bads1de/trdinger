/**
 * 統合戦略API
 *
 * ショーケース戦略とオートストラテジー戦略を統合して取得するAPIエンドポイント
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
    const sort_by = searchParams.get("sort_by") || "created_at";
    const sort_order = searchParams.get("sort_order") || "desc";
    params.append("sort_by", sort_by);
    params.append("sort_order", sort_order);

    // フィルター
    const category = searchParams.get("category");
    if (category) params.append("category", category);

    const risk_level = searchParams.get("risk_level");
    if (risk_level) params.append("risk_level", risk_level);

    const search_query = searchParams.get("search_query");
    if (search_query) params.append("search_query", search_query);

    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/strategies/unified?${params.toString()}`,
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
      message: "Unified strategies retrieved successfully",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Unified strategies API error:", error);

    return NextResponse.json(
      {
        success: false,
        strategies: [],
        total_count: 0,
        has_more: false,
        error: error instanceof Error ? error.message : "Unknown error",
        message: "Failed to retrieve unified strategies",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
