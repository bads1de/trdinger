/**
 * 戦略取得API
 *
 * バックエンドの /api/strategies エンドポイントを呼び出し、
 * 生成済み戦略の一覧を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // バックエンドAPIに転送するパラメータを構築
    const params = new URLSearchParams();

    // パラメータを動的に追加
    const allowedParams = [
      "limit",
      "offset",
      "sort_by",
      "sort_order",
      "category",
      "risk_level",
      "experiment_id",
      "min_fitness",
      "search_query",
    ];

    allowedParams.forEach((param) => {
      if (searchParams.has(param)) {
        params.append(param, searchParams.get(param)!);
      }
    });

    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/strategies?${params.toString()}`,
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
      const errorData = await response.text();
      console.error("Backend API Error Response:", errorData);
      throw new Error(
        `Backend API error: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      strategies: data.data.strategies || [],
      total_count: data.data.total_count || 0,
      has_more: data.data.has_more || false,
      message: data.message || "Strategies retrieved successfully",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Strategies API error:", error);

    return NextResponse.json(
      {
        success: false,
        strategies: [],
        total_count: 0,
        has_more: false,
        error: error instanceof Error ? error.message : "Unknown error",
        message: "Failed to retrieve strategies",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
