/**
 * MLハイパーパラメータベイジアン最適化API
 *
 * バックエンドでMLハイパーパラメータのベイジアン最適化を実行し、結果を返します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // リクエストボディのバリデーション
    if (!body.model_type) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required field: model_type",
          message: "Invalid request body",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // バックエンドAPIを呼び出し
    const response = await fetch(`${BACKEND_API_URL}/api/bayesian-optimization/ml-hyperparameters`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        {
          success: false,
          error: errorData.detail || "Backend API error",
          message: "MLハイパーパラメータのベイジアン最適化に失敗しました",
          timestamp: new Date().toISOString(),
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("MLハイパーパラメータベイジアン最適化API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        message: "MLハイパーパラメータのベイジアン最適化中にエラーが発生しました",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
