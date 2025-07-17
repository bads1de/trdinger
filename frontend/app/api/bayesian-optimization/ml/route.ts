import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * MLハイパーパラメータ最適化API（プロファイル保存機能付き）
 * POST /api/bayesian-optimization/ml
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // リクエストボディの検証
    if (!body.model_type) {
      return NextResponse.json(
        { 
          success: false, 
          error: "model_typeは必須です" 
        },
        { status: 400 }
      );
    }

    const response = await fetch(`${BACKEND_API_URL}/api/bayesian-optimization/ml`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error("Backend API error:", errorData);
      return NextResponse.json(
        { 
          success: false, 
          error: "ベイジアン最適化の実行に失敗しました",
          details: errorData 
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("ベイジアン最適化エラー:", error);
    return NextResponse.json(
      { 
        success: false, 
        error: "ベイジアン最適化の実行中にエラーが発生しました",
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}
