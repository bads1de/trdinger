import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

/**
 * デフォルトプロファイル取得API
 * GET /api/bayesian-optimization/profiles/default/[modelType]
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ modelType: string }> }
) {
  try {
    const { modelType } = await params;

    const response = await fetch(
      `${BACKEND_URL}/api/bayesian-optimization/profiles/default/${encodeURIComponent(
        modelType
      )}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
      const errorData = await response.text();
      console.error("Backend API error:", errorData);
      return NextResponse.json(
        {
          success: false,
          error: "デフォルトプロファイルの取得に失敗しました",
          details: errorData,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("デフォルトプロファイル取得エラー:", error);
    return NextResponse.json(
      {
        success: false,
        error: "デフォルトプロファイルの取得中にエラーが発生しました",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
