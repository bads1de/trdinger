import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * プロファイル一覧取得API
 * GET /api/bayesian-optimization/profiles
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const targetModelType = searchParams.get("target_model_type");
    const includeInactive = searchParams.get("include_inactive") === "true";
    const limit = searchParams.get("limit");

    // バックエンドAPIへのリクエストパラメータを構築
    const params = new URLSearchParams();
    if (targetModelType) {
      params.append("target_model_type", targetModelType);
    }
    if (includeInactive) {
      params.append("include_inactive", "true");
    }
    if (limit) {
      params.append("limit", limit);
    }

    const backendUrl = `${BACKEND_API_URL}/api/bayesian-optimization/profiles${
      params.toString() ? `?${params.toString()}` : ""
    }`;

    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error("Backend API error:", errorData);
      return NextResponse.json(
        { 
          success: false, 
          error: "プロファイル一覧の取得に失敗しました",
          details: errorData 
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("プロファイル一覧取得エラー:", error);
    return NextResponse.json(
      { 
        success: false, 
        error: "プロファイル一覧の取得中にエラーが発生しました",
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}

/**
 * プロファイル作成API
 * POST /api/bayesian-optimization/profiles
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${BACKEND_API_URL}/api/bayesian-optimization/profiles`, {
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
          error: "プロファイルの作成に失敗しました",
          details: errorData 
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("プロファイル作成エラー:", error);
    return NextResponse.json(
      { 
        success: false, 
        error: "プロファイルの作成中にエラーが発生しました",
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}
