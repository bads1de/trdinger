import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

/**
 * 特定プロファイル取得API
 * GET /api/bayesian-optimization/profiles/[id]
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const response = await fetch(
      `${BACKEND_URL}/api/bayesian-optimization/profiles/${id}`,
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
          error: "プロファイルの取得に失敗しました",
          details: errorData,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("プロファイル取得エラー:", error);
    return NextResponse.json(
      {
        success: false,
        error: "プロファイルの取得中にエラーが発生しました",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

/**
 * プロファイル更新API
 * PUT /api/bayesian-optimization/profiles/[id]
 */
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();

    const response = await fetch(
      `${BACKEND_URL}/api/bayesian-optimization/profiles/${id}`,
      {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      }
    );

    if (!response.ok) {
      const errorData = await response.text();
      console.error("Backend API error:", errorData);
      return NextResponse.json(
        {
          success: false,
          error: "プロファイルの更新に失敗しました",
          details: errorData,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("プロファイル更新エラー:", error);
    return NextResponse.json(
      {
        success: false,
        error: "プロファイルの更新中にエラーが発生しました",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

/**
 * プロファイル削除API
 * DELETE /api/bayesian-optimization/profiles/[id]
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const response = await fetch(
      `${BACKEND_URL}/api/bayesian-optimization/profiles/${id}`,
      {
        method: "DELETE",
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
          error: "プロファイルの削除に失敗しました",
          details: errorData,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("プロファイル削除エラー:", error);
    return NextResponse.json(
      {
        success: false,
        error: "プロファイルの削除中にエラーが発生しました",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
