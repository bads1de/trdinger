/**
 * データ状況取得APIルート
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${BACKEND_API_URL}/api/data-reset/status`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        {
          success: false,
          message: data.message || "データ状況の取得に失敗しました",
        },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("データ状況取得APIエラー:", error);
    return NextResponse.json(
      { success: false, message: "データ状況取得中にエラーが発生しました" },
      { status: 500 }
    );
  }
}
