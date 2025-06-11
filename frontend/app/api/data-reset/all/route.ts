/**
 * 全データリセットAPIルート
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function DELETE(request: NextRequest) {
  try {
    const response = await fetch(`${BACKEND_API_URL}/api/data-reset/all`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        {
          success: false,
          message: data.message || "データリセットに失敗しました",
        },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("全データリセットAPIエラー:", error);
    return NextResponse.json(
      { success: false, message: "データリセット中にエラーが発生しました" },
      { status: 500 }
    );
  }
}
