/**
 * ファンディングレートデータリセットAPIルート
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function DELETE(request: NextRequest) {
  try {
    const response = await fetch(
      `${BACKEND_API_URL}/api/data-reset/funding-rates`,
      {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        {
          success: false,
          message:
            data.message || "ファンディングレートデータリセットに失敗しました",
        },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("ファンディングレートデータリセットAPIエラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "ファンディングレートデータリセット中にエラーが発生しました",
      },
      { status: 500 }
    );
  }
}
