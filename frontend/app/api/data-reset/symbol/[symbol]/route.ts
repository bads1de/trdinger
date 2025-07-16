/**
 * シンボル別データリセットAPIルート
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  try {
    const { symbol } = await params;

    if (!symbol) {
      return NextResponse.json(
        { success: false, message: "シンボルが指定されていません" },
        { status: 400 }
      );
    }

    const response = await fetch(
      `${BACKEND_API_URL}/api/data-reset/symbol/${encodeURIComponent(symbol)}`,
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
            data.message ||
            `シンボル「${symbol}」のデータリセットに失敗しました`,
        },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("シンボル別データリセットAPIエラー:", error);
    return NextResponse.json(
      {
        success: false,
        message: "シンボル別データリセット中にエラーが発生しました",
      },
      { status: 500 }
    );
  }
}
