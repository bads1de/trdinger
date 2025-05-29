/**
 * FRデータ収集API
 *
 * フロントエンドからのFRデータ収集リクエストを受け取り、
 * バックエンドAPIに転送してFRデータの収集を実行するAPIエンドポイントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { NextRequest, NextResponse } from "next/server";
import { FundingRateCollectionResponse } from "@/types/strategy";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/data/funding-rates/collect
 *
 * 指定されたシンボルのFRデータを収集します。
 */
export async function POST(request: NextRequest) {
  try {
    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || "BTC/USDT";
    const limit = searchParams.get("limit") || "100";
    const fetchAll = searchParams.get("fetch_all") === "true";

    // バックエンドAPIに転送
    let backendUrl = `${BACKEND_API_URL}/api/funding-rates/collect?symbol=${encodeURIComponent(
      symbol
    )}&limit=${limit}`;
    if (fetchAll) {
      backendUrl += "&fetch_all=true";
    }

    try {
      const backendResponse = await fetch(backendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // タイムアウトを設定（全期間取得の場合は5分、通常は30秒）
        signal: AbortSignal.timeout(fetchAll ? 300000 : 30000),
      });

      if (!backendResponse.ok) {
        const errorData = await backendResponse.json().catch(() => ({}));
        throw new Error(
          `バックエンドAPIエラー: ${backendResponse.status} ${
            backendResponse.statusText
          } - ${errorData.detail || errorData.message || "Unknown error"}`
        );
      }

      const backendData = await backendResponse.json();

      if (!backendData.success) {
        throw new Error(
          backendData.message || "バックエンドAPIからエラーレスポンス"
        );
      }

      // 成功レスポンス
      const response: FundingRateCollectionResponse = {
        success: true,
        data: backendData.data,
        message: backendData.message,
      };

      return NextResponse.json(response);
    } catch (fetchError) {
      console.error("バックエンドAPI呼び出しエラー:", fetchError);

      // バックエンドAPIエラーの詳細なハンドリング
      if (fetchError instanceof Error) {
        if (fetchError.name === "TimeoutError") {
          return NextResponse.json(
            {
              success: false,
              message:
                "FRデータ収集がタイムアウトしました。しばらく待ってから再試行してください。",
              error: "TIMEOUT_ERROR",
            },
            { status: 408 }
          );
        }

        if (fetchError.message.includes("ECONNREFUSED")) {
          return NextResponse.json(
            {
              success: false,
              message:
                "バックエンドサーバーに接続できません。サーバーが起動しているか確認してください。",
              error: "CONNECTION_ERROR",
            },
            { status: 503 }
          );
        }
      }

      return NextResponse.json(
        {
          success: false,
          message: `FRデータ収集中にエラーが発生しました: ${
            fetchError instanceof Error ? fetchError.message : "Unknown error"
          }`,
          error: "BACKEND_API_ERROR",
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error("FRデータ収集API内部エラー:", error);

    return NextResponse.json(
      {
        success: false,
        message: `内部エラーが発生しました: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        error: "INTERNAL_ERROR",
      },
      { status: 500 }
    );
  }
}
