/**
 * OHLCVデータ収集API
 *
 * フロントエンドからのOHLCVデータ収集リクエストを受け取り、
 * バックエンドAPIに転送してOHLCVデータの収集を実行するAPIエンドポイントです。
 *
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";
import { 
  validateSymbol, 
  validateTimeframe,
  createSymbolValidationError,
  createTimeframeValidationError 
} from "@/lib/validation";

/**
 * POST /api/data/ohlcv/collect
 *
 * 指定されたシンボルと時間足のOHLCVデータを収集します。
 */
export async function POST(request: NextRequest) {
  try {
    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || "BTC/USDT:USDT";
    const timeframe = searchParams.get("timeframe") || "1h";
    const limit = searchParams.get("limit") || "100";
    const daysBack = searchParams.get("days_back") || "30";

    // シンボルバリデーション
    if (!validateSymbol(symbol)) {
      return NextResponse.json(
        createSymbolValidationError(symbol),
        { status: 400 }
      );
    }

    // 時間足バリデーション
    if (!validateTimeframe(timeframe)) {
      return NextResponse.json(
        createTimeframeValidationError(timeframe),
        { status: 400 }
      );
    }

    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/data-collection/ohlcv/collect?symbol=${encodeURIComponent(
      symbol
    )}&timeframe=${timeframe}&limit=${limit}&days_back=${daysBack}`;

    try {
      const backendResponse = await fetch(backendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // タイムアウトを設定（2分）
        signal: AbortSignal.timeout(120000),
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
      return NextResponse.json({
        success: true,
        data: backendData.data,
        message: backendData.message || `${symbol} ${timeframe} OHLCVデータの収集が完了しました`,
        timestamp: new Date().toISOString(),
      });

    } catch (fetchError) {
      console.error("バックエンドAPI呼び出しエラー:", fetchError);

      // バックエンドAPIエラーの詳細なハンドリング
      if (fetchError instanceof Error) {
        if (fetchError.name === "TimeoutError") {
          return NextResponse.json(
            {
              success: false,
              message:
                "OHLCVデータ収集がタイムアウトしました。しばらく待ってから再試行してください。",
              error: "TIMEOUT_ERROR",
              timestamp: new Date().toISOString(),
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
              timestamp: new Date().toISOString(),
            },
            { status: 503 }
          );
        }
      }

      return NextResponse.json(
        {
          success: false,
          message: `OHLCVデータ収集中にエラーが発生しました: ${
            fetchError instanceof Error ? fetchError.message : "Unknown error"
          }`,
          error: "BACKEND_API_ERROR",
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error("OHLCVデータ収集API内部エラー:", error);

    return NextResponse.json(
      {
        success: false,
        message: `内部エラーが発生しました: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        error: "INTERNAL_ERROR",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
