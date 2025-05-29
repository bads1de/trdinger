/**
 * 一括FRデータ収集API
 *
 * フロントエンドからの一括FRデータ収集リクエストを受け取り、
 * バックエンドAPIに転送して全データの一括収集を実行するAPIエンドポイントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { NextRequest, NextResponse } from "next/server";
import { BulkFundingRateCollectionResult } from "@/types/strategy";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/data/funding-rates/bulk
 *
 * 全ての主要シンボルでFRデータの一括収集を開始します。
 */
export async function POST(request: NextRequest) {
  try {
    console.log("一括FRデータ収集リクエスト開始");

    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get("limit") || "100";

    console.log(`一括収集開始: 各シンボルあたり${limit}件`);

    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/funding-rates/bulk-collect?limit=${limit}`;

    try {
      const backendResponse = await fetch(backendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // タイムアウトを設定（5分）
        signal: AbortSignal.timeout(300000),
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

      console.log("一括FRデータ収集成功:", backendData);

      // 成功レスポンス
      const response: {
        success: boolean;
        data: BulkFundingRateCollectionResult;
        message?: string;
      } = {
        success: true,
        data: {
          success: true,
          message: backendData.message,
          started_at: new Date().toISOString(),
          status: "completed",
          total_symbols: backendData.data.total_symbols,
          successful_symbols: backendData.data.successful_symbols,
          failed_symbols: backendData.data.failed_symbols,
          total_saved_records: backendData.data.total_saved_records,
          results: backendData.data.results,
          failures: backendData.data.failures,
        },
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
                "一括FRデータ収集がタイムアウトしました。処理が長時間実行されている可能性があります。",
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
          message: `一括FRデータ収集中にエラーが発生しました: ${
            fetchError instanceof Error ? fetchError.message : "Unknown error"
          }`,
          error: "BACKEND_API_ERROR",
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error("一括FRデータ収集API内部エラー:", error);

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
