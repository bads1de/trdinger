/**
 * ファンディングレートデータ収集API
 *
 * フロントエンドからのファンディングレートデータ収集リクエストを受け取り、
 * バックエンドAPIに転送してファンディングレートデータの収集を実行するAPIエンドポイントです。
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
 * 指定されたシンボルのファンディングレートデータを収集します。
 */
export async function POST(request: NextRequest) {
  try {
    console.log("ファンディングレートデータ収集リクエスト開始");

    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTC/USDT';
    const limit = searchParams.get('limit') || '100';

    console.log(`収集対象: ${symbol}, 件数: ${limit}`);

    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/funding-rates/collect?symbol=${encodeURIComponent(symbol)}&limit=${limit}`;

    try {
      const backendResponse = await fetch(backendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // タイムアウトを設定（30秒）
        signal: AbortSignal.timeout(30000),
      });

      if (!backendResponse.ok) {
        const errorData = await backendResponse.json().catch(() => ({}));
        throw new Error(
          `バックエンドAPIエラー: ${backendResponse.status} ${backendResponse.statusText} - ${
            errorData.detail || errorData.message || "Unknown error"
          }`
        );
      }

      const backendData = await backendResponse.json();

      if (!backendData.success) {
        throw new Error(
          backendData.message || "バックエンドAPIからエラーレスポンス"
        );
      }

      console.log("ファンディングレートデータ収集成功:", backendData);

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
        if (fetchError.name === 'TimeoutError') {
          return NextResponse.json(
            {
              success: false,
              message: "ファンディングレートデータ収集がタイムアウトしました。しばらく待ってから再試行してください。",
              error: "TIMEOUT_ERROR",
            },
            { status: 408 }
          );
        }

        if (fetchError.message.includes('ECONNREFUSED')) {
          return NextResponse.json(
            {
              success: false,
              message: "バックエンドサーバーに接続できません。サーバーが起動しているか確認してください。",
              error: "CONNECTION_ERROR",
            },
            { status: 503 }
          );
        }
      }

      return NextResponse.json(
        {
          success: false,
          message: `ファンディングレートデータ収集中にエラーが発生しました: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}`,
          error: "BACKEND_API_ERROR",
        },
        { status: 500 }
      );
    }

  } catch (error) {
    console.error("ファンディングレートデータ収集API内部エラー:", error);

    return NextResponse.json(
      {
        success: false,
        message: `内部エラーが発生しました: ${error instanceof Error ? error.message : 'Unknown error'}`,
        error: "INTERNAL_ERROR",
      },
      { status: 500 }
    );
  }
}
