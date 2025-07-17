/**
 * 一括差分取得API
 *
 * OHLCV、ファンディングレート、オープンインタレストの
 * 差分データを一括で取得するAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BulkIncrementalUpdateResponse } from "@/types/data-collection";
import { TimeFrame } from "@/types/market-data";
import { BACKEND_API_URL } from "@/constants";

/**
 * 利用可能な時間軸の定義
 */
const VALID_TIMEFRAMES: TimeFrame[] = ["15m", "30m", "1h", "4h", "1d"];

/**
 * POST /api/data/bulk-incremental-update
 *
 * OHLCV、ファンディングレート、オープンインタレストの差分データを一括で取得します。
 * 全時間足（15m, 30m, 1h, 4h, 1d）を自動的に処理します。
 *
 * クエリパラメータ:
 * - symbol: 通貨ペア (オプション、デフォルト: BTC/USDT:USDT)
 */
export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // パラメータの取得
    const symbol = searchParams.get("symbol") || "BTC/USDT:USDT";

    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/data-collection/bulk-incremental-update?symbol=${encodeURIComponent(
      symbol
    )}`;

    try {
      const backendResponse = await fetch(backendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // タイムアウトを設定（60秒）
        signal: AbortSignal.timeout(60000),
      });

      const backendResult = await backendResponse.json();

      if (backendResponse.ok) {
        // バックエンドからの成功レスポンス
        const result: BulkIncrementalUpdateResponse = {
          success: backendResult.success,
          message: backendResult.message || "一括差分データ更新が完了しました",
          data: backendResult.data,
          timestamp: new Date().toISOString(),
        };

        return NextResponse.json(result, { status: 200 });
      } else {
        // バックエンドからのエラーレスポンス
        console.error(
          `バックエンドAPIエラー: ${backendResponse.status} - ${JSON.stringify(
            backendResult
          )}`
        );

        const result: BulkIncrementalUpdateResponse = {
          success: false,
          message: `バックエンドAPIエラー: ${
            backendResult.detail ||
            backendResult.message ||
            "一括差分データ更新に失敗しました"
          }`,
          data: {
            success: false,
            message: "エラーが発生しました",
            data: {
              ohlcv: {
                symbol,
                timeframe: "1h",
                saved_count: 0,
                success: false,
              },
              funding_rate: { symbol, saved_count: 0, success: false },
              open_interest: { symbol, saved_count: 0, success: false },
              external_market: {
                fetched_count: 0,
                inserted_count: 0,
                success: false,
              },
            },
            total_saved_count: 0,
            timestamp: new Date().toISOString(),
          },
          timestamp: new Date().toISOString(),
        };

        return NextResponse.json(result, { status: backendResponse.status });
      }
    } catch (networkError) {
      // ネットワークエラー
      console.error(`ネットワークエラー: ${networkError}`);

      const result: BulkIncrementalUpdateResponse = {
        success: false,
        message: "ネットワークエラー: バックエンドAPIに接続できませんでした",
        data: {
          success: false,
          message: "ネットワークエラー",
          data: {
            ohlcv: { symbol, timeframe: "1h", saved_count: 0, success: false },
            funding_rate: { symbol, saved_count: 0, success: false },
            open_interest: { symbol, saved_count: 0, success: false },
          },
          total_saved_count: 0,
          timestamp: new Date().toISOString(),
        },
        timestamp: new Date().toISOString(),
      };

      return NextResponse.json(result, { status: 500 });
    }
  } catch (error) {
    // 予期しないエラー
    console.error(`一括差分取得API予期しないエラー: ${error}`);

    const result: BulkIncrementalUpdateResponse = {
      success: false,
      message: "内部サーバーエラーが発生しました",
      data: {
        success: false,
        message: "内部サーバーエラー",
        data: {
          ohlcv: {
            symbol: "BTC/USDT:USDT",
            timeframe: "1h",
            saved_count: 0,
            success: false,
          },
          funding_rate: {
            symbol: "BTC/USDT:USDT",
            saved_count: 0,
            success: false,
          },
          open_interest: {
            symbol: "BTC/USDT:USDT",
            saved_count: 0,
            success: false,
          },
        },
        total_saved_count: 0,
        timestamp: new Date().toISOString(),
      },
      timestamp: new Date().toISOString(),
    };

    return NextResponse.json(result, { status: 500 });
  }
}
