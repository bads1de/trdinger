/**
 * 一括OHLCVデータ収集API
 *
 * フロントエンドからの一括OHLCVデータ収集リクエストを受け取り、
 * バックエンドAPIに転送して全データの一括収集を実行するAPIエンドポイントです。
 *
 */

import { NextRequest, NextResponse } from "next/server";
import { BulkOHLCVCollectionResult } from "@/types/strategy";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/data/ohlcv/bulk
 *
 * BTC/USDT:USDT無期限先物の複数時間足でOHLCVデータの一括収集を開始します。
 * 対象時間足: 15m, 30m, 1h, 4h, 1d
 *
 * 注意: BTC/USDT:USDTのみをサポートします。
 */
export async function POST(request: NextRequest) {
  try {
    // バックエンドAPIに転送
    const backendUrl = `${BACKEND_API_URL}/api/data-collection/bulk-historical`;

    try {
      const backendResponse = await fetch(backendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const backendResult = await backendResponse.json();

      if (backendResponse.ok) {
        // バックエンドからの成功レスポンス
        const result: BulkOHLCVCollectionResult = {
          success: true,
          message: backendResult.message || "全データの一括収集を開始しました",
          status: backendResult.status || "started",
          started_at: backendResult.started_at || new Date().toISOString(),
          total_combinations: backendResult.total_combinations,
          actual_tasks: backendResult.actual_tasks,
          skipped_tasks: backendResult.skipped_tasks,
          failed_tasks: backendResult.failed_tasks,
          symbols: backendResult.symbols,
          timeframes: backendResult.timeframes,
          task_details: backendResult.task_details,
          // 後方互換性のため
          total_tasks:
            backendResult.actual_tasks || backendResult.total_tasks || 0,
          completed_tasks: backendResult.completed_tasks,
          successful_tasks: backendResult.successful_tasks,
          task_results: backendResult.task_results,
        };

        return NextResponse.json(result, { status: 200 });
      } else {
        // バックエンドからのエラーレスポンス
        console.error(
          `バックエンドAPIエラー: ${backendResponse.status} - ${JSON.stringify(
            backendResult
          )}`
        );

        const result: BulkOHLCVCollectionResult = {
          success: false,
          message: `バックエンドAPIエラー: ${
            backendResult.detail ||
            backendResult.message ||
            "一括データ収集に失敗しました"
          }`,
          status: "error",
          total_tasks: 0,
          started_at: new Date().toISOString(),
        };

        return NextResponse.json(result, { status: backendResponse.status });
      }
    } catch (networkError) {
      // ネットワークエラー
      console.error(`ネットワークエラー: ${networkError}`);

      const result: BulkOHLCVCollectionResult = {
        success: false,
        message: "ネットワークエラー: バックエンドAPIに接続できませんでした",
        status: "error",
        total_tasks: 0,
        started_at: new Date().toISOString(),
      };

      return NextResponse.json(result, { status: 500 });
    }
  } catch (error) {
    // 予期しないエラー
    console.error(`一括OHLCVデータ収集API予期しないエラー: ${error}`);

    const result: BulkOHLCVCollectionResult = {
      success: false,
      message: "内部サーバーエラーが発生しました",
      status: "error",
      total_tasks: 0,
      started_at: new Date().toISOString(),
    };

    return NextResponse.json(result, { status: 500 });
  }
}
