/**
 * OHLCVデータ収集API
 *
 * フロントエンドからのOHLCVデータ収集リクエストを受け取り、
 * バックエンドAPIに転送してデータ収集を実行するAPIエンドポイントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { NextRequest, NextResponse } from "next/server";
import {
  OHLCVCollectionResult,
  OHLCVCollectionRequest,
} from "@/types/strategy";
import { BACKEND_API_URL } from "@/constants";

/**
 * POST /api/data/ohlcv
 *
 * OHLCVデータの収集を開始します。
 *
 * リクエストボディ:
 * - symbol: 取引ペアシンボル (必須)
 * - timeframe: 時間軸 (必須)
 */
export async function POST(request: NextRequest) {
  try {
    // リクエストボディの解析
    let requestData: OHLCVCollectionRequest;

    try {
      requestData = await request.json();
    } catch (error) {
      return NextResponse.json(
        {
          success: false,
          message: "無効なJSONリクエストです",
          timestamp: new Date().toISOString(),
        } as OHLCVCollectionResult,
        { status: 400 }
      );
    }

    // バリデーション
    if (!requestData.symbol) {
      return NextResponse.json(
        {
          success: false,
          message: "symbol パラメータは必須です",
          timestamp: new Date().toISOString(),
        } as OHLCVCollectionResult,
        { status: 400 }
      );
    }

    if (!requestData.timeframe) {
      return NextResponse.json(
        {
          success: false,
          message: "timeframe パラメータは必須です",
          timestamp: new Date().toISOString(),
        } as OHLCVCollectionResult,
        { status: 400 }
      );
    }

    console.log(
      `OHLCVデータ収集リクエスト: ${requestData.symbol} ${requestData.timeframe}`
    );

    // バックエンドAPIに転送（クエリパラメータとして送信）
    const backendUrl = `${BACKEND_API_URL}/api/data-collection/historical?symbol=${encodeURIComponent(
      requestData.symbol
    )}&timeframe=${encodeURIComponent(requestData.timeframe)}`;

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
        const result: OHLCVCollectionResult = {
          success: true,
          message:
            backendResult.message ||
            `${requestData.symbol} ${requestData.timeframe} のデータ収集を開始しました`,
          status: backendResult.status || "started",
          saved_count: backendResult.saved_count,
          skipped_count: backendResult.skipped_count,
        };

        console.log(
          `OHLCVデータ収集成功: ${requestData.symbol} ${requestData.timeframe}`
        );

        return NextResponse.json(result, { status: 200 });
      } else {
        // バックエンドからのエラーレスポンス
        console.error(
          `バックエンドAPIエラー: ${backendResponse.status} - ${JSON.stringify(
            backendResult
          )}`
        );

        const result: OHLCVCollectionResult = {
          success: false,
          message: `バックエンドAPIエラー: ${
            backendResult.detail ||
            backendResult.message ||
            "データ収集に失敗しました"
          }`,
          status: "error",
        };

        return NextResponse.json(result, { status: backendResponse.status });
      }
    } catch (networkError) {
      // ネットワークエラー
      console.error(`ネットワークエラー: ${networkError}`);

      const result: OHLCVCollectionResult = {
        success: false,
        message: "ネットワークエラー: バックエンドAPIに接続できませんでした",
        status: "error",
      };

      return NextResponse.json(result, { status: 500 });
    }
  } catch (error) {
    // 予期しないエラー
    console.error(`OHLCVデータ収集API予期しないエラー: ${error}`);

    const result: OHLCVCollectionResult = {
      success: false,
      message: "内部サーバーエラーが発生しました",
      status: "error",
    };

    return NextResponse.json(result, { status: 500 });
  }
}
