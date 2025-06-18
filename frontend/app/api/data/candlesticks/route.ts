/**
 * OHLCVデータ取得API
 *
 * データベースに保存されたOHLCVデータを取得するAPIエンドポイントです。
 * データテーブル表示用のデータを提供します。
 *
 */

import { NextRequest, NextResponse } from "next/server";
import { PriceData, TimeFrame } from "@/types/strategy";
import { BACKEND_API_URL } from "@/constants";

/**
 * 利用可能な時間軸の定義
 */
const VALID_TIMEFRAMES: TimeFrame[] = ["15m", "30m", "1h", "4h", "1d"];

/**
 * バックエンドAPIからOHLCVデータを取得する関数
 *
 * データベースに保存されたOHLCVデータを取得します。
 *
 * @param symbol 通貨ペア
 * @param timeframe 時間軸
 * @param limit データ件数
 * @param startDate 開始日時
 * @param endDate 終了日時
 * @returns OHLCVデータの配列
 */
async function fetchDatabaseOHLCVData(
  symbol: string,
  timeframe: TimeFrame,
  limit: number = 100,
  startDate?: string,
  endDate?: string
): Promise<PriceData[]> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/market-data/ohlcv", BACKEND_API_URL);
  apiUrl.searchParams.set("symbol", symbol);
  apiUrl.searchParams.set("timeframe", timeframe);
  apiUrl.searchParams.set("limit", limit.toString());

  if (startDate) {
    apiUrl.searchParams.set("start_date", startDate);
  }
  if (endDate) {
    apiUrl.searchParams.set("end_date", endDate);
  }

  // バックエンドAPIを呼び出し
  const response = await fetch(apiUrl.toString(), {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    // タイムアウトを設定（30秒）
    signal: AbortSignal.timeout(30000),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      `データベースAPIエラー: ${response.status} ${response.statusText} - ${
        errorData.detail?.message || errorData.message || "Unknown error"
      }`
    );
  }

  const backendData = await response.json();

  if (!backendData.success) {
    throw new Error(
      backendData.message || "データベースAPIからエラーレスポンス"
    );
  }

  // デバッグ用ログ出力
  console.log(
    "バックエンドレスポンス構造:",
    JSON.stringify(backendData, null, 2)
  );

  // バックエンドのOHLCVデータをフロントエンド形式に変換
  const ohlcvData = backendData.data.ohlcv_data;

  // データが配列であることを確認
  if (!Array.isArray(ohlcvData)) {
    console.error("OHLCVデータの型:", typeof ohlcvData);
    console.error("OHLCVデータの内容:", ohlcvData);
    throw new Error("バックエンドから返されたOHLCVデータが配列ではありません");
  }

  const priceData: PriceData[] = ohlcvData.map((candle: number[]) => {
    const [timestamp, open, high, low, close, volume] = candle;

    return {
      timestamp: new Date(timestamp).toISOString(),
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Number(volume.toFixed(2)),
    };
  });

  return priceData;
}

/**
 * GET /api/data/candlesticks
 *
 * OHLCVデータを取得します。
 *
 * クエリパラメータ:
 * - symbol: 通貨ペア (必須)
 * - timeframe: 時間軸 (必須)
 * - limit: 取得件数 (オプション、デフォルト: 100)
 * - start_date: 開始日時 (オプション)
 * - end_date: 終了日時 (オプション)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // パラメータの取得
    const symbol = searchParams.get("symbol");
    const timeframe = searchParams.get("timeframe") as TimeFrame;
    const limit = parseInt(searchParams.get("limit") || "100");
    const startDate = searchParams.get("start_date");
    const endDate = searchParams.get("end_date");

    // バリデーション
    if (!symbol) {
      return NextResponse.json(
        {
          success: false,
          message: "symbol パラメータは必須です",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // シンボルバリデーション (簡易版)
    // 本来はサポートされているシンボルリストと照合すべきだが、今回は削除されたため簡易化
    if (typeof symbol !== "string" || symbol.trim() === "") {
      return NextResponse.json(
        {
          success: false,
          message: "無効なシンボルです",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    if (!timeframe) {
      return NextResponse.json(
        {
          success: false,
          message: "timeframe パラメータは必須です",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // 時間足バリデーション (簡易版)
    if (!VALID_TIMEFRAMES.includes(timeframe)) {
      return NextResponse.json(
        {
          success: false,
          message: `サポートされていない時間足です: ${timeframe}. サポートされている時間足: ${VALID_TIMEFRAMES.join(
            ", "
          )}`,
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    if (limit < 1 || limit > 1000) {
      return NextResponse.json(
        {
          success: false,
          message: "limit は 1 から 1000 の間で指定してください",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // データベースからデータを取得
    const ohlcvData = await fetchDatabaseOHLCVData(
      symbol,
      timeframe,
      limit,
      startDate || undefined,
      endDate || undefined
    );

    // データが取得できない場合のエラーハンドリング
    if (!ohlcvData || ohlcvData.length === 0) {
      return NextResponse.json(
        {
          success: false,
          message: "データが見つかりません。データ収集を実行してください。",
          timestamp: new Date().toISOString(),
        },
        { status: 404 }
      );
    }

    // レスポンスの返却
    return NextResponse.json({
      success: true,
      data: {
        symbol,
        timeframe,
        ohlcv: ohlcvData,
      },
      message: `${symbol} の ${timeframe} OHLCVデータを取得しました（${ohlcvData.length}件）`,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("OHLCVデータ取得エラー:", error);

    return NextResponse.json(
      {
        success: false,
        message: "サーバー内部エラーが発生しました",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
