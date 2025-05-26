/**
 * ローソク足データ取得API
 *
 * データベースに保存されたOHLCVデータを取得するAPIエンドポイントです。
 * バックテスト用のデータを提供します。
 *
 * @author Trdinger Development Team
 * @version 3.0.0
 */

import { NextRequest, NextResponse } from "next/server";
import { CandlestickData, TimeFrame } from "@/types/strategy";

/**
 * バックエンドAPIのベースURL
 */
const BACKEND_API_URL = "http://127.0.0.1:8000";

/**
 * 利用可能な時間軸の定義
 */
const VALID_TIMEFRAMES: TimeFrame[] = [
  "1m",
  "5m",
  "15m",
  "30m",
  "1h",
  "4h",
  "1d",
];

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
 * @returns ローソク足データの配列
 */
async function fetchDatabaseOHLCVData(
  symbol: string,
  timeframe: TimeFrame,
  limit: number = 100,
  startDate?: string,
  endDate?: string
): Promise<CandlestickData[]> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/v1/market-data/ohlcv", BACKEND_API_URL);
  apiUrl.searchParams.set("symbol", symbol);
  apiUrl.searchParams.set("timeframe", timeframe);
  apiUrl.searchParams.set("limit", limit.toString());

  if (startDate) {
    apiUrl.searchParams.set("start_date", startDate);
  }
  if (endDate) {
    apiUrl.searchParams.set("end_date", endDate);
  }

  console.log(`データベースAPI呼び出し: ${apiUrl.toString()}`);

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

  // バックエンドのOHLCVデータをフロントエンド形式に変換
  const ohlcvData = backendData.data;
  const candlesticks: CandlestickData[] = ohlcvData.map((candle: number[]) => {
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

  console.log(`取得したデータ件数: ${candlesticks.length}`);
  return candlesticks;
}

/**
 * GET /api/data/candlesticks
 *
 * ローソク足データを取得します。
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

    if (!VALID_TIMEFRAMES.includes(timeframe)) {
      return NextResponse.json(
        {
          success: false,
          message: `サポートされていない時間軸です。利用可能: ${VALID_TIMEFRAMES.join(
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
    const candlesticks = await fetchDatabaseOHLCVData(
      symbol,
      timeframe,
      limit,
      startDate || undefined,
      endDate || undefined
    );

    // データが取得できない場合のエラーハンドリング
    if (!candlesticks || candlesticks.length === 0) {
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
        candlesticks: candlesticks,
      },
      message: `${symbol} の ${timeframe} ローソク足データを取得しました（${candlesticks.length}件）`,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("ローソク足データ取得エラー:", error);

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
