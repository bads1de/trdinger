/**
 * FRデータ取得API
 *
 * データベースに保存されたFRデータを取得するAPIエンドポイントです。
 *
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";
import { FundingRateData, FundingRateResponse } from "@/types/funding-rate";

/**
 * バックエンドAPIからFRデータを取得する関数
 *
 * データベースに保存されたFRデータを取得します。
 *
 * @param symbol 通貨ペア
 * @param limit データ件数
 * @param startDate 開始日時
 * @param endDate 終了日時
 * @returns FRデータの配列
 */
async function fetchDatabaseFundingRateData(
  symbol: string,
  limit: number = 100,
  startDate?: string,
  endDate?: string
): Promise<FundingRateData[]> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/funding-rates", BACKEND_API_URL);
  apiUrl.searchParams.set("symbol", symbol);
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
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`バックエンドAPIエラー: ${response.status} - ${errorText}`);
    throw new Error(
      `バックエンドAPIエラー: ${response.status} - ${response.statusText}`
    );
  }

  const backendData = await response.json();

  if (!backendData.success) {
    throw new Error(backendData.message || "データの取得に失敗しました");
  }

  // バックエンドのFRデータをフロントエンド形式に変換
  const fundingRateData = backendData.data.funding_rates;
  const fundingRates: FundingRateData[] = fundingRateData.map((rate: any) => ({
    symbol: rate.symbol,
    funding_rate: Number(rate.funding_rate),
    funding_timestamp: rate.funding_timestamp,
    timestamp: rate.timestamp,
    next_funding_timestamp: rate.next_funding_timestamp || null,
    mark_price: rate.mark_price ? Number(rate.mark_price) : null,
    index_price: rate.index_price ? Number(rate.index_price) : null,
  }));

  return fundingRates;
}

/**
 * GET /api/data/funding-rates
 *
 * FRデータを取得します。
 */
export async function GET(request: NextRequest) {
  try {
    // URLパラメータを取得
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol") || "BTC/USDT";
    const timeframe = searchParams.get("timeframe") || "1d";
    const limitParam = searchParams.get("limit");
    const startDate = searchParams.get("start_date");
    const endDate = searchParams.get("end_date");

    // パラメータの検証
    const limit = limitParam ? parseInt(limitParam, 10) : 100;
    if (isNaN(limit) || limit < 1 || limit > 1000) {
      return NextResponse.json(
        {
          success: false,
          message: "limitパラメータは1から1000の間で指定してください",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // データベースからFRデータを取得
    const fundingRates = await fetchDatabaseFundingRateData(
      symbol,
      limit,
      startDate || undefined,
      endDate || undefined
    );

    // データが取得できない場合のエラーハンドリング
    if (!fundingRates || fundingRates.length === 0) {
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
    const response: FundingRateResponse = {
      success: true,
      data: {
        symbol,
        count: fundingRates.length,
        funding_rates: fundingRates,
      },
      message: `${symbol} のFRデータを取得しました（${fundingRates.length}件）`,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("FRデータ取得エラー:", error);

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
