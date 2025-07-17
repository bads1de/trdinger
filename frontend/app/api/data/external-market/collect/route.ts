/**
 * 外部市場データ収集API
 *
 * バックエンドAPIを通じて外部市場データを収集するAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * 外部市場データ収集結果の型定義
 */
export interface ExternalMarketCollectionResult {
  /** 成功フラグ */
  success: boolean;
  /** 取得件数 */
  fetched_count: number;
  /** 挿入件数 */
  inserted_count: number;
  /** メッセージ */
  message: string;
  /** エラー（失敗時） */
  error?: string;
}

/**
 * 外部市場データ収集レスポンスの型定義
 */
export interface ExternalMarketCollectionResponse {
  /** 成功フラグ */
  success: boolean;
  /** メッセージ */
  message: string;
  /** データ */
  data: ExternalMarketCollectionResult;
}

/**
 * バックエンドAPIで外部市場データを収集する関数
 *
 * @param symbols 取得するシンボルのリスト
 * @param period 取得期間
 * @returns 収集結果
 */
async function collectExternalMarketData(
  symbols?: string[],
  period: string = "1mo"
): Promise<ExternalMarketCollectionResult> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/external-market/collect", BACKEND_API_URL);
  apiUrl.searchParams.set("period", period);

  if (symbols && symbols.length > 0) {
    symbols.forEach(symbol => {
      apiUrl.searchParams.append("symbols", symbol);
    });
  }

  // バックエンドAPIを呼び出し
  const response = await fetch(apiUrl.toString(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(
      `外部市場データ収集エラー: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(`外部市場データ収集に失敗しました: ${response.status}`);
  }

  const result = await response.json();

  if (!result.success) {
    console.error("外部市場データ収集失敗:", result.message);
    throw new Error(result.message || "外部市場データ収集に失敗しました");
  }

  return result.data;
}

/**
 * POST /api/data/external-market/collect
 *
 * 外部市場データを収集します。
 */
export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const period = searchParams.get("period") || "1mo";
    
    // シンボルのパラメータを取得（複数可能）
    const symbolsParam = searchParams.getAll("symbols");
    const symbols = symbolsParam.length > 0 ? symbolsParam : undefined;

    console.log(
      `外部市場データ収集リクエスト: symbols=${symbols}, period=${period}`
    );

    const result = await collectExternalMarketData(symbols, period);

    const response: ExternalMarketCollectionResponse = {
      success: true,
      message: result.message,
      data: result,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("外部市場データ収集エラー:", error);

    const errorResponse: ExternalMarketCollectionResponse = {
      success: false,
      message: error instanceof Error ? error.message : "外部市場データ収集に失敗しました",
      data: {
        success: false,
        fetched_count: 0,
        inserted_count: 0,
        message: "収集に失敗しました",
        error: error instanceof Error ? error.message : "Unknown error",
      },
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
