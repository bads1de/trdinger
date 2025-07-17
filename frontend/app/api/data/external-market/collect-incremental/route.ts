/**
 * 外部市場データ差分収集API
 *
 * バックエンドAPIを通じて外部市場データの差分収集を行うAPIエンドポイントです。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

/**
 * 外部市場データ差分収集結果の型定義
 */
export interface ExternalMarketIncrementalCollectionResult {
  /** 成功フラグ */
  success: boolean;
  /** 取得件数 */
  fetched_count: number;
  /** 挿入件数 */
  inserted_count: number;
  /** メッセージ */
  message: string;
  /** 収集タイプ */
  collection_type: string;
  /** 収集前の最新タイムスタンプ */
  latest_timestamp_before?: string;
  /** エラー（失敗時） */
  error?: string;
}

/**
 * 外部市場データ差分収集レスポンスの型定義
 */
export interface ExternalMarketIncrementalCollectionResponse {
  /** 成功フラグ */
  success: boolean;
  /** メッセージ */
  message: string;
  /** データ */
  data: ExternalMarketIncrementalCollectionResult;
}

/**
 * バックエンドAPIで外部市場データの差分収集を行う関数
 *
 * @param symbols 取得するシンボルのリスト
 * @returns 差分収集結果
 */
async function collectIncrementalExternalMarketData(
  symbols?: string[]
): Promise<ExternalMarketIncrementalCollectionResult> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/external-market/collect-incremental", BACKEND_API_URL);

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
      `外部市場データ差分収集エラー: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(`外部市場データ差分収集に失敗しました: ${response.status}`);
  }

  const result = await response.json();

  if (!result.success) {
    console.error("外部市場データ差分収集失敗:", result.message);
    throw new Error(result.message || "外部市場データ差分収集に失敗しました");
  }

  return result.data;
}

/**
 * POST /api/data/external-market/collect-incremental
 *
 * 外部市場データの差分収集を行います。
 */
export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    
    // シンボルのパラメータを取得（複数可能）
    const symbolsParam = searchParams.getAll("symbols");
    const symbols = symbolsParam.length > 0 ? symbolsParam : undefined;

    console.log(
      `外部市場データ差分収集リクエスト: symbols=${symbols}`
    );

    const result = await collectIncrementalExternalMarketData(symbols);

    const response: ExternalMarketIncrementalCollectionResponse = {
      success: true,
      message: result.message,
      data: result,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("外部市場データ差分収集エラー:", error);

    const errorResponse: ExternalMarketIncrementalCollectionResponse = {
      success: false,
      message: error instanceof Error ? error.message : "外部市場データ差分収集に失敗しました",
      data: {
        success: false,
        fetched_count: 0,
        inserted_count: 0,
        message: "差分収集に失敗しました",
        collection_type: "incremental",
        error: error instanceof Error ? error.message : "Unknown error",
      },
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
