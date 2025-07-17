import { NextRequest, NextResponse } from "next/server";

/**
 * バックエンドAPIのベースURL
 */
const BACKEND_API_URL = process.env.BACKEND_API_URL || "http://localhost:8000";

/**
 * 外部市場データ履歴収集結果の型定義
 */
interface ExternalMarketHistoricalCollectionResult {
  success: boolean;
  message: string;
  data: {
    fetched_count: number;
    inserted_count: number;
    collection_type: string;
    existing_data_range?: {
      oldest: string;
      newest: string;
    };
    start_date?: string;
    end_date?: string;
  };
}

/**
 * バックエンドAPIで外部市場データの履歴データを収集する関数
 *
 * @param symbols 取得するシンボルのリスト
 * @param period 取得期間
 * @param startDate 開始日（YYYY-MM-DD形式）
 * @param endDate 終了日（YYYY-MM-DD形式）
 * @returns 履歴収集結果
 */
async function collectHistoricalExternalMarketData(
  symbols?: string[],
  period: string = "5y",
  startDate?: string,
  endDate?: string
): Promise<ExternalMarketHistoricalCollectionResult> {
  // バックエンドAPIのURLを構築
  const apiUrl = new URL("/api/external-market/collect-historical", BACKEND_API_URL);
  apiUrl.searchParams.set("period", period);

  if (symbols && symbols.length > 0) {
    symbols.forEach(symbol => {
      apiUrl.searchParams.append("symbols", symbol);
    });
  }

  if (startDate) {
    apiUrl.searchParams.set("start_date", startDate);
  }

  if (endDate) {
    apiUrl.searchParams.set("end_date", endDate);
  }

  // バックエンドAPIを呼び出し
  const response = await fetch(apiUrl.toString(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || 
      errorData.message || 
      `HTTP ${response.status}: 外部市場データの履歴収集に失敗しました`
    );
  }

  return await response.json();
}

/**
 * POST /api/data/external-market/collect-historical
 * 外部市場データの履歴データを収集
 */
export async function POST(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const symbols = url.searchParams.getAll("symbols");
    const period = url.searchParams.get("period") || "5y";
    const startDate = url.searchParams.get("start_date") || undefined;
    const endDate = url.searchParams.get("end_date") || undefined;

    const result = await collectHistoricalExternalMarketData(
      symbols.length > 0 ? symbols : undefined,
      period,
      startDate,
      endDate
    );

    return NextResponse.json(result);
  } catch (error) {
    console.error("外部市場データ履歴収集エラー:", error);
    
    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : "外部市場データの履歴収集中に予期しないエラーが発生しました",
        data: null,
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/data/external-market/collect-historical
 * 外部市場データの履歴収集機能の情報を取得
 */
export async function GET() {
  return NextResponse.json({
    success: true,
    message: "外部市場データ履歴収集API",
    data: {
      description: "外部市場データ（S&P 500、NASDAQ、DXY、VIX）の履歴データを収集します",
      supported_symbols: ["^GSPC", "^IXIC", "DX-Y.NYB", "^VIX"],
      supported_periods: ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
      date_format: "YYYY-MM-DD",
      note: "start_dateとend_dateを指定した場合、periodより優先されます",
    },
  });
}
