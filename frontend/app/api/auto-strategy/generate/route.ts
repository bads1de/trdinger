/**
 * オートストラテジー生成API
 *
 * バックエンドでGA（遺伝的アルゴリズム）を使用した戦略生成を実行し、結果を返します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function POST(request: NextRequest) {
  try {
    console.log("=== Next.js API Route: オートストラテジー生成開始 ===");
    const body = await request.json();
    console.log("Request body:", JSON.stringify(body, null, 2));

    // リクエストボディのバリデーション
    if (!body.experiment_name || !body.base_config || !body.ga_config) {
      return NextResponse.json(
        {
          success: false,
          error:
            "Missing required fields: experiment_name, base_config, or ga_config",
          message: "Invalid request body",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // base_configの必須フィールドをチェック
    const requiredBaseConfigFields = [
      "symbol",
      "timeframe",
      "start_date",
      "end_date",
      "initial_capital",
      "commission_rate",
    ];

    for (const field of requiredBaseConfigFields) {
      if (!body.base_config[field]) {
        return NextResponse.json(
          {
            success: false,
            error: `Missing required base_config field: ${field}`,
            message: "Invalid base_config",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        );
      }
    }

    // ga_configの必須フィールドをチェック
    const requiredGAConfigFields = [
      "population_size",
      "generations",
      "crossover_rate",
      "mutation_rate",
      "elite_size",
    ];

    for (const field of requiredGAConfigFields) {
      if (
        body.ga_config[field] === undefined ||
        body.ga_config[field] === null
      ) {
        return NextResponse.json(
          {
            success: false,
            error: `Missing required ga_config field: ${field}`,
            message: "Invalid ga_config",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        );
      }
    }

    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/auto-strategy/generate`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error(
        `バックエンドAPIエラー: ${response.status} ${response.statusText}`,
        errorText
      );

      // バックエンドからのエラーレスポンスを解析
      let errorMessage = `バックエンドAPIエラー: ${response.status}`;
      try {
        const errorData = JSON.parse(errorText);
        errorMessage = errorData.detail || errorData.message || errorMessage;
      } catch {
        // JSON解析に失敗した場合はデフォルトメッセージを使用
      }

      return NextResponse.json(
        {
          success: false,
          error: errorMessage,
          message: "オートストラテジー生成に失敗しました",
          timestamp: new Date().toISOString(),
        },
        { status: response.status }
      );
    }

    const data = await response.json();

    // 成功レスポンスを返す
    return NextResponse.json({
      success: true,
      experiment_id: data.experiment_id,
      message: data.message || "オートストラテジー生成を開始しました",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("オートストラテジー生成API エラー:", error);

    return NextResponse.json(
      {
        success: false,
        error: "Internal server error",
        message: "サーバー内部エラーが発生しました",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
