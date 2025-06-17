/**
 * オートストラテジー実験進捗取得API
 *
 * 指定された実験IDの進捗状況を取得します。
 */

import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL } from "@/constants";

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = await params;

    // 実験IDのバリデーション
    if (!id || id.trim() === "") {
      return NextResponse.json(
        {
          success: false,
          error: "Missing experiment ID",
          message: "実験IDが指定されていません",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // バックエンドAPIを呼び出し
    const response = await fetch(
      `${BACKEND_API_URL}/api/auto-strategy/experiments/${encodeURIComponent(
        id
      )}/progress`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error(
        `バックエンドAPIエラー: ${response.status} ${response.statusText}`,
        errorText
      );

      if (response.status === 404) {
        return NextResponse.json(
          {
            success: false,
            error: "Experiment not found",
            message: "指定された実験が見つかりませんでした",
            timestamp: new Date().toISOString(),
          },
          { status: 404 }
        );
      }

      return NextResponse.json(
        {
          success: false,
          error: `バックエンドAPIエラー: ${response.status}`,
          message: "進捗情報の取得に失敗しました",
          timestamp: new Date().toISOString(),
        },
        { status: response.status }
      );
    }

    const data = await response.json();

    // 成功レスポンスを返す
    return NextResponse.json({
      success: true,
      progress: data.progress || null,
      message: data.message || "進捗情報を取得しました",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("進捗取得API エラー:", error);

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
