/**
 * バックテスト結果削除API
 * 
 * 指定されたIDのバックテスト結果をバックエンドから削除します。
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;

    // IDの検証
    if (!id || isNaN(Number(id))) {
      return NextResponse.json(
        {
          success: false,
          message: '無効なIDです',
        },
        { status: 400 }
      );
    }

    // バックエンドAPIを呼び出し
    const response = await fetch(`${BACKEND_URL}/api/backtest/results/${id}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          {
            success: false,
            message: 'バックテスト結果が見つかりません',
          },
          { status: 404 }
        );
      }
      
      throw new Error(`Backend API error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      message: 'バックテスト結果を削除しました',
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error('Delete backtest result error:', error);
    
    return NextResponse.json(
      {
        success: false,
        message: 'バックテスト結果の削除に失敗しました',
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
