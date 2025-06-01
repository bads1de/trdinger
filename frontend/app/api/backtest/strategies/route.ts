/**
 * バックテスト戦略一覧取得API
 * 
 * バックエンドから利用可能な戦略一覧を取得します。
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/backtest/strategies`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Backend API error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      strategies: data.strategies || {},
      message: 'Strategies retrieved successfully',
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error('Error fetching strategies:', error);
    
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        message: 'Failed to fetch strategies',
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
