/**
 * 月次リターンヒートマップコンポーネントのテスト
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import MonthlyReturnsHeatmap from '@/components/backtest/charts/MonthlyReturnsHeatmap';
import { generateMockEquityCurve } from '../../utils/chartTestUtils';

describe('MonthlyReturnsHeatmap', () => {
  const mockEquityCurve = generateMockEquityCurve(365); // 1年分のデータ

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('正常なデータでヒートマップをレンダリングする', () => {
    render(
      <MonthlyReturnsHeatmap
        data={mockEquityCurve}
        title="月次リターンヒートマップ"
      />
    );

    // ヒートマップコンテナが表示される
    expect(screen.getByTestId('heatmap-container')).toBeInTheDocument();
    
    // 月のヘッダーが表示される
    expect(screen.getByText('Jan')).toBeInTheDocument();
    expect(screen.getByText('Dec')).toBeInTheDocument();
    
    // 年のラベルが表示される
    expect(screen.getByText('2024')).toBeInTheDocument();
  });

  it('複数年のデータを正しく表示する', () => {
    const multiYearData = [
      ...generateMockEquityCurve(365).map(point => ({
        ...point,
        timestamp: point.timestamp.replace('2024', '2023')
      })),
      ...generateMockEquityCurve(365)
    ];

    render(
      <MonthlyReturnsHeatmap
        data={multiYearData}
        title="月次リターンヒートマップ"
      />
    );

    // 複数年のラベルが表示される
    expect(screen.getByText('2023')).toBeInTheDocument();
    expect(screen.getByText('2024')).toBeInTheDocument();
  });

  it('空のデータでエラー状態を表示する', () => {
    render(
      <MonthlyReturnsHeatmap
        data={[]}
        title="月次リターンヒートマップ"
      />
    );

    // エラー状態が表示される
    expect(screen.getByText('データがありません')).toBeInTheDocument();
  });

  it('ローディング状態を表示する', () => {
    render(
      <MonthlyReturnsHeatmap
        data={mockEquityCurve}
        loading={true}
        title="月次リターンヒートマップ"
      />
    );

    // ローディング状態が表示される
    expect(screen.getByText('チャートを読み込み中...')).toBeInTheDocument();
  });

  it('エラー状態を表示する', () => {
    const errorMessage = 'ヒートマップの生成に失敗しました';
    
    render(
      <MonthlyReturnsHeatmap
        data={mockEquityCurve}
        error={errorMessage}
        title="月次リターンヒートマップ"
      />
    );

    // エラー状態が表示される
    expect(screen.getByText('エラーが発生しました')).toBeInTheDocument();
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('カスタムの高さとクラス名を適用する', () => {
    const customHeight = 300;
    const customClassName = 'custom-heatmap';
    
    render(
      <MonthlyReturnsHeatmap
        data={mockEquityCurve}
        height={customHeight}
        className={customClassName}
        title="月次リターンヒートマップ"
      />
    );

    // カスタムクラスが適用される
    const container = screen.getByTestId('heatmap-container').closest('.bg-gray-800\\/30');
    expect(container).toHaveClass(customClassName);
  });

  it('サブタイトルを表示する', () => {
    const subtitle = '月別パフォーマンスの季節性分析';
    
    render(
      <MonthlyReturnsHeatmap
        data={mockEquityCurve}
        title="月次リターンヒートマップ"
        subtitle={subtitle}
      />
    );

    expect(screen.getByText(subtitle)).toBeInTheDocument();
  });

  it('アクションボタンを表示する', () => {
    const actions = (
      <button data-testid="export-heatmap">エクスポート</button>
    );
    
    render(
      <MonthlyReturnsHeatmap
        data={mockEquityCurve}
        title="月次リターンヒートマップ"
        actions={actions}
      />
    );

    expect(screen.getByTestId('export-heatmap')).toBeInTheDocument();
  });

  it('カラースケールの凡例を表示する', () => {
    render(
      <MonthlyReturnsHeatmap
        data={mockEquityCurve}
        showLegend={true}
        title="月次リターンヒートマップ"
      />
    );

    // 凡例が表示される
    expect(screen.getByTestId('color-legend')).toBeInTheDocument();
  });

  it('統計情報を表示する', () => {
    render(
      <MonthlyReturnsHeatmap
        data={mockEquityCurve}
        showStatistics={true}
        title="月次リターンヒートマップ"
      />
    );

    // 統計情報が表示される
    expect(screen.getByTestId('monthly-statistics')).toBeInTheDocument();
  });

  it('単一月のデータでも正しく表示する', () => {
    const singleMonthData = mockEquityCurve.slice(0, 30); // 1ヶ月分
    
    render(
      <MonthlyReturnsHeatmap
        data={singleMonthData}
        title="月次リターンヒートマップ"
      />
    );

    // ヒートマップが正常にレンダリングされる
    expect(screen.getByTestId('heatmap-container')).toBeInTheDocument();
  });

  it('年の境界をまたぐデータを正しく処理する', () => {
    const crossYearData = [
      { timestamp: '2023-12-15T00:00:00Z', equity: 100000 },
      { timestamp: '2023-12-31T00:00:00Z', equity: 105000 },
      { timestamp: '2024-01-01T00:00:00Z', equity: 106000 },
      { timestamp: '2024-01-15T00:00:00Z', equity: 110000 },
    ];

    render(
      <MonthlyReturnsHeatmap
        data={crossYearData}
        title="月次リターンヒートマップ"
      />
    );

    // 両年のデータが表示される
    expect(screen.getByText('2023')).toBeInTheDocument();
    expect(screen.getByText('2024')).toBeInTheDocument();
  });
});
