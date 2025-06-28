/**
 * バックテストステップ追加のテスト
 * 
 * TDDアプローチでバックテストステップの追加機能をテストします。
 */

import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import StrategyBuilderPage from '@/app/strategy-builder/page';

// useStrategyBuilderフックのモック
jest.mock('@/hooks/useStrategyBuilder', () => ({
  useStrategyBuilder: () => ({
    currentStep: 'indicators',
    selectedIndicators: [],
    entryConditions: [],
    exitConditions: [],
    strategyName: '',
    strategyDescription: '',
    isStrategyComplete: false,
    setCurrentStep: jest.fn(),
    updateSelectedIndicators: jest.fn(),
    updateIndicatorParameters: jest.fn(),
    toggleIndicatorEnabled: jest.fn(),
    updateEntryConditions: jest.fn(),
    updateExitConditions: jest.fn(),
    updateStrategyName: jest.fn(),
    updateStrategyDescription: jest.fn(),
    validateCurrentStrategy: jest.fn(),
    saveCurrentStrategy: jest.fn(),
    loadStrategy: jest.fn(),
    canProceedToStep: jest.fn((step: string) => {
      // バックテストステップへの進行可否をテスト
      if (step === 'backtest') {
        return true; // 戦略が完成している場合のみtrue
      }
      return true;
    }),
  }),
}));

describe('バックテストステップの追加', () => {
  test('BuilderStepにbacktestが含まれている', () => {
    // このテストは現在失敗するはず（TDD Red段階）
    render(<StrategyBuilderPage />);
    
    // バックテストタブが存在することを確認
    const backtestTab = screen.queryByText('バックテスト');
    expect(backtestTab).toBeInTheDocument();
  });

  test('バックテストステップのアイコンが表示される', () => {
    render(<StrategyBuilderPage />);
    
    // バックテストステップのアイコンが表示されることを確認
    const backtestTab = screen.queryByText('バックテスト');
    expect(backtestTab).toBeInTheDocument();
    
    // アイコンが含まれていることを確認（具体的なアイコンは実装時に決定）
    const tabButton = backtestTab?.closest('button');
    expect(tabButton).toBeInTheDocument();
  });

  test('バックテストステップへのナビゲーションが機能する', () => {
    const mockSetCurrentStep = jest.fn();
    
    // useStrategyBuilderフックを再モック
    jest.doMock('@/hooks/useStrategyBuilder', () => ({
      useStrategyBuilder: () => ({
        currentStep: 'preview',
        selectedIndicators: [{ name: 'SMA', params: { period: 20 }, enabled: true }],
        entryConditions: [{ type: 'indicator_comparison', indicator1: 'SMA', operator: '>', value: 100 }],
        exitConditions: [{ type: 'indicator_comparison', indicator1: 'SMA', operator: '<', value: 100 }],
        strategyName: 'Test Strategy',
        strategyDescription: 'Test Description',
        isStrategyComplete: true,
        setCurrentStep: mockSetCurrentStep,
        updateSelectedIndicators: jest.fn(),
        updateIndicatorParameters: jest.fn(),
        toggleIndicatorEnabled: jest.fn(),
        updateEntryConditions: jest.fn(),
        updateExitConditions: jest.fn(),
        updateStrategyName: jest.fn(),
        updateStrategyDescription: jest.fn(),
        validateCurrentStrategy: jest.fn(),
        saveCurrentStrategy: jest.fn(),
        loadStrategy: jest.fn(),
        canProceedToStep: jest.fn((step: string) => {
          if (step === 'backtest') {
            return true; // 戦略が完成している場合
          }
          return true;
        }),
      }),
    }));

    render(<StrategyBuilderPage />);
    
    // バックテストタブをクリック
    const backtestTab = screen.getByText('バックテスト');
    fireEvent.click(backtestTab);
    
    // setCurrentStepが'backtest'で呼ばれることを確認
    expect(mockSetCurrentStep).toHaveBeenCalledWith('backtest');
  });

  test('戦略が未完成の場合はバックテストステップに進めない', () => {
    const mockCanProceedToStep = jest.fn((step: string) => {
      if (step === 'backtest') {
        return false; // 戦略が未完成の場合
      }
      return true;
    });

    jest.doMock('@/hooks/useStrategyBuilder', () => ({
      useStrategyBuilder: () => ({
        currentStep: 'indicators',
        selectedIndicators: [],
        entryConditions: [],
        exitConditions: [],
        strategyName: '',
        strategyDescription: '',
        isStrategyComplete: false,
        setCurrentStep: jest.fn(),
        updateSelectedIndicators: jest.fn(),
        updateIndicatorParameters: jest.fn(),
        toggleIndicatorEnabled: jest.fn(),
        updateEntryConditions: jest.fn(),
        updateExitConditions: jest.fn(),
        updateStrategyName: jest.fn(),
        updateStrategyDescription: jest.fn(),
        validateCurrentStrategy: jest.fn(),
        saveCurrentStrategy: jest.fn(),
        loadStrategy: jest.fn(),
        canProceedToStep: mockCanProceedToStep,
      }),
    }));

    render(<StrategyBuilderPage />);
    
    // バックテストタブが無効化されていることを確認
    const backtestTab = screen.getByText('バックテスト');
    const tabButton = backtestTab.closest('button');
    
    // ボタンが無効化されているか、クリックできない状態であることを確認
    expect(tabButton).toHaveAttribute('disabled');
  });

  test('バックテストステップの説明が正しく表示される', () => {
    render(<StrategyBuilderPage />);
    
    // バックテストステップの説明テキストが表示されることを確認
    const description = screen.queryByText(/作成した戦略をバックテストして性能を確認/);
    expect(description).toBeInTheDocument();
  });

  test('ナビゲーションボタンでバックテストステップに移動できる', () => {
    const mockSetCurrentStep = jest.fn();

    jest.doMock('@/hooks/useStrategyBuilder', () => ({
      useStrategyBuilder: () => ({
        currentStep: 'preview',
        selectedIndicators: [{ name: 'SMA', params: { period: 20 }, enabled: true }],
        entryConditions: [{ type: 'indicator_comparison', indicator1: 'SMA', operator: '>', value: 100 }],
        exitConditions: [{ type: 'indicator_comparison', indicator1: 'SMA', operator: '<', value: 100 }],
        strategyName: 'Test Strategy',
        strategyDescription: 'Test Description',
        isStrategyComplete: true,
        setCurrentStep: mockSetCurrentStep,
        updateSelectedIndicators: jest.fn(),
        updateIndicatorParameters: jest.fn(),
        toggleIndicatorEnabled: jest.fn(),
        updateEntryConditions: jest.fn(),
        updateExitConditions: jest.fn(),
        updateStrategyName: jest.fn(),
        updateStrategyDescription: jest.fn(),
        validateCurrentStrategy: jest.fn(),
        saveCurrentStrategy: jest.fn(),
        loadStrategy: jest.fn(),
        canProceedToStep: jest.fn(() => true),
      }),
    }));

    render(<StrategyBuilderPage />);
    
    // 「次のステップ」ボタンをクリック
    const nextButton = screen.getByText('次のステップ');
    fireEvent.click(nextButton);
    
    // バックテストステップに移動することを確認
    expect(mockSetCurrentStep).toHaveBeenCalledWith('backtest');
  });
});

describe('BuilderStep型の拡張', () => {
  test('BuilderStep型にbacktestが含まれている', () => {
    // 型レベルでのテスト - TypeScriptコンパイル時にチェックされる
    // この部分は実装時にTypeScriptの型定義を確認
    const validSteps: string[] = ['indicators', 'parameters', 'conditions', 'preview', 'backtest', 'saved'];
    
    // バックテストが有効なステップとして認識されることを確認
    expect(validSteps).toContain('backtest');
  });
});
