import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import OptimizationSettings, { OptimizationSettingsConfig } from '@/components/ml/OptimizationSettings';

describe('OptimizationSettings', () => {
  const defaultSettings: OptimizationSettingsConfig = {
    enabled: false,
    method: 'bayesian',
    n_calls: 50,
    parameter_space: {},
  };

  const mockOnChange = jest.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  it('renders correctly with default settings', () => {
    render(
      <OptimizationSettings
        settings={defaultSettings}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByText('ハイパーパラメータ最適化設定')).toBeInTheDocument();
    expect(screen.getByLabelText('ハイパーパラメータ自動最適化を有効にする')).not.toBeChecked();
  });

  it('shows optimization options when enabled', () => {
    const enabledSettings = { ...defaultSettings, enabled: true };
    
    render(
      <OptimizationSettings
        settings={enabledSettings}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByText('最適化手法')).toBeInTheDocument();
    expect(screen.getByText('ベイジアン最適化')).toBeInTheDocument();
    expect(screen.getByText('グリッドサーチ')).toBeInTheDocument();
    expect(screen.getByText('ランダムサーチ')).toBeInTheDocument();
    expect(screen.getByText('最適化試行回数')).toBeInTheDocument();
    expect(screen.getByText('パラメータ空間設定')).toBeInTheDocument();
  });

  it('calls onChange when enabled state changes', () => {
    render(
      <OptimizationSettings
        settings={defaultSettings}
        onChange={mockOnChange}
      />
    );

    const enableSwitch = screen.getByLabelText('ハイパーパラメータ自動最適化を有効にする');
    fireEvent.click(enableSwitch);

    expect(mockOnChange).toHaveBeenCalledWith({
      ...defaultSettings,
      enabled: true,
    });
  });

  it('calls onChange when method changes', () => {
    const enabledSettings = { ...defaultSettings, enabled: true };
    
    render(
      <OptimizationSettings
        settings={enabledSettings}
        onChange={mockOnChange}
      />
    );

    const gridSearchButton = screen.getByText('グリッドサーチ');
    fireEvent.click(gridSearchButton);

    expect(mockOnChange).toHaveBeenCalledWith({
      ...enabledSettings,
      method: 'grid',
    });
  });

  it('calls onChange when n_calls changes', () => {
    const enabledSettings = { ...defaultSettings, enabled: true };
    
    render(
      <OptimizationSettings
        settings={enabledSettings}
        onChange={mockOnChange}
      />
    );

    const nCallsInput = screen.getByDisplayValue('50');
    fireEvent.change(nCallsInput, { target: { value: '100' } });

    expect(mockOnChange).toHaveBeenCalledWith({
      ...enabledSettings,
      n_calls: 100,
    });
  });

  it('adds new parameter correctly', () => {
    const enabledSettings = { ...defaultSettings, enabled: true };
    
    render(
      <OptimizationSettings
        settings={enabledSettings}
        onChange={mockOnChange}
      />
    );

    // パラメータ名を入力
    const paramNameInput = screen.getByPlaceholderText('parameter_name');
    fireEvent.change(paramNameInput, { target: { value: 'test_param' } });

    // 追加ボタンをクリック
    const addButton = screen.getByText('追加');
    fireEvent.click(addButton);

    expect(mockOnChange).toHaveBeenCalledWith({
      ...enabledSettings,
      parameter_space: {
        test_param: {
          type: 'real',
          low: 0.01,
          high: 1.0,
        },
      },
    });
  });

  it('removes parameter correctly', () => {
    const settingsWithParam: OptimizationSettingsConfig = {
      ...defaultSettings,
      enabled: true,
      parameter_space: {
        test_param: {
          type: 'real',
          low: 0.01,
          high: 1.0,
        },
      },
    };
    
    render(
      <OptimizationSettings
        settings={settingsWithParam}
        onChange={mockOnChange}
      />
    );

    // 削除ボタンをクリック
    const deleteButton = screen.getByRole('button', { name: '' }); // Trash2アイコンのボタン
    fireEvent.click(deleteButton);

    expect(mockOnChange).toHaveBeenCalledWith({
      ...settingsWithParam,
      parameter_space: {},
    });
  });

  it('loads default parameters correctly', () => {
    const enabledSettings = { ...defaultSettings, enabled: true };
    
    render(
      <OptimizationSettings
        settings={enabledSettings}
        onChange={mockOnChange}
      />
    );

    const loadDefaultButton = screen.getByText('デフォルト設定を読み込み');
    fireEvent.click(loadDefaultButton);

    expect(mockOnChange).toHaveBeenCalledWith({
      ...enabledSettings,
      parameter_space: {
        num_leaves: { type: 'integer', low: 10, high: 100 },
        learning_rate: { type: 'real', low: 0.01, high: 0.3 },
        feature_fraction: { type: 'real', low: 0.5, high: 1.0 },
        bagging_fraction: { type: 'real', low: 0.5, high: 1.0 },
        min_data_in_leaf: { type: 'integer', low: 5, high: 50 },
      },
    });
  });

  it('updates parameter values correctly', () => {
    const settingsWithParam: OptimizationSettingsConfig = {
      ...defaultSettings,
      enabled: true,
      parameter_space: {
        test_param: {
          type: 'real',
          low: 0.01,
          high: 1.0,
        },
      },
    };
    
    render(
      <OptimizationSettings
        settings={settingsWithParam}
        onChange={mockOnChange}
      />
    );

    // 最小値を変更
    const lowInput = screen.getByDisplayValue('0.01');
    fireEvent.change(lowInput, { target: { value: '0.05' } });

    expect(mockOnChange).toHaveBeenCalledWith({
      ...settingsWithParam,
      parameter_space: {
        test_param: {
          type: 'real',
          low: 0.05,
          high: 1.0,
        },
      },
    });
  });

  it('handles categorical parameters correctly', () => {
    const enabledSettings = { ...defaultSettings, enabled: true };
    
    render(
      <OptimizationSettings
        settings={enabledSettings}
        onChange={mockOnChange}
      />
    );

    // パラメータ名を入力
    const paramNameInput = screen.getByPlaceholderText('parameter_name');
    fireEvent.change(paramNameInput, { target: { value: 'algorithm' } });

    // 型をカテゴリに変更
    const typeSelect = screen.getByDisplayValue('実数');
    fireEvent.change(typeSelect, { target: { value: 'categorical' } });

    // 追加ボタンをクリック
    const addButton = screen.getByText('追加');
    fireEvent.click(addButton);

    expect(mockOnChange).toHaveBeenCalledWith({
      ...enabledSettings,
      parameter_space: {
        algorithm: {
          type: 'categorical',
          categories: ['option1', 'option2'],
        },
      },
    });
  });

  it('validates parameter name input', () => {
    const enabledSettings = { ...defaultSettings, enabled: true };
    
    render(
      <OptimizationSettings
        settings={enabledSettings}
        onChange={mockOnChange}
      />
    );

    // 空のパラメータ名で追加ボタンをクリック
    const addButton = screen.getByText('追加');
    expect(addButton).toBeDisabled();

    // パラメータ名を入力すると有効になる
    const paramNameInput = screen.getByPlaceholderText('parameter_name');
    fireEvent.change(paramNameInput, { target: { value: 'test' } });
    
    expect(addButton).not.toBeDisabled();
  });

  it('displays method descriptions correctly', () => {
    const enabledSettings = { ...defaultSettings, enabled: true };
    
    render(
      <OptimizationSettings
        settings={enabledSettings}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByText('効率的な最適化')).toBeInTheDocument();
    expect(screen.getByText('網羅的な探索')).toBeInTheDocument();
    expect(screen.getByText('ランダムな探索')).toBeInTheDocument();
  });
});
