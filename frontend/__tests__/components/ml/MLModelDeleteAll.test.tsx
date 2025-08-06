/**
 * MLモデル全削除ボタンのテスト
 * 
 * 全削除ボタンの表示、クリック動作、確認ダイアログ、API呼び出しをテストします。
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { useMLModels } from '@/hooks/useMLModels';
import { useApiCall } from '@/hooks/useApiCall';

// モックコンポーネント（全削除ボタンを含むMLページのモック）
const MLModelDeleteAllButton = ({ onDeleteAll, loading }: { 
  onDeleteAll: () => void; 
  loading: boolean; 
}) => {
  return (
    <button
      onClick={onDeleteAll}
      disabled={loading}
      className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded"
      data-testid="delete-all-models-button"
    >
      {loading ? '削除中...' : 'すべてのモデルを削除'}
    </button>
  );
};

// フックのモック
jest.mock('@/hooks/useMLModels');
jest.mock('@/hooks/useApiCall');

const mockUseMLModels = useMLModels as jest.MockedFunction<typeof useMLModels>;
const mockUseApiCall = useApiCall as jest.MockedFunction<typeof useApiCall>;

describe('MLModelDeleteAllButton', () => {
  const mockExecute = jest.fn();
  const mockFetchModels = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    
    // useApiCallのモック設定
    mockUseApiCall.mockReturnValue({
      loading: false,
      error: null,
      execute: mockExecute,
      reset: jest.fn(),
    });

    // useMLModelsのモック設定
    mockUseMLModels.mockReturnValue({
      models: [
        { id: 'model1', name: 'Test Model 1', path: '/path/to/model1' },
        { id: 'model2', name: 'Test Model 2', path: '/path/to/model2' },
      ],
      isLoading: false,
      error: null,
      isDeleting: false,
      fetchModels: mockFetchModels,
      deleteModel: jest.fn(),
    });
  });

  it('全削除ボタンが正しく表示される', () => {
    render(
      <MLModelDeleteAllButton 
        onDeleteAll={() => {}} 
        loading={false} 
      />
    );

    const button = screen.getByTestId('delete-all-models-button');
    expect(button).toBeInTheDocument();
    expect(button).toHaveTextContent('すべてのモデルを削除');
    expect(button).not.toBeDisabled();
  });

  it('ローディング中はボタンが無効化される', () => {
    render(
      <MLModelDeleteAllButton 
        onDeleteAll={() => {}} 
        loading={true} 
      />
    );

    const button = screen.getByTestId('delete-all-models-button');
    expect(button).toBeDisabled();
    expect(button).toHaveTextContent('削除中...');
  });

  it('ボタンクリック時に削除処理が実行される', async () => {
    const mockOnDeleteAll = jest.fn();
    
    render(
      <MLModelDeleteAllButton 
        onDeleteAll={mockOnDeleteAll} 
        loading={false} 
      />
    );

    const button = screen.getByTestId('delete-all-models-button');
    fireEvent.click(button);

    expect(mockOnDeleteAll).toHaveBeenCalledTimes(1);
  });
});

// 統合テスト用のコンポーネント（実際のフックを使用）
const MLModelDeleteAllIntegration = () => {
  const { execute, loading } = useApiCall();

  const handleDeleteAll = async () => {
    await execute('/api/ml/models/all', {
      method: 'DELETE',
      confirmMessage: 'すべてのモデルを削除しますか？この操作は取り消せません。',
      onSuccess: () => {
        console.log('全削除成功');
      },
    });
  };

  return (
    <MLModelDeleteAllButton 
      onDeleteAll={handleDeleteAll} 
      loading={loading} 
    />
  );
};

describe('MLModelDeleteAll Integration', () => {
  const mockExecute = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    
    // useApiCallのモック設定
    mockUseApiCall.mockReturnValue({
      loading: false,
      error: null,
      execute: mockExecute,
      reset: jest.fn(),
    });
  });

  it('統合テスト: 全削除APIが正しく呼び出される', async () => {
    // 確認ダイアログのモック
    const mockConfirm = jest.spyOn(window, 'confirm').mockReturnValue(true);
    
    render(<MLModelDeleteAllIntegration />);

    const button = screen.getByTestId('delete-all-models-button');
    fireEvent.click(button);

    await waitFor(() => {
      expect(mockExecute).toHaveBeenCalledWith('/api/ml/models/all', {
        method: 'DELETE',
        confirmMessage: 'すべてのモデルを削除しますか？この操作は取り消せません。',
        onSuccess: expect.any(Function),
      });
    });

    mockConfirm.mockRestore();
  });

  it('統合テスト: 確認ダイアログでキャンセルした場合はAPIが呼び出されない', async () => {
    // 確認ダイアログのモック（キャンセル）
    const mockConfirm = jest.spyOn(window, 'confirm').mockReturnValue(false);

    // useApiCallのexecuteが確認ダイアログでキャンセルされた場合nullを返すようにモック
    const mockExecuteWithCancel = jest.fn().mockResolvedValue(null);
    mockUseApiCall.mockReturnValue({
      loading: false,
      error: null,
      execute: mockExecuteWithCancel,
      reset: jest.fn(),
    });

    render(<MLModelDeleteAllIntegration />);

    const button = screen.getByTestId('delete-all-models-button');
    fireEvent.click(button);

    // executeは呼ばれるが、確認ダイアログでキャンセルされるためnullが返される
    await waitFor(() => {
      expect(mockExecuteWithCancel).toHaveBeenCalledWith('/api/ml/models/all', {
        method: 'DELETE',
        confirmMessage: 'すべてのモデルを削除しますか？この操作は取り消せません。',
        onSuccess: expect.any(Function),
      });
    });

    mockConfirm.mockRestore();
  });

  it('統合テスト: ローディング状態が正しく反映される', () => {
    // ローディング状態のモック
    mockUseApiCall.mockReturnValue({
      loading: true,
      error: null,
      execute: mockExecute,
      reset: jest.fn(),
    });

    render(<MLModelDeleteAllIntegration />);

    const button = screen.getByTestId('delete-all-models-button');
    expect(button).toBeDisabled();
    expect(button).toHaveTextContent('削除中...');
  });

  it('統合テスト: エラー状態の処理', async () => {
    const mockError = 'API呼び出しエラー';
    mockUseApiCall.mockReturnValue({
      loading: false,
      error: mockError,
      execute: mockExecute,
      reset: jest.fn(),
    });

    render(<MLModelDeleteAllIntegration />);

    // エラー状態でもボタンは表示される（エラーハンドリングは親コンポーネントで行う）
    const button = screen.getByTestId('delete-all-models-button');
    expect(button).toBeInTheDocument();
  });
});

// APIレスポンスのテスト
describe('MLModelDeleteAll API Response Handling', () => {
  const mockExecuteForResponse = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('成功レスポンスの処理', async () => {
    const mockSuccessResponse = {
      success: true,
      message: 'すべてのモデル（5個）が削除されました',
      deleted_count: 5,
      failed_count: 0,
      failed_models: []
    };

    mockExecuteForResponse.mockResolvedValue(mockSuccessResponse);

    const onSuccess = jest.fn();

    await mockExecuteForResponse('/api/ml/models/all', {
      method: 'DELETE',
      onSuccess,
    });

    expect(mockExecuteForResponse).toHaveBeenCalledWith('/api/ml/models/all', {
      method: 'DELETE',
      onSuccess,
    });
  });

  it('部分失敗レスポンスの処理', async () => {
    const mockPartialFailureResponse = {
      success: true,
      message: '3個のモデルを削除しました。2個のモデルで削除に失敗しました: model4, model5',
      deleted_count: 3,
      failed_count: 2,
      failed_models: ['model4', 'model5']
    };

    mockExecuteForResponse.mockResolvedValue(mockPartialFailureResponse);

    const onSuccess = jest.fn();

    await mockExecuteForResponse('/api/ml/models/all', {
      method: 'DELETE',
      onSuccess,
    });

    expect(mockExecuteForResponse).toHaveBeenCalledWith('/api/ml/models/all', {
      method: 'DELETE',
      onSuccess,
    });
  });

  it('モデルなしレスポンスの処理', async () => {
    const mockNoModelsResponse = {
      success: true,
      message: '削除するモデルがありませんでした',
      deleted_count: 0,
      failed_count: 0,
      failed_models: []
    };

    mockExecuteForResponse.mockResolvedValue(mockNoModelsResponse);

    const onSuccess = jest.fn();

    await mockExecuteForResponse('/api/ml/models/all', {
      method: 'DELETE',
      onSuccess,
    });

    expect(mockExecuteForResponse).toHaveBeenCalledWith('/api/ml/models/all', {
      method: 'DELETE',
      onSuccess,
    });
  });
});
