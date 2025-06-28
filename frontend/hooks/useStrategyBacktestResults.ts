/**
 * ストラテジービルダーのバックテスト結果管理フック
 * 
 * バックテスト結果の保存、読み込み、削除機能を提供します。
 */

import { useState, useEffect, useCallback } from 'react';
import { BacktestResult } from '@/types/backtest';

const STORAGE_KEY = 'strategy_backtest_results';
const MAX_RESULTS = 50; // 最大保存数

export interface SavedBacktestResult extends BacktestResult {
  saved_at: string;
}

export interface UseStrategyBacktestResultsReturn {
  savedResults: SavedBacktestResult[];
  saveResult: (result: BacktestResult) => Promise<void>;
  deleteResult: (id: string) => Promise<void>;
  loadResults: () => void;
  exportResults: (results: SavedBacktestResult[]) => void;
  clearAllResults: () => Promise<void>;
  isLoading: boolean;
  error: string | null;
}

/**
 * ストラテジービルダーのバックテスト結果管理フック
 */
export const useStrategyBacktestResults = (): UseStrategyBacktestResultsReturn => {
  const [savedResults, setSavedResults] = useState<SavedBacktestResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * localStorageから結果を読み込む
   */
  const loadResults = useCallback(() => {
    try {
      setIsLoading(true);
      setError(null);
      
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const results: SavedBacktestResult[] = JSON.parse(stored);
        // 作成日時でソート（新しい順）
        results.sort((a, b) => new Date(b.saved_at).getTime() - new Date(a.saved_at).getTime());
        setSavedResults(results);
      } else {
        setSavedResults([]);
      }
    } catch (err) {
      console.error('結果読み込みエラー:', err);
      setError('保存された結果の読み込みに失敗しました');
      setSavedResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * 結果をlocalStorageに保存する
   */
  const saveResult = useCallback(async (result: BacktestResult): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);

      const savedResult: SavedBacktestResult = {
        ...result,
        saved_at: new Date().toISOString(),
      };

      // 既存の結果を読み込み
      const stored = localStorage.getItem(STORAGE_KEY);
      let existingResults: SavedBacktestResult[] = [];
      
      if (stored) {
        existingResults = JSON.parse(stored);
      }

      // 新しい結果を先頭に追加
      const updatedResults = [savedResult, ...existingResults];

      // 最大数を超えた場合は古いものを削除
      if (updatedResults.length > MAX_RESULTS) {
        updatedResults.splice(MAX_RESULTS);
      }

      // localStorageに保存
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedResults));
      
      // 状態を更新
      setSavedResults(updatedResults);
      
    } catch (err) {
      console.error('結果保存エラー:', err);
      setError('結果の保存に失敗しました');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * 指定されたIDの結果を削除する
   */
  const deleteResult = useCallback(async (id: string): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);

      const stored = localStorage.getItem(STORAGE_KEY);
      if (!stored) {
        return;
      }

      const existingResults: SavedBacktestResult[] = JSON.parse(stored);
      const filteredResults = existingResults.filter(result => result.id !== id);

      localStorage.setItem(STORAGE_KEY, JSON.stringify(filteredResults));
      setSavedResults(filteredResults);
      
    } catch (err) {
      console.error('結果削除エラー:', err);
      setError('結果の削除に失敗しました');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * 結果をJSONファイルとしてエクスポートする
   */
  const exportResults = useCallback((results: SavedBacktestResult[]) => {
    try {
      const dataStr = JSON.stringify(results, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `strategy_backtest_results_${new Date().toISOString().split('T')[0]}.json`;
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('エクスポートエラー:', err);
      setError('結果のエクスポートに失敗しました');
    }
  }, []);

  /**
   * 全ての結果を削除する
   */
  const clearAllResults = useCallback(async (): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);

      localStorage.removeItem(STORAGE_KEY);
      setSavedResults([]);
      
    } catch (err) {
      console.error('全削除エラー:', err);
      setError('結果の全削除に失敗しました');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // 初期化時に結果を読み込み
  useEffect(() => {
    loadResults();
  }, [loadResults]);

  return {
    savedResults,
    saveResult,
    deleteResult,
    loadResults,
    exportResults,
    clearAllResults,
    isLoading,
    error,
  };
};
