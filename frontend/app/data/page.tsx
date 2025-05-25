/**
 * データページコンポーネント
 *
 * ローソク足チャートを表示し、通貨ペアと時間軸を選択できるページです。
 * リアルタイムでチャートデータを取得・表示します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

'use client';

import React, { useState, useEffect } from 'react';
import CandlestickChart from '@/components/CandlestickChart';
import TimeFrameSelector from '@/components/TimeFrameSelector';
import SymbolSelector from '@/components/SymbolSelector';
import { 
  CandlestickData, 
  TimeFrame, 
  TradingPair, 
  CandlestickResponse 
} from '@/types/strategy';

/**
 * データページコンポーネント
 */
const DataPage: React.FC = () => {
  // 状態管理
  const [symbols, setSymbols] = useState<TradingPair[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('BTC/USD');
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<TimeFrame>('1d');
  const [candlestickData, setCandlestickData] = useState<CandlestickData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [symbolsLoading, setSymbolsLoading] = useState<boolean>(true);

  /**
   * 通貨ペア一覧を取得
   */
  const fetchSymbols = async () => {
    try {
      setSymbolsLoading(true);
      const response = await fetch('/api/data/symbols');
      const result = await response.json();
      
      if (result.success) {
        setSymbols(result.data);
      } else {
        setError('通貨ペア一覧の取得に失敗しました');
      }
    } catch (err) {
      setError('通貨ペア一覧の取得中にエラーが発生しました');
      console.error('通貨ペア取得エラー:', err);
    } finally {
      setSymbolsLoading(false);
    }
  };

  /**
   * ローソク足データを取得
   */
  const fetchCandlestickData = async () => {
    try {
      setLoading(true);
      setError('');
      
      const params = new URLSearchParams({
        symbol: selectedSymbol,
        timeframe: selectedTimeFrame,
        limit: '100',
      });
      
      const response = await fetch(`/api/data/candlesticks?${params}`);
      const result: CandlestickResponse = await response.json();
      
      if (result.success) {
        setCandlestickData(result.data.candlesticks);
      } else {
        setError(result.message || 'データの取得に失敗しました');
      }
    } catch (err) {
      setError('データの取得中にエラーが発生しました');
      console.error('ローソク足データ取得エラー:', err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * 通貨ペア変更ハンドラ
   */
  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol);
  };

  /**
   * 時間軸変更ハンドラ
   */
  const handleTimeFrameChange = (timeFrame: TimeFrame) => {
    setSelectedTimeFrame(timeFrame);
  };

  /**
   * データ更新ハンドラ
   */
  const handleRefresh = () => {
    fetchCandlestickData();
  };

  // 初期データ取得
  useEffect(() => {
    fetchSymbols();
  }, []);

  // 通貨ペアまたは時間軸変更時にデータを再取得
  useEffect(() => {
    if (selectedSymbol && selectedTimeFrame) {
      fetchCandlestickData();
    }
  }, [selectedSymbol, selectedTimeFrame]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* ヘッダー */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                チャートデータ
              </h1>
              <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                仮想通貨のローソク足チャートを表示します
              </p>
            </div>
            
            <button
              onClick={handleRefresh}
              disabled={loading}
              className={`
                px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200
                ${
                  loading
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                }
              `}
            >
              {loading ? '更新中...' : 'データ更新'}
            </button>
          </div>
        </div>
      </div>

      {/* メインコンテンツ */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* コントロールパネル */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            {/* 通貨ペア選択 */}
            <SymbolSelector
              symbols={symbols}
              selectedSymbol={selectedSymbol}
              onSymbolChange={handleSymbolChange}
              loading={symbolsLoading}
              disabled={loading}
            />
            
            {/* 時間軸選択 */}
            <TimeFrameSelector
              selectedTimeFrame={selectedTimeFrame}
              onTimeFrameChange={handleTimeFrameChange}
              disabled={loading}
            />
          </div>
        </div>

        {/* チャート表示エリア */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {selectedSymbol} - {selectedTimeFrame}足チャート
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {candlestickData.length > 0 && !loading && (
                `${candlestickData.length}件のデータを表示中`
              )}
            </p>
          </div>
          
          <CandlestickChart
            data={candlestickData}
            height={500}
            loading={loading}
            error={error}
          />
        </div>

        {/* データ情報 */}
        {candlestickData.length > 0 && !loading && !error && (
          <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              データ概要
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-600 dark:text-gray-400">データ期間:</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {new Date(candlestickData[0]?.timestamp).toLocaleDateString('ja-JP')} - {' '}
                  {new Date(candlestickData[candlestickData.length - 1]?.timestamp).toLocaleDateString('ja-JP')}
                </p>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">最新価格:</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  ${candlestickData[candlestickData.length - 1]?.close.toFixed(2)}
                </p>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">最高値:</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  ${Math.max(...candlestickData.map(d => d.high)).toFixed(2)}
                </p>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">最安値:</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  ${Math.min(...candlestickData.map(d => d.low)).toFixed(2)}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataPage;
