import React, { useState, useEffect } from 'react';
import { format, subDays } from 'date-fns';
import { Info, CheckCircle, XCircle, HelpCircle, X, Play, StopCircle } from 'lucide-react';
import { SUPPORTED_TRADING_PAIRS, SUPPORTED_TIMEFRAMES, DEFAULT_TRADING_PAIR, DEFAULT_TIMEFRAME } from '@/constants';
import ActionButton from '@/components/common/ActionButton';

interface MLTrainingModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface MLTrainingConfig {
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  validation_split: number;
  prediction_horizon: number;
  threshold_up: number;
  threshold_down: number;
  save_model: boolean;
  train_test_split: number;
  cross_validation_folds: number;
  random_state: number;
  early_stopping_rounds: number;
  max_depth: number;
  n_estimators: number;
  learning_rate: number;
}

interface MLTrainingStatus {
  is_training: boolean;
  progress: number;
  status: string;
  message: string;
  start_time?: string;
  end_time?: string;
  model_info?: {
    accuracy: number;
    loss: number;
    model_path: string;
    feature_count: number;
    training_samples: number;
    validation_split: number;
  };
  error?: string;
}

const MLTrainingModal: React.FC<MLTrainingModalProps> = ({ isOpen, onClose }) => {
  // const toast = useToast(); // TODO: Replace with a proper notification system
  const [config, setConfig] = useState<MLTrainingConfig>({
    symbol: DEFAULT_TRADING_PAIR,
    timeframe: DEFAULT_TIMEFRAME,
    start_date: format(subDays(new Date(), 90), 'yyyy-MM-dd'),
    end_date: format(new Date(), 'yyyy-MM-dd'),
    validation_split: 0.2,
    prediction_horizon: 24,
    threshold_up: 0.02,
    threshold_down: -0.02,
    save_model: true,
    // 新しい設定項目のデフォルト値
    train_test_split: 0.8,
    cross_validation_folds: 5,
    random_state: 42,
    early_stopping_rounds: 100,
    max_depth: 10,
    n_estimators: 100,
    learning_rate: 0.1,
  });

  const [status, setStatus] = useState<MLTrainingStatus>({
    is_training: false,
    progress: 0,
    status: 'idle',
    message: '',
  });

  const [isLoading, setIsLoading] = useState(false);

  // ステータスポーリング
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (status.is_training) {
      interval = setInterval(async () => {
        try {
          const response = await fetch('/api/ml/status');
          const statusData = await response.json();
          setStatus(statusData);
          
          if (!statusData.is_training) {
            clearInterval(interval);
            if (statusData.status === 'completed') {
              console.log('Training completed');
              // toast({
              //   title: 'トレーニング完了',
              //   description: 'MLモデルのトレーニングが正常に完了しました',
              //   status: 'success',
              //   duration: 5000,
              //   isClosable: true,
              // });
            } else if (statusData.status === 'error') {
              console.error('Training error:', statusData.error);
              // toast({
              //   title: 'トレーニングエラー',
              //   description: statusData.error || 'トレーニング中にエラーが発生しました',
              //   status: 'error',
              //   duration: 5000,
              //   isClosable: true,
              // });
            }
          }
        } catch (error) {
          console.error('ステータス取得エラー:', error);
        }
      }, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [status.is_training]);

  const handleStartTraining = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/ml/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      const result = await response.json();

      if (result.success) {
        setStatus(prev => ({ ...prev, is_training: true }));
        console.log('Training started');
        // toast({
        //   title: 'トレーニング開始',
        //   description: 'MLモデルのトレーニングを開始しました',
        //   status: 'info',
        //   duration: 3000,
        //   isClosable: true,
        // });
      } else {
        throw new Error(result.message || 'トレーニングの開始に失敗しました');
      }
    } catch (error) {
      console.error('Failed to start training:', error);
      // toast({
      //   title: 'エラー',
      //   description: error instanceof Error ? error.message : 'トレーニングの開始に失敗しました',
      //   status: 'error',
      //   duration: 5000,
      //   isClosable: true,
      // });
    } finally {
      setIsLoading(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      const response = await fetch('/api/ml/stop', {
        method: 'POST',
      });

      const result = await response.json();

      if (result.success) {
        setStatus(prev => ({ ...prev, is_training: false }));
        console.log('Training stopped');
        // toast({
        //   title: 'トレーニング停止',
        //   description: 'MLモデルのトレーニングを停止しました',
        //   status: 'warning',
        //   duration: 3000,
        //   isClosable: true,
        // });
      }
    } catch (error) {
      console.error('Failed to stop training:', error);
      // toast({
      //   title: 'エラー',
      //   description: 'トレーニングの停止に失敗しました',
      //   status: 'error',
      //   duration: 5000,
      //   isClosable: true,
      // });
    }
  };

  const getStatusInfo = (status: string) => {
    switch (status) {
      case 'completed':
        return { color: 'green', icon: <CheckCircle /> };
      case 'error':
        return { color: 'red', icon: <XCircle /> };
      case 'training':
      case 'loading_data':
      case 'initializing':
        return { color: 'blue', icon: <Info /> };
      default:
        return { color: 'gray', icon: <HelpCircle /> };
    }
  };

  const getStatusMessage = (status: string) => {
    switch (status) {
      case 'idle':
        return 'トレーニング待機中';
      case 'starting':
        return 'トレーニング開始中...';
      case 'loading_data':
        return 'データ読み込み中...';
      case 'initializing':
        return 'MLサービス初期化中...';
      case 'training':
        return 'モデルトレーニング中...';
      case 'completed':
        return 'トレーニング完了';
      case 'error':
        return 'エラーが発生しました';
      case 'stopped':
        return 'トレーニングが停止されました';
      default:
        return status;
    }
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className={`fixed inset-0 z-50 ${isOpen ? 'flex' : 'hidden'} items-center justify-center bg-black bg-opacity-50 p-4`}
      onClick={handleBackdropClick}
    >
      <div className="bg-secondary-950 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden border border-secondary-700 flex flex-col">
        {/* ヘッダー */}
        <div className="flex items-center justify-between p-6 border-b border-secondary-700">
          <div>
            <h2 className="text-2xl font-bold text-secondary-100">🧠 MLモデルトレーニング</h2>
            <p className="text-sm text-secondary-400 mt-1">
              機械学習モデルをトレーニングして、市場の予測シグナルを生成します
            </p>
          </div>
          {!status.is_training && (
            <button
              onClick={onClose}
              className="p-2 hover:bg-secondary-700 rounded-lg transition-colors disabled:opacity-50"
            >
              <X className="w-6 h-6 text-secondary-400" />
            </button>
          )}
        </div>

        {/* コンテンツ */}
        <div className="p-6 overflow-y-auto flex-grow">
          <div className="space-y-6">
            {/* トレーニング設定 */}
            <div>
              <p className="text-lg font-bold mb-2 text-secondary-200">基本設定</p>
              <p className="text-sm text-secondary-400 mb-4">
                MLモデルのトレーニングに使用するデータの範囲と基本パラメータを設定します。
              </p>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1 text-secondary-300">シンボル</label>
                    <select
                      value={config.symbol}
                      onChange={(e) => setConfig(prev => ({ ...prev, symbol: e.target.value }))}
                      disabled={status.is_training}
                      className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                    >
                      {SUPPORTED_TRADING_PAIRS.map((pair) => (
                        <option key={pair.symbol} value={pair.symbol}>
                          {pair.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1 text-secondary-300">時間軸</label>
                    <select
                      value={config.timeframe}
                      onChange={(e) => setConfig(prev => ({ ...prev, timeframe: e.target.value }))}
                      disabled={status.is_training}
                      className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                    >
                      {SUPPORTED_TIMEFRAMES.map((timeframe) => (
                        <option key={timeframe.value} value={timeframe.value}>
                          {timeframe.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1 text-secondary-300">開始日</label>
                    <input
                      type="date"
                      value={config.start_date}
                      onChange={(e) => setConfig(prev => ({ ...prev, start_date: e.target.value }))}
                      disabled={status.is_training}
                      className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1 text-secondary-300">終了日</label>
                    <input
                      type="date"
                      value={config.end_date}
                      onChange={(e) => setConfig(prev => ({ ...prev, end_date: e.target.value }))}
                      disabled={status.is_training}
                      className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                    />
                  </div>
                </div>

                {/* 詳細設定 */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1 text-secondary-300">検証データ分割比率</label>
                    <input
                      type="number"
                      min="0.1"
                      max="0.5"
                      step="0.05"
                      value={config.validation_split}
                      onChange={(e) => setConfig(prev => ({ ...prev, validation_split: parseFloat(e.target.value) }))}
                      disabled={status.is_training}
                      className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1 text-secondary-300">予測期間（時間）</label>
                    <input
                      type="number"
                      min="1"
                      max="168"
                      value={config.prediction_horizon}
                      onChange={(e) => setConfig(prev => ({ ...prev, prediction_horizon: parseInt(e.target.value) }))}
                      disabled={status.is_training}
                      className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1 text-secondary-300">上昇判定閾値</label>
                    <input
                      type="number"
                      min="0.001"
                      max="0.1"
                      step="0.001"
                      value={config.threshold_up}
                      onChange={(e) => setConfig(prev => ({ ...prev, threshold_up: parseFloat(e.target.value) }))}
                      disabled={status.is_training}
                      className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1 text-secondary-300">下落判定閾値</label>
                    <input
                      type="number"
                      min="-0.1"
                      max="-0.001"
                      step="0.001"
                      value={config.threshold_down}
                      onChange={(e) => setConfig(prev => ({ ...prev, threshold_down: parseFloat(e.target.value) }))}
                      disabled={status.is_training}
                      className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                    />
                  </div>
                </div>

                {/* データ分割設定 */}
                <div>
                  <h3 className="text-md font-bold mb-2 text-secondary-200">データ分割設定</h3>
                  <p className="text-sm text-secondary-400 mb-3">
                    機械学習でよく使われる7:3や8:2などのデータ分割比率を設定します。
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1 text-secondary-300">
                        トレーニング/テスト分割比率
                        <span className="text-xs text-secondary-500 ml-1">
                          (トレーニング用データの割合)
                        </span>
                      </label>
                      <input
                        type="number"
                        min="0.5"
                        max="0.95"
                        step="0.05"
                        value={config.train_test_split}
                        onChange={(e) => setConfig(prev => ({ ...prev, train_test_split: parseFloat(e.target.value) }))}
                        disabled={status.is_training}
                        className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                      />
                      <div className="text-xs text-secondary-500 mt-1">
                        {Math.round(config.train_test_split * 100)}% トレーニング / {Math.round((1 - config.train_test_split) * 100)}% テスト
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1 text-secondary-300">
                        クロスバリデーション分割数
                      </label>
                      <input
                        type="number"
                        min="3"
                        max="10"
                        value={config.cross_validation_folds}
                        onChange={(e) => setConfig(prev => ({ ...prev, cross_validation_folds: parseInt(e.target.value) }))}
                        disabled={status.is_training}
                        className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                      />
                    </div>
                  </div>
                </div>

                {/* モデルパラメータ設定 */}
                <div>
                  <h3 className="text-md font-bold mb-2 text-secondary-200">モデルパラメータ</h3>
                  <p className="text-sm text-secondary-400 mb-3">
                    RandomForestモデルの詳細パラメータを調整できます。デフォルト値でも十分な性能が期待できます。
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1 text-secondary-300">推定器数 (n_estimators)</label>
                      <input
                        type="number"
                        min="50"
                        max="500"
                        step="10"
                        value={config.n_estimators}
                        onChange={(e) => setConfig(prev => ({ ...prev, n_estimators: parseInt(e.target.value) }))}
                        disabled={status.is_training}
                        className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1 text-secondary-300">最大深度 (max_depth)</label>
                      <input
                        type="number"
                        min="3"
                        max="20"
                        value={config.max_depth}
                        onChange={(e) => setConfig(prev => ({ ...prev, max_depth: parseInt(e.target.value) }))}
                        disabled={status.is_training}
                        className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1 text-secondary-300">学習率 (learning_rate)</label>
                      <input
                        type="number"
                        min="0.01"
                        max="0.3"
                        step="0.01"
                        value={config.learning_rate}
                        onChange={(e) => setConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                        disabled={status.is_training}
                        className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                    <div>
                      <label className="block text-sm font-medium mb-1 text-secondary-300">早期停止ラウンド数</label>
                      <input
                        type="number"
                        min="10"
                        max="200"
                        step="10"
                        value={config.early_stopping_rounds}
                        onChange={(e) => setConfig(prev => ({ ...prev, early_stopping_rounds: parseInt(e.target.value) }))}
                        disabled={status.is_training}
                        className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1 text-secondary-300">ランダムシード</label>
                      <input
                        type="number"
                        min="1"
                        max="999"
                        value={config.random_state}
                        onChange={(e) => setConfig(prev => ({ ...prev, random_state: parseInt(e.target.value) }))}
                        disabled={status.is_training}
                        className="w-full bg-secondary-800 border border-secondary-700 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="save_model"
                    checked={config.save_model}
                    onChange={(e) => setConfig(prev => ({ ...prev, save_model: e.target.checked }))}
                    disabled={status.is_training}
                    className="h-4 w-4 text-primary-600 bg-secondary-800 border-secondary-700 rounded focus:ring-primary-500"
                  />
                  <label htmlFor="save_model" className="ml-2 block text-sm text-secondary-300">
                    モデルを保存する
                  </label>
                </div>
              </div>
            </div>

            {/* トレーニングステータス */}
            {status.is_training && (
              <div className="mt-6 p-4 bg-secondary-900 rounded-lg border border-secondary-700">
                <h3 className="text-lg font-bold mb-2 text-secondary-200">トレーニング状況</h3>
                <div className="flex items-center mb-2">
                  <div className={`mr-2 text-${getStatusInfo(status.status).color}-400`}>
                    {getStatusInfo(status.status).icon}
                  </div>
                  <p className="font-semibold text-secondary-300">{getStatusMessage(status.status)}</p>
                </div>
                <div className="w-full bg-secondary-700 rounded-full h-2.5">
                  <div
                    className="bg-primary-600 h-2.5 rounded-full"
                    style={{ width: `${status.progress}%` }}
                  ></div>
                </div>
                <p className="text-right text-sm mt-1 text-secondary-400">{status.progress.toFixed(1)}%</p>
                <p className="text-sm text-secondary-400 mt-2">{status.message}</p>
              </div>
            )}

            {/* 結果表示 */}
            {status.status === 'completed' && status.model_info && (
              <div className="mt-6 p-4 bg-green-900 bg-opacity-30 rounded-lg border border-green-700">
                <h3 className="text-lg font-bold text-green-300 mb-2">トレーニング完了</h3>
                <div className="grid grid-cols-2 gap-2 text-sm text-secondary-300">
                  <p>精度:</p><p className="font-mono">{(status.model_info.accuracy * 100).toFixed(2)}%</p>
                  <p>損失:</p><p className="font-mono">{status.model_info.loss.toFixed(4)}</p>
                  <p>特徴量数:</p><p className="font-mono">{status.model_info.feature_count}</p>
                  <p>学習サンプル数:</p><p className="font-mono">{status.model_info.training_samples}</p>
                  <p>モデルパス:</p><p className="font-mono text-xs break-all">{status.model_info.model_path}</p>
                </div>
              </div>
            )}
            {status.status === 'error' && (
              <div className="mt-6 p-4 bg-red-900 bg-opacity-30 rounded-lg border border-red-700">
                <h3 className="text-lg font-bold text-red-300 mb-2">エラー</h3>
                <p className="text-sm text-red-300">{status.error}</p>
              </div>
            )}
          </div>
        </div>

        {/* フッター */}
        <div className="flex items-center justify-end p-6 border-t border-secondary-700 bg-secondary-900 space-x-4">
          {status.is_training ? (
            <ActionButton
              variant="danger"
              onClick={handleStopTraining}
              icon={<StopCircle size={16} />}
            >
              停止
            </ActionButton>
          ) : (
            <>
              <ActionButton
                variant="secondary"
                onClick={onClose}
              >
                キャンセル
              </ActionButton>
              <ActionButton
                variant="primary"
                onClick={handleStartTraining}
                disabled={isLoading}
                icon={<Play size={16} />}
              >
                {isLoading ? '開始中...' : 'トレーニング開始'}
              </ActionButton>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default MLTrainingModal;
