/**
 * 保存済み戦略コンポーネント
 *
 * 保存済み戦略の一覧表示、編集、削除機能を提供します。
 */

"use client";

import React, { useState, useEffect } from "react";
import { InputField } from "@/components/common/InputField";
import ApiButton from "@/components/button/ApiButton";
import Modal from "@/components/common/Modal";

interface UserStrategy {
  id: number;
  name: string;
  description?: string;
  strategy_config: any;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface SavedStrategiesProps {
  onLoadStrategy?: (strategy: UserStrategy) => void;
  onEditStrategy?: (strategy: UserStrategy) => void;
}

/**
 * 保存済み戦略コンポーネント
 */
const SavedStrategies: React.FC<SavedStrategiesProps> = ({
  onLoadStrategy,
  onEditStrategy,
}) => {
  // 状態管理
  const [strategies, setStrategies] = useState<UserStrategy[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedStrategy, setSelectedStrategy] = useState<UserStrategy | null>(
    null
  );
  const [showDeleteModal, setShowDeleteModal] = useState<boolean>(false);
  const [showDetailModal, setShowDetailModal] = useState<boolean>(false);
  const [deleting, setDeleting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // 戦略一覧を取得
  const fetchStrategies = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch("/api/strategy-builder/strategies");

      if (!response.ok) {
        throw new Error("戦略一覧の取得に失敗しました");
      }

      const data = await response.json();

      if (data.success) {
        setStrategies(data.data.strategies);
      } else {
        throw new Error(data.message || "戦略一覧の取得に失敗しました");
      }
    } catch (err) {
      console.error("戦略取得エラー:", err);
      setError(
        err instanceof Error ? err.message : "不明なエラーが発生しました"
      );

      // フォールバック用のダミーデータ
      setStrategies([
        {
          id: 1,
          name: "SMA クロス戦略",
          description: "短期と長期の移動平均のクロスオーバーを利用した戦略",
          strategy_config: {
            indicators: [
              { type: "SMA", parameters: { period: 20 } },
              { type: "SMA", parameters: { period: 50 } },
            ],
            entry_conditions: [
              {
                type: "crossover",
                indicator1: "SMA_20",
                indicator2: "SMA_50",
                operator: "above",
              },
            ],
            exit_conditions: [
              {
                type: "crossover",
                indicator1: "SMA_20",
                indicator2: "SMA_50",
                operator: "below",
              },
            ],
          },
          is_active: true,
          created_at: "2024-01-15T10:30:00Z",
          updated_at: "2024-01-15T10:30:00Z",
        },
        {
          id: 2,
          name: "RSI 逆張り戦略",
          description: "RSIの過買い・過売りシグナルを利用した逆張り戦略",
          strategy_config: {
            indicators: [{ type: "RSI", parameters: { period: 14 } }],
            entry_conditions: [
              { type: "threshold", indicator: "RSI", operator: "<", value: 30 },
            ],
            exit_conditions: [
              { type: "threshold", indicator: "RSI", operator: ">", value: 70 },
            ],
          },
          is_active: true,
          created_at: "2024-01-14T15:45:00Z",
          updated_at: "2024-01-14T15:45:00Z",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // 戦略を削除
  const deleteStrategy = async (strategyId: number) => {
    try {
      setDeleting(true);

      const response = await fetch(
        `/api/strategy-builder/strategies/${strategyId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error("戦略の削除に失敗しました");
      }

      const data = await response.json();

      if (data.success) {
        // 戦略一覧から削除
        setStrategies((prev) => prev.filter((s) => s.id !== strategyId));
        setShowDeleteModal(false);
        setSelectedStrategy(null);
      } else {
        throw new Error(data.message || "戦略の削除に失敗しました");
      }
    } catch (err) {
      console.error("戦略削除エラー:", err);
      alert(err instanceof Error ? err.message : "戦略の削除に失敗しました");
    } finally {
      setDeleting(false);
    }
  };

  // 初期化
  useEffect(() => {
    fetchStrategies();
  }, []);

  // 検索フィルタリング
  const filteredStrategies = strategies.filter(
    (strategy) =>
      strategy.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (strategy.description &&
        strategy.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  // 戦略の詳細情報を取得
  const getStrategyStats = (strategy: UserStrategy) => {
    const config = strategy.strategy_config;
    return {
      indicators: config.indicators?.length || 0,
      entryConditions: config.entry_conditions?.length || 0,
      exitConditions: config.exit_conditions?.length || 0,
    };
  };

  // 日付フォーマット
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("ja-JP", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (loading) {
    return (
      <div className="bg-secondary-950 rounded-lg p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-300">戦略を読み込み中...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-secondary-950 rounded-lg p-6">
      {/* エラー表示 */}
      {error && (
        <div className="mb-4 p-4 bg-red-900/50 border border-red-700 rounded-lg">
          <p className="text-red-300 text-sm">⚠️ {error}</p>
          <p className="text-red-400 text-xs mt-1">
            ダミーデータを表示しています
          </p>
        </div>
      )}

      {/* ヘッダー */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">保存済み戦略</h3>
          <p className="text-gray-400 text-sm">作成済みの戦略を管理できます</p>
        </div>
        <button
          onClick={fetchStrategies}
          className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
        >
          <svg
            className="w-4 h-4 inline mr-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
          更新
        </button>
      </div>

      {/* 検索フィールド */}
      <div className="mb-6">
        <InputField
          label="戦略を検索"
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="戦略名や説明で検索..."
          className="bg-gray-700"
        />
      </div>

      {/* 戦略一覧 */}
      {filteredStrategies.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredStrategies.map((strategy) => {
            const stats = getStrategyStats(strategy);

            return (
              <div
                key={strategy.id}
                className="border border-gray-600 rounded-lg p-4 bg-gray-700 hover:border-gray-500 transition-colors"
              >
                {/* 戦略ヘッダー */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h5 className="font-medium text-white truncate">
                      {strategy.name}
                    </h5>
                    <p className="text-xs text-gray-400 mt-1">
                      ID: {strategy.id}
                    </p>
                  </div>
                  <span
                    className={`
                    px-2 py-1 text-xs rounded
                    ${
                      strategy.is_active
                        ? "bg-green-600 text-white"
                        : "bg-gray-600 text-gray-300"
                    }
                  `}
                  >
                    {strategy.is_active ? "有効" : "無効"}
                  </span>
                </div>

                {/* 戦略説明 */}
                {strategy.description && (
                  <p className="text-sm text-gray-300 mb-3 line-clamp-2">
                    {strategy.description}
                  </p>
                )}

                {/* 戦略統計 */}
                <div className="grid grid-cols-3 gap-2 mb-4">
                  <div className="text-center">
                    <p className="text-xs text-gray-400">指標</p>
                    <p className="text-sm font-medium text-white">
                      {stats.indicators}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-gray-400">エントリー</p>
                    <p className="text-sm font-medium text-white">
                      {stats.entryConditions}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-gray-400">イグジット</p>
                    <p className="text-sm font-medium text-white">
                      {stats.exitConditions}
                    </p>
                  </div>
                </div>

                {/* 作成日時 */}
                <p className="text-xs text-gray-500 mb-4">
                  作成: {formatDate(strategy.created_at)}
                </p>

                {/* アクションボタン */}
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setSelectedStrategy(strategy);
                      setShowDetailModal(true);
                    }}
                    className="flex-1 px-3 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
                  >
                    詳細
                  </button>
                  {onLoadStrategy && (
                    <button
                      onClick={() => onLoadStrategy(strategy)}
                      className="flex-1 px-3 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700 transition-colors"
                    >
                      読込
                    </button>
                  )}
                  {onEditStrategy && (
                    <button
                      onClick={() => onEditStrategy(strategy)}
                      className="px-3 py-2 bg-gray-600 text-white text-sm rounded hover:bg-gray-500 transition-colors"
                    >
                      編集
                    </button>
                  )}
                  <button
                    onClick={() => {
                      setSelectedStrategy(strategy);
                      setShowDeleteModal(true);
                    }}
                    className="px-3 py-2 bg-red-600 text-white text-sm rounded hover:bg-red-700 transition-colors"
                  >
                    削除
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-12">
          <svg
            className="w-16 h-16 text-gray-500 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
            />
          </svg>
          <p className="text-gray-400 text-lg">
            {searchQuery
              ? "検索条件に一致する戦略が見つかりません"
              : "保存済み戦略がありません"}
          </p>
          <p className="text-gray-500 text-sm mt-2">
            {searchQuery
              ? "検索条件を変更してください"
              : "ストラテジービルダーで新しい戦略を作成してください"}
          </p>
        </div>
      )}

      {/* 削除確認モーダル */}
      <Modal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        title="戦略の削除"
      >
        {selectedStrategy && (
          <div>
            <p className="text-gray-300 mb-4">
              戦略「{selectedStrategy.name}」を削除しますか？
            </p>
            <p className="text-red-400 text-sm mb-6">
              この操作は取り消せません。
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowDeleteModal(false)}
                className="flex-1 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-500 transition-colors"
              >
                キャンセル
              </button>
              <ApiButton
                onClick={() => deleteStrategy(selectedStrategy.id)}
                loading={deleting}
                variant="error"
                className="flex-1"
              >
                削除
              </ApiButton>
            </div>
          </div>
        )}
      </Modal>

      {/* 詳細表示モーダル */}
      <Modal
        isOpen={showDetailModal}
        onClose={() => setShowDetailModal(false)}
        title="戦略詳細"
        size="lg"
      >
        {selectedStrategy && (
          <div className="space-y-4">
            <div>
              <h5 className="font-medium text-white mb-2">
                {selectedStrategy.name}
              </h5>
              {selectedStrategy.description && (
                <p className="text-gray-300 text-sm">
                  {selectedStrategy.description}
                </p>
              )}
            </div>

            <div className="bg-secondary-950 p-4 rounded">
              <h6 className="font-medium text-white mb-2">戦略設定 (JSON)</h6>
              <pre className="text-xs text-gray-300 overflow-auto max-h-64">
                {JSON.stringify(selectedStrategy.strategy_config, null, 2)}
              </pre>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-400">作成日時</p>
                <p className="text-white">
                  {formatDate(selectedStrategy.created_at)}
                </p>
              </div>
              <div>
                <p className="text-gray-400">更新日時</p>
                <p className="text-white">
                  {formatDate(selectedStrategy.updated_at)}
                </p>
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default SavedStrategies;
