/**
 * 投資戦略ショーケースページ
 *
 * 自動生成された30個の投資戦略を一覧表示し、
 * フィルタリング・ソート・詳細表示機能を提供します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React, { useState, useEffect } from "react";
import StrategyCard from "./components/StrategyCard";
import StrategyFilters from "./components/StrategyFilters";
import StrategyModal from "./components/StrategyModal";
import {
  StrategyShowcase,
  StrategyFilter,
  Pagination,
  ShowcaseStatistics,
  StrategyCategory,
  RiskLevel,
  LoadingState,
  ErrorState,
} from "@/types/strategy-showcase";

/**
 * ショーケースページコンポーネント
 */
const StrategiesShowcasePage: React.FC = () => {
  // 状態管理
  const [strategies, setStrategies] = useState<StrategyShowcase[]>([]);
  const [statistics, setStatistics] = useState<ShowcaseStatistics | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<
    StrategyShowcase | undefined
  >();
  const [isModalOpen, setIsModalOpen] = useState(false);

  // フィルター・ページネーション
  const [filter, setFilter] = useState<StrategyFilter>({});
  const [pagination, setPagination] = useState<Pagination>({
    offset: 0,
    sort_by: "expected_return",
    sort_order: "desc",
    limit: 12,
  });

  // メタデータ
  const [categories, setCategories] = useState<
    Record<StrategyCategory, string>
  >({} as Record<StrategyCategory, string>);
  const [riskLevels, setRiskLevels] = useState<Record<RiskLevel, string>>(
    {} as Record<RiskLevel, string>
  );

  // ローディング・エラー状態
  const [loading, setLoading] = useState<LoadingState>({
    strategies: false,
    statistics: false,
    generation: false,
    detail: false,
  });

  const [errors, setErrors] = useState<ErrorState>({});

  // 初期データ取得
  useEffect(() => {
    fetchInitialData();
  }, []);

  // フィルター・ページネーション変更時にデータ再取得
  useEffect(() => {
    fetchStrategies();
  }, [filter, pagination]);

  /**
   * 初期データを取得
   */
  const fetchInitialData = async () => {
    await Promise.all([
      fetchCategories(),
      fetchRiskLevels(),
      fetchStatistics(),
      fetchStrategies(),
    ]);
  };

  /**
   * 戦略一覧を取得
   */
  const fetchStrategies = async () => {
    try {
      setLoading((prev) => ({ ...prev, strategies: true }));
      setErrors((prev) => ({ ...prev, strategies: undefined }));

      // クエリパラメータを構築
      const params = new URLSearchParams();

      if (filter.category) params.append("category", filter.category);
      if (filter.risk_level) params.append("risk_level", filter.risk_level);
      if (pagination.limit) params.append("limit", pagination.limit.toString());
      params.append("offset", pagination.offset.toString());
      params.append("sort_by", pagination.sort_by);
      params.append("sort_order", pagination.sort_order);

      const response = await fetch(
        `/api/strategies/showcase?${params.toString()}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        setStrategies(data.strategies);
      } else {
        throw new Error(data.message || "戦略取得に失敗しました");
      }
    } catch (error) {
      console.error("戦略取得エラー:", error);
      setErrors((prev) => ({
        ...prev,
        strategies:
          error instanceof Error ? error.message : "戦略取得に失敗しました",
      }));
    } finally {
      setLoading((prev) => ({ ...prev, strategies: false }));
    }
  };

  /**
   * 統計情報を取得
   */
  const fetchStatistics = async () => {
    try {
      setLoading((prev) => ({ ...prev, statistics: true }));
      setErrors((prev) => ({ ...prev, statistics: undefined }));

      const response = await fetch(`/api/strategies/showcase/stats`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        setStatistics(data.statistics);
      } else {
        throw new Error(data.message || "統計情報取得に失敗しました");
      }
    } catch (error) {
      console.error("統計情報取得エラー:", error);
      setErrors((prev) => ({
        ...prev,
        statistics:
          error instanceof Error ? error.message : "統計情報取得に失敗しました",
      }));
    } finally {
      setLoading((prev) => ({ ...prev, statistics: false }));
    }
  };

  /**
   * カテゴリ一覧を取得
   */
  const fetchCategories = async () => {
    try {
      const response = await fetch(`/api/strategies/categories`);
      const data = await response.json();

      if (data.success) {
        setCategories(data.categories);
      }
    } catch (error) {
      console.error("カテゴリ取得エラー:", error);
    }
  };

  /**
   * リスクレベル一覧を取得
   */
  const fetchRiskLevels = async () => {
    try {
      const response = await fetch(`/api/strategies/risk-levels`);
      const data = await response.json();

      if (data.success) {
        setRiskLevels(data.risk_levels);
      }
    } catch (error) {
      console.error("リスクレベル取得エラー:", error);
    }
  };

  /**
   * 戦略生成を実行（オートストラテジー機能へのリダイレクト）
   */
  const generateStrategies = async () => {
    // オートストラテジー機能へのリダイレクト
    if (
      confirm(
        "戦略生成機能はオートストラテジー機能に移行しました。オートストラテジーページに移動しますか？"
      )
    ) {
      window.location.href = "/auto-strategy";
    }
  };

  /**
   * 戦略詳細を表示
   */
  const handleViewDetail = (strategy: StrategyShowcase) => {
    setSelectedStrategy(strategy);
    setIsModalOpen(true);
  };

  /**
   * モーダルを閉じる
   */
  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedStrategy(undefined);
  };

  /**
   * フィルター変更ハンドラ
   */
  const handleFilterChange = (newFilter: StrategyFilter) => {
    setFilter(newFilter);
    setPagination((prev) => ({ ...prev, offset: 0 })); // ページをリセット
  };

  /**
   * ページネーション変更ハンドラ
   */
  const handlePaginationChange = (newPagination: Pagination) => {
    setPagination(newPagination);
  };

  return (
    <div className="min-h-screen bg-secondary-50 dark:bg-secondary-950 animate-fade-in">
      {/* ヘッダー */}
      <div className="enterprise-card border-0 rounded-none border-b border-secondary-200 dark:border-secondary-700 shadow-enterprise-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="animate-slide-up">
              <h1 className="text-3xl font-bold text-gradient">
                🚀 投資戦略ショーケース
              </h1>
              <p className="mt-2 text-base text-secondary-600 dark:text-secondary-400">
                AI生成による30種類の多様な投資戦略を比較・検討
              </p>

              {/* 統計サマリー */}
              {statistics && (
                <div className="mt-4 flex items-center gap-4">
                  <span className="badge-primary">
                    総戦略数: {statistics.total_strategies}
                  </span>
                  <span className="badge-success">
                    平均リターン: {statistics.avg_return.toFixed(1)}%
                  </span>
                  <span className="badge-info">
                    平均シャープレシオ: {statistics.avg_sharpe_ratio.toFixed(2)}
                  </span>
                </div>
              )}
            </div>

            {/* オートストラテジーへのリンクボタン */}
            <div className="flex items-center gap-3">
              <button onClick={generateStrategies} className="btn-primary">
                🚀 オートストラテジーで生成
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* メインコンテンツ */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* フィルター */}
        <StrategyFilters
          filter={filter}
          pagination={pagination}
          onFilterChange={handleFilterChange}
          onPaginationChange={handlePaginationChange}
          categories={categories}
          riskLevels={riskLevels}
        />

        {/* エラー表示 */}
        {errors.strategies && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-700 rounded-lg">
            <p className="text-red-400">エラー: {errors.strategies}</p>
          </div>
        )}

        {/* 戦略グリッド */}
        {loading.strategies ? (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
            <span className="ml-3 text-secondary-600 dark:text-secondary-400">
              戦略を読み込み中...
            </span>
          </div>
        ) : strategies.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {strategies.map((strategy) => (
              <StrategyCard
                key={strategy.id}
                strategy={strategy}
                onViewDetail={handleViewDetail}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-secondary-600 dark:text-secondary-400">
              戦略が見つかりませんでした。戦略を生成してください。
            </p>
          </div>
        )}
      </div>

      {/* 戦略詳細モーダル */}
      <StrategyModal
        strategy={selectedStrategy}
        isOpen={isModalOpen}
        onClose={handleCloseModal}
      />
    </div>
  );
};

export default StrategiesShowcasePage;
