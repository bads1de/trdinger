"""
戦略ショーケースサービス

オートストラテジー機能で生成された投資戦略のショーケース機能を提供するサービスクラス
GeneratedStrategyテーブルからデータを取得してStrategyShowcase形式で表示します
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import joinedload
from sqlalchemy import desc, asc, and_

from database.models import GeneratedStrategy, BacktestResult
from database.connection import get_db

logger = logging.getLogger(__name__)


class StrategyShowcaseService:
    """
    戦略ショーケースサービス

    オートストラテジー機能で生成された戦略をショーケース形式で表示します。
    GeneratedStrategyテーブルからデータを取得し、StrategyShowcase形式に変換します。
    """

    def __init__(self):
        """初期化"""
        # 戦略カテゴリ定義
        self.strategy_categories = {
            "trend_following": "トレンドフォロー",
            "mean_reversion": "逆張り",
            "breakout": "ブレイクアウト",
            "range_trading": "レンジ取引",
            "momentum": "モメンタム",
        }

        # リスクレベル定義
        self.risk_levels = ["low", "medium", "high"]

    def get_showcase_strategies(
        self,
        category: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        sort_by: str = "expected_return",
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """
        ショーケース戦略を取得（GeneratedStrategyテーブルから）

        Args:
            category: 戦略カテゴリでフィルタ
            risk_level: リスクレベルでフィルタ
            limit: 取得件数制限
            offset: オフセット
            sort_by: ソート項目
            sort_order: ソート順序（asc/desc）

        Returns:
            戦略データのリスト
        """
        try:
            with next(get_db()) as db:
                # GeneratedStrategyとBacktestResultをJOINして取得
                query = (
                    db.query(GeneratedStrategy)
                    .join(
                        BacktestResult,
                        GeneratedStrategy.backtest_result_id == BacktestResult.id,
                        isouter=True,
                    )
                    .options(joinedload(GeneratedStrategy.backtest_result))
                    .filter(GeneratedStrategy.fitness_score.isnot(None))
                )

                # フィルタ適用
                if category:
                    # gene_dataからカテゴリを抽出してフィルタ
                    query = query.filter(
                        GeneratedStrategy.gene_data.op("->>")('"category"') == category
                    )

                if risk_level:
                    # fitness_scoreからリスクレベルを推定してフィルタ
                    if risk_level == "low":
                        query = query.filter(GeneratedStrategy.fitness_score >= 0.7)
                    elif risk_level == "medium":
                        query = query.filter(
                            and_(
                                GeneratedStrategy.fitness_score >= 0.4,
                                GeneratedStrategy.fitness_score < 0.7,
                            )
                        )
                    elif risk_level == "high":
                        query = query.filter(GeneratedStrategy.fitness_score < 0.4)

                # ソート適用
                if sort_by == "expected_return" and sort_order.lower() == "desc":
                    query = query.order_by(desc(GeneratedStrategy.fitness_score))
                elif sort_by == "expected_return" and sort_order.lower() == "asc":
                    query = query.order_by(asc(GeneratedStrategy.fitness_score))
                else:
                    # デフォルトソート
                    query = query.order_by(desc(GeneratedStrategy.created_at))

                # ページネーション
                if limit:
                    query = query.limit(limit)
                if offset:
                    query = query.offset(offset)

                generated_strategies = query.all()

                # GeneratedStrategyをStrategyShowcase形式に変換
                showcase_strategies = []
                for strategy in generated_strategies:
                    showcase_data = self._convert_to_showcase_format(strategy)
                    if showcase_data:
                        showcase_strategies.append(showcase_data)

                return showcase_strategies

        except Exception as e:
            logger.error(f"ショーケース戦略取得エラー: {e}")
            raise

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """
        IDで戦略を取得（GeneratedStrategyテーブルから）

        Args:
            strategy_id: 戦略ID

        Returns:
            戦略データ
        """
        try:
            with next(get_db()) as db:
                strategy = (
                    db.query(GeneratedStrategy)
                    .options(joinedload(GeneratedStrategy.backtest_result))
                    .filter(GeneratedStrategy.id == strategy_id)
                    .first()
                )

                if strategy:
                    return self._convert_to_showcase_format(strategy)
                return None

        except Exception as e:
            logger.error(f"戦略取得エラー: {e}")
            raise

    def get_showcase_statistics(self) -> Dict[str, Any]:
        """
        ショーケース統計情報を取得（GeneratedStrategyテーブルから）

        Returns:
            統計情報
        """
        try:
            with next(get_db()) as db:
                strategies = (
                    db.query(GeneratedStrategy)
                    .options(joinedload(GeneratedStrategy.backtest_result))
                    .filter(GeneratedStrategy.fitness_score.isnot(None))
                    .all()
                )

                if not strategies:
                    return {
                        "total_strategies": 0,
                        "avg_return": 0,
                        "avg_sharpe_ratio": 0,
                        "avg_max_drawdown": 0,
                        "category_distribution": {},
                        "risk_distribution": {},
                    }

                # 統計計算用のデータを収集
                returns = []
                sharpe_ratios = []
                drawdowns = []
                category_dist = {}
                risk_dist = {}

                for strategy in strategies:
                    # バックテスト結果からパフォーマンス指標を取得
                    if strategy.backtest_result:
                        performance = strategy.backtest_result.performance_metrics or {}

                        if performance.get("total_return"):
                            returns.append(performance["total_return"])
                        if performance.get("sharpe_ratio"):
                            sharpe_ratios.append(performance["sharpe_ratio"])
                        if performance.get("max_drawdown"):
                            drawdowns.append(abs(performance["max_drawdown"]))

                    # カテゴリ分布
                    category = self._extract_category_from_gene(strategy.gene_data)
                    if category:
                        category_dist[category] = category_dist.get(category, 0) + 1

                    # リスク分布
                    risk_level = self._calculate_risk_level(strategy.fitness_score)
                    risk_dist[risk_level] = risk_dist.get(risk_level, 0) + 1

                return {
                    "total_strategies": len(strategies),
                    "avg_return": sum(returns) / len(returns) if returns else 0,
                    "avg_sharpe_ratio": (
                        sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
                    ),
                    "avg_max_drawdown": (
                        sum(drawdowns) / len(drawdowns) if drawdowns else 0
                    ),
                    "category_distribution": category_dist,
                    "risk_distribution": risk_dist,
                }

        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            raise

    def _convert_to_showcase_format(
        self, strategy: GeneratedStrategy
    ) -> Optional[Dict[str, Any]]:
        """
        GeneratedStrategyをStrategyShowcase形式に変換

        Args:
            strategy: GeneratedStrategy オブジェクト

        Returns:
            StrategyShowcase形式の辞書
        """
        try:
            gene_data = strategy.gene_data or {}

            # 基本情報の抽出
            name = self._generate_strategy_name(gene_data, strategy.id)
            description = self._generate_strategy_description(gene_data)
            category = self._extract_category_from_gene(gene_data)
            indicators = self._extract_indicators_from_gene(gene_data)
            parameters = self._extract_parameters_from_gene(gene_data)

            # パフォーマンス指標の抽出
            performance_metrics = self._extract_performance_metrics(strategy)

            # リスクレベルの計算
            risk_level = self._calculate_risk_level(strategy.fitness_score)

            # 推奨時間軸の決定
            recommended_timeframe = self._determine_recommended_timeframe(
                category, risk_level
            )

            return {
                "id": strategy.id,
                "name": name,
                "description": description,
                "category": category,
                "indicators": indicators,
                "parameters": parameters,
                "expected_return": performance_metrics.get("expected_return", 0.0),
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0.0),
                "max_drawdown": performance_metrics.get("max_drawdown", 0.0),
                "win_rate": performance_metrics.get("win_rate", 0.0),
                "gene_data": gene_data,
                "backtest_result_id": strategy.backtest_result_id,
                "risk_level": risk_level,
                "recommended_timeframe": recommended_timeframe,
                "is_active": True,
                "created_at": (
                    strategy.created_at.isoformat() if strategy.created_at else None
                ),
                "updated_at": (
                    strategy.created_at.isoformat() if strategy.created_at else None
                ),
            }

        except Exception as e:
            logger.error(f"戦略変換エラー: {e}")
            return None

    def get_showcase_statistics(self) -> Dict[str, Any]:
        """
        ショーケース統計情報を取得（GeneratedStrategyテーブルから）

        Returns:
            統計情報
        """
        try:
            with next(get_db()) as db:
                strategies = (
                    db.query(GeneratedStrategy)
                    .options(joinedload(GeneratedStrategy.backtest_result))
                    .filter(GeneratedStrategy.fitness_score.isnot(None))
                    .all()
                )

                if not strategies:
                    return {
                        "total_strategies": 0,
                        "avg_return": 0,
                        "avg_sharpe_ratio": 0,
                        "avg_max_drawdown": 0,
                        "category_distribution": {},
                        "risk_distribution": {},
                    }

                # 統計計算用のデータを収集
                returns = []
                sharpe_ratios = []
                drawdowns = []
                category_dist = {}
                risk_dist = {}

                for strategy in strategies:
                    # バックテスト結果からパフォーマンス指標を取得
                    if strategy.backtest_result:
                        performance = strategy.backtest_result.performance_metrics or {}

                        if performance.get("total_return"):
                            returns.append(performance["total_return"])
                        if performance.get("sharpe_ratio"):
                            sharpe_ratios.append(performance["sharpe_ratio"])
                        if performance.get("max_drawdown"):
                            drawdowns.append(abs(performance["max_drawdown"]))

                    # カテゴリ分布
                    category = self._extract_category_from_gene(strategy.gene_data)
                    if category:
                        category_dist[category] = category_dist.get(category, 0) + 1

                    # リスク分布
                    risk_level = self._calculate_risk_level(strategy.fitness_score)
                    risk_dist[risk_level] = risk_dist.get(risk_level, 0) + 1

                return {
                    "total_strategies": len(strategies),
                    "avg_return": sum(returns) / len(returns) if returns else 0,
                    "avg_sharpe_ratio": (
                        sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
                    ),
                    "avg_max_drawdown": (
                        sum(drawdowns) / len(drawdowns) if drawdowns else 0
                    ),
                    "category_distribution": category_dist,
                    "risk_distribution": risk_dist,
                }

        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            raise

    # ヘルパーメソッド群
    def _generate_strategy_name(
        self, gene_data: Dict[str, Any], strategy_id: int
    ) -> str:
        """戦略名を生成"""
        indicators = self._extract_indicators_from_gene(gene_data)
        category = self._extract_category_from_gene(gene_data)

        if len(indicators) == 1:
            return f"{indicators[0]} {category.title()} #{strategy_id}"
        elif len(indicators) == 2:
            return f"{indicators[0]}-{indicators[1]} Combo #{strategy_id}"
        else:
            return f"Multi-Signal Strategy #{strategy_id}"

    def _generate_strategy_description(self, gene_data: Dict[str, Any]) -> str:
        """戦略説明を生成"""
        indicators = self._extract_indicators_from_gene(gene_data)
        category = self._extract_category_from_gene(gene_data)

        category_descriptions = {
            "trend_following": "トレンドフォロー戦略",
            "mean_reversion": "逆張り戦略",
            "breakout": "ブレイクアウト戦略",
            "range_trading": "レンジ取引戦略",
            "momentum": "モメンタム戦略",
        }

        base_desc = category_descriptions.get(category, "投資戦略")
        indicator_list = "、".join(indicators)

        return f"{indicator_list}を使用した{base_desc}"

    def _extract_category_from_gene(self, gene_data: Dict[str, Any]) -> str:
        """遺伝子データからカテゴリを抽出"""
        # gene_dataから直接カテゴリが取得できる場合
        if "category" in gene_data:
            return gene_data["category"]

        # 指標から推定
        indicators = self._extract_indicators_from_gene(gene_data)

        if any(ind in ["SMA", "EMA", "MACD", "ADX"] for ind in indicators):
            return "trend_following"
        elif any(
            ind in ["RSI", "BB", "STOCH", "CCI", "WILLIAMS"] for ind in indicators
        ):
            return "mean_reversion"
        elif any(ind in ["ATR"] for ind in indicators):
            return "breakout"
        else:
            return "momentum"

    def _extract_indicators_from_gene(self, gene_data: Dict[str, Any]) -> List[str]:
        """遺伝子データから指標リストを抽出"""
        indicators = []

        if "indicators" in gene_data:
            if isinstance(gene_data["indicators"], list):
                for indicator in gene_data["indicators"]:
                    if isinstance(indicator, dict) and "name" in indicator:
                        indicators.append(indicator["name"])
                    elif isinstance(indicator, str):
                        indicators.append(indicator)

        return indicators or ["SMA"]  # デフォルト

    def _extract_parameters_from_gene(
        self, gene_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """遺伝子データからパラメータを抽出"""
        parameters = {}

        if "indicators" in gene_data:
            for indicator in gene_data["indicators"]:
                if isinstance(indicator, dict) and "name" in indicator:
                    name = indicator["name"]
                    params = {k: v for k, v in indicator.items() if k != "name"}
                    parameters[name] = params

        return parameters

    def _extract_performance_metrics(
        self, strategy: GeneratedStrategy
    ) -> Dict[str, float]:
        """戦略からパフォーマンス指標を抽出"""
        metrics = {
            "expected_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

        if strategy.backtest_result and strategy.backtest_result.performance_metrics:
            performance = strategy.backtest_result.performance_metrics

            metrics["expected_return"] = performance.get("total_return", 0.0) or 0.0
            metrics["sharpe_ratio"] = performance.get("sharpe_ratio", 0.0) or 0.0
            metrics["max_drawdown"] = abs(performance.get("max_drawdown", 0.0) or 0.0)
            metrics["win_rate"] = (performance.get("win_rate", 0.0) or 0.0) * 100

        return metrics

    def _calculate_risk_level(self, fitness_score: Optional[float]) -> str:
        """フィットネススコアからリスクレベルを計算"""
        if fitness_score is None:
            return "medium"

        if fitness_score >= 0.7:
            return "low"
        elif fitness_score >= 0.4:
            return "medium"
        else:
            return "high"

    def _determine_recommended_timeframe(self, category: str, risk_level: str) -> str:
        """カテゴリとリスクレベルから推奨時間軸を決定"""
        if category == "momentum" and risk_level == "high":
            return "15m"
        elif category == "trend_following" and risk_level == "low":
            return "1d"
        elif category == "breakout":
            return "1h"
        elif category == "range_trading":
            return "4h"
        else:
            return "1h"

    # 以下は廃止予定のメソッド（後方互換性のため残す）
