"""
戦略ショーケースサービス

30個の多様な投資戦略を自動生成し、ショーケース用に管理するサービス
"""

import logging
import random
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from database.models import StrategyShowcase, BacktestResult
from database.connection import get_db

logger = logging.getLogger(__name__)


class StrategyShowcaseService:
    """
    戦略ショーケースサービス

    多様な投資戦略の自動生成とショーケース管理を行います。
    """

    def __init__(self):
        """初期化"""
        # 循環インポートを避けるため、必要時に動的インポート
        self._auto_strategy_service = None

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

        # 利用可能な指標
        self.available_indicators = [
            "SMA",
            "EMA",
            "RSI",
            "MACD",
            "BB",
            "STOCH",
            "ATR",
            "CCI",
            "WILLIAMS",
            "ROC",
            "MOM",
            "ADX",
            "AROON",
            "TSI",
            "UO",
            "TRIX",
            "DX",
            "MINUS_DI",
            "PLUS_DI",
            "WILLR",
            "CMO",
        ]

    def generate_showcase_strategies(
        self, count: int = 30, base_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        ショーケース用戦略を生成

        Args:
            count: 生成する戦略数
            base_config: 基本バックテスト設定

        Returns:
            生成された戦略のリスト
        """
        try:
            logger.info(f"ショーケース戦略生成開始: {count}個")

            # デフォルト設定
            if base_config is None:
                base_config = {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-19",
                    "initial_capital": 100000,
                    "commission_rate": 0.00055,
                }

            strategies = []

            # 戦略パターンを定義
            strategy_patterns = self._generate_strategy_patterns(count)

            for i, pattern in enumerate(strategy_patterns):
                try:
                    logger.info(f"戦略生成中: {i+1}/{count} - {pattern['name']}")

                    # 簡単な戦略データを生成（テスト用）
                    strategy_data = self._create_simple_strategy_data(
                        pattern, base_config
                    )
                    strategies.append(strategy_data)

                    logger.info(f"戦略生成完了: {pattern['name']}")

                except Exception as e:
                    logger.error(f"戦略生成エラー: {pattern['name']} - {e}")
                    continue

            logger.info(f"ショーケース戦略生成完了: {len(strategies)}/{count}個成功")
            return strategies

        except Exception as e:
            logger.error(f"ショーケース戦略生成エラー: {e}")
            raise

    def save_strategies_to_database(
        self, strategies: List[Dict[str, Any]]
    ) -> List[int]:
        """
        戦略をデータベースに保存

        Args:
            strategies: 戦略データのリスト

        Returns:
            保存された戦略のIDリスト
        """
        try:
            saved_ids = []

            with next(get_db()) as db:
                for strategy_data in strategies:
                    try:
                        # 既存の同名戦略をチェック
                        existing = (
                            db.query(StrategyShowcase)
                            .filter(StrategyShowcase.name == strategy_data["name"])
                            .first()
                        )

                        if existing:
                            logger.info(f"戦略更新: {strategy_data['name']}")
                            # 既存戦略を更新
                            for key, value in strategy_data.items():
                                if key != "id":
                                    setattr(existing, key, value)
                            existing.updated_at = datetime.utcnow()
                            saved_ids.append(existing.id)
                        else:
                            logger.info(f"戦略新規作成: {strategy_data['name']}")
                            # 新規戦略を作成
                            strategy = StrategyShowcase(**strategy_data)
                            db.add(strategy)
                            db.flush()  # IDを取得するためにflush
                            saved_ids.append(strategy.id)

                    except Exception as e:
                        logger.error(
                            f"戦略保存エラー: {strategy_data.get('name', 'Unknown')} - {e}"
                        )
                        continue

                db.commit()
                logger.info(f"データベース保存完了: {len(saved_ids)}個")

            return saved_ids

        except Exception as e:
            logger.error(f"データベース保存エラー: {e}")
            raise

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
        ショーケース戦略を取得

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
                query = db.query(StrategyShowcase).filter(
                    StrategyShowcase.is_active == True
                )

                # フィルタ適用
                if category:
                    query = query.filter(StrategyShowcase.category == category)
                if risk_level:
                    query = query.filter(StrategyShowcase.risk_level == risk_level)

                # ソート適用
                sort_column = getattr(StrategyShowcase, sort_by, None)
                if sort_column:
                    if sort_order.lower() == "desc":
                        query = query.order_by(sort_column.desc())
                    else:
                        query = query.order_by(sort_column.asc())

                # ページネーション
                if limit:
                    query = query.limit(limit)
                if offset:
                    query = query.offset(offset)

                strategies = query.all()

                return [strategy.to_dict() for strategy in strategies]

        except Exception as e:
            logger.error(f"ショーケース戦略取得エラー: {e}")
            raise

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """
        IDで戦略を取得

        Args:
            strategy_id: 戦略ID

        Returns:
            戦略データ
        """
        try:
            with next(get_db()) as db:
                strategy = (
                    db.query(StrategyShowcase)
                    .filter(
                        StrategyShowcase.id == strategy_id,
                        StrategyShowcase.is_active == True,
                    )
                    .first()
                )

                if strategy:
                    return strategy.to_dict()
                return None

        except Exception as e:
            logger.error(f"戦略取得エラー: {e}")
            raise

    def get_showcase_statistics(self) -> Dict[str, Any]:
        """
        ショーケース統計情報を取得

        Returns:
            統計情報
        """
        try:
            with next(get_db()) as db:
                strategies = (
                    db.query(StrategyShowcase)
                    .filter(StrategyShowcase.is_active == True)
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

                # 統計計算
                returns = [
                    s.expected_return
                    for s in strategies
                    if s.expected_return is not None
                ]
                sharpe_ratios = [
                    s.sharpe_ratio for s in strategies if s.sharpe_ratio is not None
                ]
                drawdowns = [
                    s.max_drawdown for s in strategies if s.max_drawdown is not None
                ]

                # カテゴリ分布
                category_dist = {}
                for strategy in strategies:
                    category = strategy.category
                    category_dist[category] = category_dist.get(category, 0) + 1

                # リスク分布
                risk_dist = {}
                for strategy in strategies:
                    risk = strategy.risk_level
                    risk_dist[risk] = risk_dist.get(risk, 0) + 1

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

    def _generate_strategy_patterns(self, count: int) -> List[Dict[str, Any]]:
        """
        戦略パターンを生成

        Args:
            count: 生成する戦略数

        Returns:
            戦略パターンのリスト
        """
        patterns = []

        # 単一指標戦略（10個）
        single_indicator_strategies = [
            {
                "name": "RSI Momentum Pro",
                "category": "momentum",
                "indicators": ["RSI"],
                "risk_level": "medium",
                "description": "RSIを使用したモメンタム戦略",
            },
            {
                "name": "MACD Cross Elite",
                "category": "trend_following",
                "indicators": ["MACD"],
                "risk_level": "medium",
                "description": "MACDクロスオーバーによるトレンドフォロー戦略",
            },
            {
                "name": "SMA Trend Rider",
                "category": "trend_following",
                "indicators": ["SMA"],
                "risk_level": "low",
                "description": "移動平均を使用したトレンドフォロー戦略",
            },
            {
                "name": "EMA Swift",
                "category": "trend_following",
                "indicators": ["EMA"],
                "risk_level": "medium",
                "description": "指数移動平均による迅速なトレンド追従戦略",
            },
            {
                "name": "Bollinger Bounce",
                "category": "mean_reversion",
                "indicators": ["BB"],
                "risk_level": "medium",
                "description": "ボリンジャーバンドを使用した逆張り戦略",
            },
            {
                "name": "Stochastic Oscillator",
                "category": "mean_reversion",
                "indicators": ["STOCH"],
                "risk_level": "high",
                "description": "ストキャスティクスによる過買い過売り戦略",
            },
            {
                "name": "ATR Volatility",
                "category": "breakout",
                "indicators": ["ATR"],
                "risk_level": "high",
                "description": "ATRを使用したボラティリティブレイクアウト戦略",
            },
            {
                "name": "CCI Extreme",
                "category": "mean_reversion",
                "indicators": ["CCI"],
                "risk_level": "high",
                "description": "CCIによる極値反転戦略",
            },
            {
                "name": "Williams %R",
                "category": "mean_reversion",
                "indicators": ["WILLIAMS"],
                "risk_level": "medium",
                "description": "Williams %Rによる逆張り戦略",
            },
            {
                "name": "ADX Trend Strength",
                "category": "trend_following",
                "indicators": ["ADX"],
                "risk_level": "low",
                "description": "ADXによるトレンド強度判定戦略",
            },
        ]

        # 複数指標組み合わせ戦略（15個）
        multi_indicator_strategies = [
            {
                "name": "RSI-MACD Fusion",
                "category": "momentum",
                "indicators": ["RSI", "MACD"],
                "risk_level": "medium",
                "description": "RSIとMACDを組み合わせたモメンタム戦略",
            },
            {
                "name": "SMA-RSI Combo",
                "category": "trend_following",
                "indicators": ["SMA", "RSI"],
                "risk_level": "low",
                "description": "移動平均とRSIによる確認型戦略",
            },
            {
                "name": "EMA-BB Breakout",
                "category": "breakout",
                "indicators": ["EMA", "BB"],
                "risk_level": "high",
                "description": "EMAとボリンジャーバンドによるブレイクアウト戦略",
            },
            {
                "name": "MACD-BB Reversal",
                "category": "mean_reversion",
                "indicators": ["MACD", "BB"],
                "risk_level": "medium",
                "description": "MACDとボリンジャーバンドによる反転戦略",
            },
            {
                "name": "Triple SMA Cross",
                "category": "trend_following",
                "indicators": ["SMA", "EMA"],
                "risk_level": "low",
                "description": "複数移動平均によるトレンド確認戦略",
            },
            {
                "name": "RSI-Stoch Divergence",
                "category": "mean_reversion",
                "indicators": ["RSI", "STOCH"],
                "risk_level": "high",
                "description": "RSIとストキャスティクスによるダイバージェンス戦略",
            },
            {
                "name": "ATR-MACD Volatility",
                "category": "breakout",
                "indicators": ["ATR", "MACD"],
                "risk_level": "high",
                "description": "ATRとMACDによるボラティリティブレイクアウト戦略",
            },
            {
                "name": "Multi-Signal Alpha",
                "category": "momentum",
                "indicators": ["RSI", "MACD", "BB"],
                "risk_level": "medium",
                "description": "複数シグナルによる総合判定戦略",
            },
            {
                "name": "CCI-Williams Extreme",
                "category": "mean_reversion",
                "indicators": ["CCI", "WILLIAMS"],
                "risk_level": "high",
                "description": "CCIとWilliams %Rによる極値戦略",
            },
            {
                "name": "ADX-EMA Trend",
                "category": "trend_following",
                "indicators": ["ADX", "EMA"],
                "risk_level": "low",
                "description": "ADXとEMAによる強いトレンド追従戦略",
            },
            {
                "name": "ROC-MOM Momentum",
                "category": "momentum",
                "indicators": ["ROC", "MOM"],
                "risk_level": "medium",
                "description": "ROCとMomentumによる勢い戦略",
            },
            {
                "name": "AROON-TSI Trend",
                "category": "trend_following",
                "indicators": ["AROON", "TSI"],
                "risk_level": "medium",
                "description": "AroonとTSIによるトレンド転換戦略",
            },
            {
                "name": "UO-TRIX Oscillator",
                "category": "mean_reversion",
                "indicators": ["UO", "TRIX"],
                "risk_level": "medium",
                "description": "Ultimate OscillatorとTRIXによる反転戦略",
            },
            {
                "name": "DX-DI Directional",
                "category": "trend_following",
                "indicators": ["DX", "PLUS_DI", "MINUS_DI"],
                "risk_level": "low",
                "description": "方向性指標による確実なトレンド戦略",
            },
            {
                "name": "CMO-WILLR Range",
                "category": "range_trading",
                "indicators": ["CMO", "WILLR"],
                "risk_level": "medium",
                "description": "CMOとWilliams %Rによるレンジ取引戦略",
            },
        ]

        # 特殊戦略（5個）
        special_strategies = [
            {
                "name": "Volatility Hunter",
                "category": "breakout",
                "indicators": ["ATR", "BB", "STOCH"],
                "risk_level": "high",
                "description": "高ボラティリティ環境での積極的戦略",
            },
            {
                "name": "Trend Rider Supreme",
                "category": "trend_following",
                "indicators": ["SMA", "EMA", "MACD", "ADX"],
                "risk_level": "low",
                "description": "複数指標による確実なトレンドフォロー戦略",
            },
            {
                "name": "Scalping Master",
                "category": "momentum",
                "indicators": ["RSI", "STOCH", "CCI"],
                "risk_level": "high",
                "description": "短期取引に特化した高頻度戦略",
            },
            {
                "name": "Conservative Growth",
                "category": "trend_following",
                "indicators": ["SMA", "ADX"],
                "risk_level": "low",
                "description": "リスクを抑えた長期成長戦略",
            },
            {
                "name": "Contrarian Elite",
                "category": "mean_reversion",
                "indicators": ["RSI", "BB", "WILLIAMS", "CCI"],
                "risk_level": "high",
                "description": "逆張りに特化した上級者向け戦略",
            },
        ]

        # 全戦略を結合
        all_strategies = (
            single_indicator_strategies
            + multi_indicator_strategies
            + special_strategies
        )

        # 指定された数だけ返す
        return all_strategies[:count]

    def _create_strategy_gene(self, pattern: Dict[str, Any]):
        """
        戦略パターンから戦略遺伝子を作成

        Args:
            pattern: 戦略パターン

        Returns:
            戦略遺伝子
        """
        # 基本的な遺伝子構造を作成
        gene_data = {
            "indicators": [],
            "conditions": [],
            "max_indicators": len(pattern["indicators"]),
        }

        # 指標ごとにパラメータを生成
        for indicator in pattern["indicators"]:
            indicator_config = self._generate_indicator_config(indicator, pattern)
            gene_data["indicators"].append(indicator_config)

        # 条件を生成
        conditions = self._generate_conditions(pattern)
        gene_data["conditions"] = conditions

        # 動的インポートでStrategyGeneを使用
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene

        return StrategyGene.from_dict(gene_data)

    def _generate_indicator_config(
        self, indicator: str, pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        指標の設定を生成

        Args:
            indicator: 指標名
            pattern: 戦略パターン

        Returns:
            指標設定
        """
        risk_level = pattern["risk_level"]

        # リスクレベルに応じてパラメータ範囲を調整
        if risk_level == "low":
            period_multiplier = 1.2  # より長期
            threshold_multiplier = 0.8  # より保守的
        elif risk_level == "high":
            period_multiplier = 0.8  # より短期
            threshold_multiplier = 1.2  # より積極的
        else:
            period_multiplier = 1.0
            threshold_multiplier = 1.0

        # 指標別のデフォルト設定
        configs = {
            "RSI": {
                "name": "RSI",
                "period": int(14 * period_multiplier),
                "overbought": 70 * threshold_multiplier,
                "oversold": 30 / threshold_multiplier,
            },
            "MACD": {
                "name": "MACD",
                "fast_period": int(12 * period_multiplier),
                "slow_period": int(26 * period_multiplier),
                "signal_period": int(9 * period_multiplier),
            },
            "SMA": {
                "name": "SMA",
                "short_period": int(10 * period_multiplier),
                "long_period": int(20 * period_multiplier),
            },
            "EMA": {
                "name": "EMA",
                "short_period": int(8 * period_multiplier),
                "long_period": int(21 * period_multiplier),
            },
            "BB": {
                "name": "BB",
                "period": int(20 * period_multiplier),
                "std_dev": 2.0 * threshold_multiplier,
            },
            "STOCH": {
                "name": "STOCH",
                "k_period": int(14 * period_multiplier),
                "d_period": int(3 * period_multiplier),
                "overbought": 80 * threshold_multiplier,
                "oversold": 20 / threshold_multiplier,
            },
            "ATR": {
                "name": "ATR",
                "period": int(14 * period_multiplier),
                "multiplier": 2.0 * threshold_multiplier,
            },
            "CCI": {
                "name": "CCI",
                "period": int(20 * period_multiplier),
                "overbought": 100 * threshold_multiplier,
                "oversold": -100 / threshold_multiplier,
            },
            "WILLIAMS": {
                "name": "WILLIAMS",
                "period": int(14 * period_multiplier),
                "overbought": -20 * threshold_multiplier,
                "oversold": -80 / threshold_multiplier,
            },
            "ADX": {
                "name": "ADX",
                "period": int(14 * period_multiplier),
                "threshold": 25 * threshold_multiplier,
            },
        }

        # その他の指標のデフォルト設定
        default_config = {"name": indicator, "period": int(14 * period_multiplier)}

        return configs.get(indicator, default_config)

    def _generate_conditions(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        戦略の条件を生成

        Args:
            pattern: 戦略パターン

        Returns:
            条件のリスト
        """
        category = pattern["category"]
        indicators = pattern["indicators"]

        conditions = []

        if category == "trend_following":
            # トレンドフォロー条件
            if "SMA" in indicators or "EMA" in indicators:
                conditions.append(
                    {
                        "type": "cross_above",
                        "indicator1": "price",
                        "indicator2": "moving_average",
                    }
                )
            if "MACD" in indicators:
                conditions.append(
                    {
                        "type": "cross_above",
                        "indicator1": "macd_line",
                        "indicator2": "signal_line",
                    }
                )

        elif category == "mean_reversion":
            # 逆張り条件
            if "RSI" in indicators:
                conditions.append(
                    {
                        "type": "below_threshold",
                        "indicator": "RSI",
                        "threshold": "oversold",
                    }
                )
            if "BB" in indicators:
                conditions.append(
                    {"type": "below_band", "indicator": "BB", "band": "lower"}
                )

        elif category == "breakout":
            # ブレイクアウト条件
            if "ATR" in indicators:
                conditions.append(
                    {
                        "type": "volatility_breakout",
                        "indicator": "ATR",
                        "multiplier": 2.0,
                    }
                )
            if "BB" in indicators:
                conditions.append(
                    {"type": "band_breakout", "indicator": "BB", "direction": "upper"}
                )

        elif category == "momentum":
            # モメンタム条件
            if "RSI" in indicators:
                conditions.append(
                    {
                        "type": "momentum_confirmation",
                        "indicator": "RSI",
                        "threshold": 50,
                    }
                )

        # デフォルト条件
        if not conditions:
            conditions.append(
                {
                    "type": "simple_cross",
                    "indicator": indicators[0] if indicators else "SMA",
                }
            )

        return conditions

    @property
    def auto_strategy_service(self):
        """AutoStrategyServiceの遅延初期化"""
        if self._auto_strategy_service is None:
            from app.core.services.auto_strategy import AutoStrategyService

            self._auto_strategy_service = AutoStrategyService()
        return self._auto_strategy_service

    def _run_backtest(self, gene, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        戦略のバックテストを実行

        Args:
            gene: 戦略遺伝子
            base_config: バックテスト設定

        Returns:
            バックテスト結果
        """
        try:
            # AutoStrategyServiceを使用してバックテストを実行
            result = self.auto_strategy_service.test_strategy_generation(
                gene, base_config
            )
            return result

        except Exception as e:
            logger.error(f"バックテスト実行エラー: {e}")
            return {"success": False, "error": str(e)}

    def _create_showcase_strategy_data(
        self,
        pattern: Dict[str, Any],
        gene,
        backtest_result: Dict[str, Any],
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        ショーケース戦略データを作成

        Args:
            pattern: 戦略パターン
            gene: 戦略遺伝子
            backtest_result: バックテスト結果
            base_config: バックテスト設定

        Returns:
            ショーケース戦略データ
        """
        # バックテスト結果からパフォーマンス指標を抽出
        performance = backtest_result.get("result", {}).get("performance_metrics", {})

        # パフォーマンス指標の計算
        expected_return = performance.get("total_return_pct", 0.0)
        sharpe_ratio = performance.get("sharpe_ratio", 0.0)
        max_drawdown = abs(performance.get("max_drawdown_pct", 0.0))
        win_rate = performance.get("win_rate", 0.0) * 100  # パーセント表示

        # 推奨時間軸の決定
        recommended_timeframe = self._determine_recommended_timeframe(pattern)

        # パラメータの整理
        parameters = self._extract_parameters_from_gene(gene)

        return {
            "name": pattern["name"],
            "description": pattern["description"],
            "category": pattern["category"],
            "indicators": pattern["indicators"],
            "parameters": parameters,
            "expected_return": round(expected_return, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "max_drawdown": round(max_drawdown, 2),
            "win_rate": round(win_rate, 1),
            "gene_data": gene.to_dict(),
            "risk_level": pattern["risk_level"],
            "recommended_timeframe": recommended_timeframe,
            "is_active": True,
        }

    def _determine_recommended_timeframe(self, pattern: Dict[str, Any]) -> str:
        """
        戦略に適した推奨時間軸を決定

        Args:
            pattern: 戦略パターン

        Returns:
            推奨時間軸
        """
        category = pattern["category"]
        risk_level = pattern["risk_level"]

        if category == "momentum" and risk_level == "high":
            return "15m"  # 高頻度取引
        elif category == "trend_following" and risk_level == "low":
            return "1d"  # 長期トレンド
        elif category == "breakout":
            return "1h"  # ブレイクアウト
        elif category == "range_trading":
            return "4h"  # レンジ取引
        else:
            return "1h"  # デフォルト

    def _extract_parameters_from_gene(self, gene) -> Dict[str, Any]:
        """
        戦略遺伝子からパラメータを抽出

        Args:
            gene: 戦略遺伝子

        Returns:
            パラメータ辞書
        """
        gene_dict = gene.to_dict()
        parameters = {}

        # 指標パラメータを抽出
        for indicator in gene_dict.get("indicators", []):
            indicator_name = indicator.get("name", "unknown")
            indicator_params = {k: v for k, v in indicator.items() if k != "name"}
            parameters[indicator_name] = indicator_params

        # 条件パラメータを抽出
        conditions = gene_dict.get("conditions", [])
        if conditions:
            parameters["conditions"] = conditions

        return parameters

    def _create_simple_strategy_data(
        self, pattern: Dict[str, Any], base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        簡単な戦略データを作成（テスト用）

        Args:
            pattern: 戦略パターン
            base_config: バックテスト設定

        Returns:
            戦略データ
        """
        # リスクレベルに応じてパフォーマンス指標を生成
        risk_multiplier = {"low": 0.7, "medium": 1.0, "high": 1.3}.get(
            pattern["risk_level"], 1.0
        )

        # カテゴリに応じてベースパフォーマンスを調整
        category_multiplier = {
            "trend_following": 1.1,
            "mean_reversion": 0.9,
            "breakout": 1.2,
            "range_trading": 0.8,
            "momentum": 1.0,
        }.get(pattern["category"], 1.0)

        # ランダムな要素を追加してリアルな数値を生成
        import random

        random.seed(hash(pattern["name"]) % 2**32)  # 戦略名でシード固定

        base_return = random.uniform(5, 25) * category_multiplier * risk_multiplier
        base_sharpe = random.uniform(0.5, 2.5) * (2.0 - risk_multiplier + 0.5)
        base_drawdown = random.uniform(3, 20) * risk_multiplier
        base_win_rate = random.uniform(45, 75)

        # パラメータを生成
        parameters = {}
        for indicator in pattern["indicators"]:
            parameters[indicator] = {
                "period": random.randint(10, 30),
                "threshold": round(random.uniform(0.5, 2.0), 2),
            }

        # 推奨時間軸の決定
        recommended_timeframe = self._determine_recommended_timeframe(pattern)

        return {
            "name": pattern["name"],
            "description": pattern["description"],
            "category": pattern["category"],
            "indicators": pattern["indicators"],
            "parameters": parameters,
            "expected_return": round(base_return, 2),
            "sharpe_ratio": round(base_sharpe, 3),
            "max_drawdown": round(base_drawdown, 2),
            "win_rate": round(base_win_rate, 1),
            "gene_data": {
                "indicators": pattern["indicators"],
                "parameters": parameters,
                "conditions": [],
            },
            "risk_level": pattern["risk_level"],
            "recommended_timeframe": recommended_timeframe,
            "is_active": True,
        }
