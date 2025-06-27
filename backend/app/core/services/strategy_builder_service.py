"""
ストラテジービルダーサービス

ユーザー定義戦略の管理とビジネスロジックを提供するサービス
"""

from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
import logging
from datetime import datetime

from database.repositories.user_strategy_repository import UserStrategyRepository
from database.models import UserStrategy
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.indicators.indicator_orchestrator import (
    TechnicalIndicatorService,
)

logger = logging.getLogger(__name__)


class StrategyBuilderService:
    """ストラテジービルダーサービスクラス"""

    def __init__(self, db: Session):
        """
        初期化

        Args:
            db: データベースセッション
        """
        self.db = db
        self.user_strategy_repo = UserStrategyRepository(db)
        self.technical_indicator_service = TechnicalIndicatorService()
        self.strategy_factory = StrategyFactory()

    def get_available_indicators(self) -> Dict[str, Any]:
        """
        利用可能なテクニカル指標の一覧を取得

        Returns:
            指標情報の辞書（カテゴリ別）
        """
        try:
            # TechnicalIndicatorServiceから指標情報を取得
            indicators_info = self.technical_indicator_service.supported_indicators

            # カテゴリ別に整理
            categorized_indicators = {
                "trend": [],
                "momentum": [],
                "volatility": [],
                "volume": [],
                "price_transform": [],
                "other": [],
            }

            # 各指標をカテゴリに分類
            for indicator_type, info in indicators_info.items():
                category = self._get_indicator_category(indicator_type)

                indicator_info = {
                    "type": indicator_type,
                    "name": info.get("name", indicator_type),
                    "description": info.get("description", ""),
                    "parameters": self._get_indicator_parameters(indicator_type),
                    "data_sources": info.get("data_sources", ["close"]),
                }

                categorized_indicators[category].append(indicator_info)

            logger.info(f"利用可能な指標を取得しました: {len(indicators_info)}種類")
            return categorized_indicators

        except Exception as e:
            logger.error(f"指標一覧取得エラー: {e}")
            raise

    def _get_indicator_category(self, indicator_type: str) -> str:
        """
        指標タイプからカテゴリを判定

        Args:
            indicator_type: 指標タイプ

        Returns:
            カテゴリ名
        """
        # トレンド系指標
        trend_indicators = [
            "SMA",
            "EMA",
            "MACD",
            "KAMA",
            "T3",
            "TEMA",
            "DEMA",
            "WMA",
            "HMA",
            "VWMA",
            "ZLEMA",
            "MIDPOINT",
            "MIDPRICE",
            "TRIMA",
        ]

        # モメンタム系指標
        momentum_indicators = [
            "RSI",
            "STOCH",
            "CCI",
            "WILLR",
            "MOM",
            "ROC",
            "ADX",
            "AROON",
            "MFI",
            "STOCHRSI",
            "ULTOSC",
            "BOP",
            "PPO",
            "PLUS_DI",
            "MINUS_DI",
            "ROCP",
            "ROCR",
            "STOCHF",
            "CMO",
            "TRIX",
            "APO",
            "AROONOSC",
            "DX",
            "ADXR",
        ]

        # ボラティリティ系指標
        volatility_indicators = [
            "ATR",
            "NATR",
            "TRANGE",
            "BB",
            "STDDEV",
            "VAR",
            "BETA",
            "CORREL",
        ]

        # ボリューム系指標
        volume_indicators = ["OBV", "AD", "ADOSC"]

        # 価格変換系指標
        price_transform_indicators = [
            "AVGPRICE",
            "MEDPRICE",
            "TYPPRICE",
            "WCLPRICE",
            "HT_DCPERIOD",
            "HT_DCPHASE",
            "HT_PHASOR",
            "HT_SINE",
            "HT_TRENDMODE",
            "MAMA",
            "FAMA",
            "SAREXT",
            "SAR",
            "APO",
        ]

        if indicator_type in trend_indicators:
            return "trend"
        elif indicator_type in momentum_indicators:
            return "momentum"
        elif indicator_type in volatility_indicators:
            return "volatility"
        elif indicator_type in volume_indicators:
            return "volume"
        elif indicator_type in price_transform_indicators:
            return "price_transform"
        else:
            return "other"

    def _get_indicator_parameters(self, indicator_type: str) -> List[Dict[str, Any]]:
        """
        指標のパラメータ定義を取得

        Args:
            indicator_type: 指標タイプ

        Returns:
            パラメータ定義のリスト
        """
        # 基本的なパラメータ定義（実際の実装では設定ファイルから取得）
        parameter_definitions = {
            "SMA": [
                {
                    "name": "period",
                    "type": "integer",
                    "default": 20,
                    "min": 2,
                    "max": 200,
                    "description": "移動平均期間",
                }
            ],
            "EMA": [
                {
                    "name": "period",
                    "type": "integer",
                    "default": 20,
                    "min": 2,
                    "max": 200,
                    "description": "移動平均期間",
                }
            ],
            "RSI": [
                {
                    "name": "period",
                    "type": "integer",
                    "default": 14,
                    "min": 2,
                    "max": 100,
                    "description": "RSI計算期間",
                }
            ],
            "MACD": [
                {
                    "name": "fast_period",
                    "type": "integer",
                    "default": 12,
                    "min": 2,
                    "max": 50,
                    "description": "短期期間",
                },
                {
                    "name": "slow_period",
                    "type": "integer",
                    "default": 26,
                    "min": 2,
                    "max": 100,
                    "description": "長期期間",
                },
                {
                    "name": "signal_period",
                    "type": "integer",
                    "default": 9,
                    "min": 2,
                    "max": 50,
                    "description": "シグナル期間",
                },
            ],
            "ATR": [
                {
                    "name": "period",
                    "type": "integer",
                    "default": 14,
                    "min": 2,
                    "max": 100,
                    "description": "ATR計算期間",
                }
            ],
            "BB": [
                {
                    "name": "period",
                    "type": "integer",
                    "default": 20,
                    "min": 2,
                    "max": 100,
                    "description": "移動平均期間",
                },
                {
                    "name": "std_dev",
                    "type": "float",
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "description": "標準偏差倍率",
                },
            ],
        }

        return parameter_definitions.get(
            indicator_type,
            [
                {
                    "name": "period",
                    "type": "integer",
                    "default": 14,
                    "min": 2,
                    "max": 200,
                    "description": "計算期間",
                }
            ],
        )

    def validate_strategy_config(
        self, strategy_config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        戦略設定の妥当性を検証

        Args:
            strategy_config: 戦略設定（StrategyGene形式）

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        try:
            # 基本的な構造チェック
            if not isinstance(strategy_config, dict):
                return False, ["戦略設定は辞書形式である必要があります"]

            # 必須フィールドのチェック
            required_fields = ["indicators", "entry_conditions", "exit_conditions"]
            for field in required_fields:
                if field not in strategy_config:
                    errors.append(f"必須フィールド '{field}' が見つかりません")

            if errors:
                return False, errors

            # 指標の検証
            indicators = strategy_config.get("indicators", [])
            if not isinstance(indicators, list):
                errors.append("indicators は配列である必要があります")
            elif len(indicators) == 0:
                errors.append("少なくとも1つの指標を設定してください")
            else:
                for i, indicator in enumerate(indicators):
                    if not isinstance(indicator, dict):
                        errors.append(f"指標 {i+1} は辞書形式である必要があります")
                        continue

                    if "type" not in indicator:
                        errors.append(f"指標 {i+1} にタイプが設定されていません")

                    if "parameters" not in indicator:
                        errors.append(f"指標 {i+1} にパラメータが設定されていません")

            # 条件の検証
            entry_conditions = strategy_config.get("entry_conditions", [])
            exit_conditions = strategy_config.get("exit_conditions", [])

            if not isinstance(entry_conditions, list):
                errors.append("entry_conditions は配列である必要があります")
            if not isinstance(exit_conditions, list):
                errors.append("exit_conditions は配列である必要があります")

            if len(entry_conditions) == 0 and len(exit_conditions) == 0:
                errors.append(
                    "少なくとも1つのエントリー条件またはイグジット条件を設定してください"
                )

            # エラーがある場合は早期リターン
            if errors:
                return False, errors

            # StrategyGeneオブジェクトに変換して詳細検証
            try:
                # 指標をIndicatorGeneオブジェクトに変換
                indicator_genes = []
                for indicator in indicators:
                    indicator_gene = IndicatorGene(
                        type=indicator["type"],
                        parameters=indicator.get("parameters", {}),
                        enabled=indicator.get("enabled", True),
                        json_config=indicator.get("json_config", {}),
                    )
                    indicator_genes.append(indicator_gene)

                # 条件をConditionオブジェクトに変換
                entry_condition_objects = []
                for condition in entry_conditions:
                    if isinstance(condition, dict):
                        # 辞書形式の条件をConditionオブジェクトに変換
                        condition_obj = Condition(
                            left_operand=condition.get(
                                "indicator", condition.get("left_operand", "")
                            ),
                            operator=condition.get("operator", ""),
                            right_operand=condition.get(
                                "value", condition.get("right_operand", 0)
                            ),
                        )
                        entry_condition_objects.append(condition_obj)

                exit_condition_objects = []
                for condition in exit_conditions:
                    if isinstance(condition, dict):
                        condition_obj = Condition(
                            left_operand=condition.get(
                                "indicator", condition.get("left_operand", "")
                            ),
                            operator=condition.get("operator", ""),
                            right_operand=condition.get(
                                "value", condition.get("right_operand", 0)
                            ),
                        )
                        exit_condition_objects.append(condition_obj)

                # StrategyGeneオブジェクトを作成
                strategy_gene = StrategyGene(
                    indicators=indicator_genes,
                    entry_conditions=entry_condition_objects,
                    exit_conditions=exit_condition_objects,
                )

                # StrategyGeneの検証
                is_valid, gene_errors = strategy_gene.validate()
                if not is_valid:
                    errors.extend(gene_errors)

                # StrategyFactoryでの検証も実行
                if is_valid:
                    try:
                        factory_valid, factory_errors = (
                            self.strategy_factory.validate_gene(strategy_gene)
                        )
                        if not factory_valid:
                            errors.extend(factory_errors)
                            is_valid = False
                    except Exception as factory_error:
                        errors.append(
                            f"戦略ファクトリー検証エラー: {str(factory_error)}"
                        )
                        is_valid = False

            except Exception as gene_error:
                errors.append(f"戦略遺伝子作成エラー: {str(gene_error)}")
                is_valid = False

            final_valid = len(errors) == 0
            logger.debug(f"戦略設定検証結果: valid={final_valid}, errors={errors}")
            return final_valid, errors

        except Exception as e:
            logger.error(f"戦略設定検証エラー: {e}")
            return False, [f"戦略設定検証中に予期しないエラーが発生しました: {str(e)}"]

    def _convert_to_strategy_gene_format(
        self, strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        戦略設定をStrategyGene形式に変換

        Args:
            strategy_config: 戦略設定

        Returns:
            StrategyGene形式の戦略設定
        """
        try:
            # 指標をIndicatorGene形式に変換
            indicator_genes = []
            for indicator in strategy_config.get("indicators", []):
                indicator_gene = {
                    "type": indicator["type"],
                    "parameters": indicator.get("parameters", {}),
                    "enabled": indicator.get("enabled", True),
                    "json_config": indicator.get(
                        "json_config",
                        {
                            "indicator_name": indicator["type"],
                            "parameters": indicator.get("parameters", {}),
                        },
                    ),
                }
                indicator_genes.append(indicator_gene)

            # 条件をCondition形式に変換
            def convert_conditions(conditions):
                converted = []
                for condition in conditions:
                    if condition.get("type") == "threshold":
                        converted_condition = {
                            "left_operand": condition.get("indicator", ""),
                            "operator": condition.get("operator", ""),
                            "right_operand": condition.get("value", 0),
                        }
                    elif condition.get("type") == "crossover":
                        converted_condition = {
                            "left_operand": condition.get("indicator1", ""),
                            "operator": condition.get("operator", ""),
                            "right_operand": condition.get("indicator2", ""),
                        }
                    elif condition.get("type") == "comparison":
                        converted_condition = {
                            "left_operand": condition.get("indicator1", ""),
                            "operator": condition.get("operator", ""),
                            "right_operand": condition.get("indicator2", ""),
                        }
                    else:
                        # デフォルト形式
                        converted_condition = {
                            "left_operand": condition.get(
                                "indicator", condition.get("left_operand", "")
                            ),
                            "operator": condition.get("operator", ""),
                            "right_operand": condition.get(
                                "value", condition.get("right_operand", 0)
                            ),
                        }
                    converted.append(converted_condition)
                return converted

            entry_conditions = convert_conditions(
                strategy_config.get("entry_conditions", [])
            )
            exit_conditions = convert_conditions(
                strategy_config.get("exit_conditions", [])
            )

            # StrategyGene形式の辞書を作成
            strategy_gene_dict = {
                "indicators": indicator_genes,
                "entry_conditions": entry_conditions,
                "exit_conditions": exit_conditions,
                "risk_management": strategy_config.get(
                    "risk_management",
                    {
                        "stop_loss_pct": 0.02,
                        "take_profit_pct": 0.05,
                        "position_sizing": "fixed",
                    },
                ),
                "metadata": strategy_config.get(
                    "metadata",
                    {
                        "created_by": "strategy_builder",
                        "version": "1.0",
                        "created_at": datetime.now().isoformat(),
                    },
                ),
            }

            return strategy_gene_dict

        except Exception as e:
            logger.error(f"戦略設定変換エラー: {e}")
            raise ValueError(f"戦略設定の変換に失敗しました: {str(e)}")

    def generate_strategy_preview(
        self, strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        戦略設定のプレビューデータを生成

        Args:
            strategy_config: 戦略設定

        Returns:
            プレビューデータ
        """
        try:
            # 使用される指標の情報を収集
            indicators_used = []
            for indicator in strategy_config.get("indicators", []):
                indicator_type = indicator.get("type", "")
                parameters = indicator.get("parameters", {})

                # 指標の詳細情報を取得
                indicator_info = {
                    "type": indicator_type,
                    "name": self._get_indicator_display_name(indicator_type),
                    "parameters": parameters,
                    "category": self._get_indicator_category(indicator_type),
                    "enabled": indicator.get("enabled", True),
                }
                indicators_used.append(indicator_info)

            # 条件の要約を生成
            entry_conditions = strategy_config.get("entry_conditions", [])
            exit_conditions = strategy_config.get("exit_conditions", [])

            conditions_summary = {
                "entry_conditions_count": len(entry_conditions),
                "exit_conditions_count": len(exit_conditions),
                "entry_conditions": [
                    self._format_condition_for_preview(cond)
                    for cond in entry_conditions
                ],
                "exit_conditions": [
                    self._format_condition_for_preview(cond) for cond in exit_conditions
                ],
            }

            # 戦略の要約を生成
            strategy_summary = {
                "total_indicators": len(indicators_used),
                "enabled_indicators": len(
                    [ind for ind in indicators_used if ind["enabled"]]
                ),
                "total_conditions": len(entry_conditions) + len(exit_conditions),
                "complexity_score": self._calculate_complexity_score(strategy_config),
                "estimated_signals_per_day": self._estimate_signals_per_day(
                    strategy_config
                ),
            }

            # カテゴリ別の指標統計
            category_stats = {}
            for indicator in indicators_used:
                category = indicator["category"]
                if category not in category_stats:
                    category_stats[category] = 0
                category_stats[category] += 1

            return {
                "strategy_summary": strategy_summary,
                "indicators_used": indicators_used,
                "conditions_summary": conditions_summary,
                "category_statistics": category_stats,
                "preview_generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"戦略プレビュー生成エラー: {e}")
            raise ValueError(f"戦略プレビューの生成に失敗しました: {str(e)}")

    def _get_indicator_display_name(self, indicator_type: str) -> str:
        """指標の表示名を取得"""
        try:
            indicator_info = self.technical_indicator_service.supported_indicators.get(
                indicator_type, {}
            )
            return indicator_info.get("name", indicator_type)
        except Exception:
            return indicator_type

    def _format_condition_for_preview(
        self, condition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """条件をプレビュー用にフォーマット"""
        return {
            "type": condition.get("type", "unknown"),
            "indicator": condition.get("indicator", ""),
            "operator": condition.get("operator", ""),
            "value": condition.get("value", 0),
            "description": self._generate_condition_description(condition),
        }

    def _generate_condition_description(self, condition: Dict[str, Any]) -> str:
        """条件の説明文を生成"""
        try:
            condition_type = condition.get("type", "")
            indicator = condition.get("indicator", "")
            operator = condition.get("operator", "")
            value = condition.get("value", 0)

            operator_map = {
                ">": "が",
                "<": "が",
                ">=": "が",
                "<=": "が",
                "==": "が",
                "!=": "が",
            }

            operator_text = operator_map.get(operator, "")

            if condition_type == "threshold":
                return f"{indicator}{operator_text}{value}{operator}の時"
            else:
                return f"{condition_type}条件"

        except Exception:
            return "条件"

    def _calculate_complexity_score(self, strategy_config: Dict[str, Any]) -> int:
        """戦略の複雑度スコアを計算（1-10）"""
        try:
            score = 1

            # 指標数による加点
            indicators_count = len(strategy_config.get("indicators", []))
            score += min(indicators_count, 5)

            # 条件数による加点
            conditions_count = len(strategy_config.get("entry_conditions", [])) + len(
                strategy_config.get("exit_conditions", [])
            )
            score += min(conditions_count, 4)

            return min(score, 10)

        except Exception:
            return 5  # デフォルト値

    def _estimate_signals_per_day(self, strategy_config: Dict[str, Any]) -> str:
        """1日あたりの推定シグナル数を計算"""
        try:
            # 簡易的な推定ロジック
            conditions_count = len(strategy_config.get("entry_conditions", [])) + len(
                strategy_config.get("exit_conditions", [])
            )

            if conditions_count <= 2:
                return "高頻度 (5-10回/日)"
            elif conditions_count <= 4:
                return "中頻度 (2-5回/日)"
            else:
                return "低頻度 (0-2回/日)"

        except Exception:
            return "不明"

    def save_strategy(
        self, name: str, description: Optional[str], strategy_config: Dict[str, Any]
    ) -> UserStrategy:
        """
        戦略を保存

        Args:
            name: 戦略名
            description: 戦略の説明
            strategy_config: 戦略設定（StrategyGene形式）

        Returns:
            保存されたUserStrategyオブジェクト

        Raises:
            ValueError: 戦略設定が無効な場合
            Exception: データベースエラーの場合
        """
        try:
            # 戦略設定の検証
            is_valid, errors = self.validate_strategy_config(strategy_config)
            if not is_valid:
                raise ValueError(f"戦略設定が無効です: {', '.join(errors)}")

            # 戦略設定をStrategyGene形式に変換
            converted_config = self._convert_to_strategy_gene_format(strategy_config)

            # 戦略データの準備
            strategy_data = {
                "name": name,
                "description": description,
                "strategy_config": converted_config,
                "is_active": True,
            }

            # データベースに保存
            user_strategy = self.user_strategy_repo.create(strategy_data)

            logger.info(f"戦略を保存しました: ID={user_strategy.id}, 名前={name}")
            return user_strategy

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"戦略保存エラー: {e}")
            raise

    def get_strategies(
        self, active_only: bool = True, limit: Optional[int] = None
    ) -> List[UserStrategy]:
        """
        保存済み戦略の一覧を取得

        Args:
            active_only: アクティブな戦略のみを取得するか
            limit: 取得件数制限

        Returns:
            UserStrategyオブジェクトのリスト
        """
        try:
            strategies = self.user_strategy_repo.get_all(
                active_only=active_only, limit=limit
            )
            logger.info(f"戦略一覧を取得しました: {len(strategies)}件")
            return strategies

        except Exception as e:
            logger.error(f"戦略一覧取得エラー: {e}")
            raise

    def get_strategy_by_id(self, strategy_id: int) -> Optional[UserStrategy]:
        """
        IDで戦略を取得

        Args:
            strategy_id: 戦略ID

        Returns:
            UserStrategyオブジェクト（見つからない場合はNone）
        """
        try:
            strategy = self.user_strategy_repo.get_by_id(strategy_id)
            if strategy:
                logger.debug(f"戦略を取得しました: ID={strategy_id}")
            else:
                logger.debug(f"戦略が見つかりません: ID={strategy_id}")
            return strategy

        except Exception as e:
            logger.error(f"戦略取得エラー (ID={strategy_id}): {e}")
            raise

    def update_strategy(
        self,
        strategy_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        strategy_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[UserStrategy]:
        """
        戦略を更新

        Args:
            strategy_id: 戦略ID
            name: 戦略名（オプション）
            description: 戦略の説明（オプション）
            strategy_config: 戦略設定（オプション）

        Returns:
            更新されたUserStrategyオブジェクト（見つからない場合はNone）

        Raises:
            ValueError: 戦略設定が無効な場合
            Exception: データベースエラーの場合
        """
        try:
            # 更新データの準備
            update_data = {}

            if name is not None:
                update_data["name"] = name
            if description is not None:
                update_data["description"] = description
            if strategy_config is not None:
                # 戦略設定の検証
                is_valid, errors = self.validate_strategy_config(strategy_config)
                if not is_valid:
                    raise ValueError(f"戦略設定が無効です: {', '.join(errors)}")
                update_data["strategy_config"] = strategy_config

            # データベースで更新
            updated_strategy = self.user_strategy_repo.update(strategy_id, update_data)

            if updated_strategy:
                logger.info(f"戦略を更新しました: ID={strategy_id}")
            else:
                logger.warning(f"更新対象の戦略が見つかりません: ID={strategy_id}")

            return updated_strategy

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"戦略更新エラー (ID={strategy_id}): {e}")
            raise

    def delete_strategy(self, strategy_id: int) -> bool:
        """
        戦略を削除（論理削除）

        Args:
            strategy_id: 戦略ID

        Returns:
            削除成功の場合True、見つからない場合False
        """
        try:
            result = self.user_strategy_repo.delete(strategy_id)
            if result:
                logger.info(f"戦略を削除しました: ID={strategy_id}")
            else:
                logger.warning(f"削除対象の戦略が見つかりません: ID={strategy_id}")
            return result

        except Exception as e:
            logger.error(f"戦略削除エラー (ID={strategy_id}): {e}")
            raise
