"""
遺伝子バリデーション

戦略遺伝子の妥当性検証を担当するモジュール。
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class GeneValidator:
    """
    遺伝子バリデーター

    戦略遺伝子の妥当性検証を担当します。
    """

    def is_valid_indicator_name(self, name: str) -> bool:
        """公開APIラッパー（旧テスト互換）"""
        return self._is_indicator_name(name)

    def __init__(self):
        """初期化"""
        self.valid_indicator_types = self._get_valid_indicator_types()
        self.valid_operators = self._get_valid_operators()
        self.valid_data_sources = self._get_valid_data_sources()

    def _get_valid_indicator_types(self) -> List[str]:
        """有効な指標タイプのリストを取得"""
        # 共通定数から取得して一貫性を保つ
        try:
            from app.services.indicators.config.indicator_config import (
                indicator_registry,
            )

            return indicator_registry.list_indicators()
        except ImportError:
            # フォールバック: 包括的な指標リスト
            return [
                # トレンド系指標
                "SMA",
                "EMA",
                "WMA",
                "DEMA",
                "TEMA",
                "TRIMA",
                "KAMA",
                "MAMA",
                "T3",
                "MA",
                "MACD",
                "MACDEXT",
                "MACDFIX",
                "BB",
                "HT_TRENDLINE",
                # モメンタム系指標
                "RSI",
                "STOCH",
                "STOCHF",
                "STOCHRSI",
                "CCI",
                "ADX",
                "ADXR",
                "DX",
                "PLUS_DI",
                "MINUS_DI",
                "PLUS_DM",
                "MINUS_DM",
                "WILLR",
                "MFI",
                "BOP",
                "ROC",
                "ROCP",
                "ROCR",
                "ROCR100",
                "MOM",
                "MOMENTUM",
                "CMO",
                "TRIX",
                "APO",
                "PPO",
                "ULTOSC",
                "AROON",
                "AROONOSC",
                # ボラティリティ系指標
                "ATR",
                "NATR",
                "TRANGE",
                "STDDEV",
                "VAR",
                # ボリューム系指標
                "OBV",
                "AD",
                "ADOSC",
                # 統計系指標
                "BETA",
                "CORREL",
                "LINEARREG",
                "LINEARREG_ANGLE",
                "LINEARREG_INTERCEPT",
                "LINEARREG_SLOPE",
                "TSF",
                # 数学変換系指標
                "ACOS",
                "ASIN",
                "ATAN",
                "COS",
                "COSH",
                "SIN",
                "SINH",
                "TAN",
                "TANH",
                "CEIL",
                "EXP",
                "FLOOR",
                "LN",
                "LOG10",
                "SQRT",
                # 数学演算子
                "ADD",
                "DIV",
                "MULT",
                "SUB",
                "MAX",
                "MIN",
                "MAXINDEX",
                "MININDEX",
                "SUM",
                "MINMAX",
                "MINMAXINDEX",
                # 価格変換系指標
                "AVGPRICE",
                "MEDPRICE",
                "TYPPRICE",
                "WCLPRICE",
                # オーバーレイ系指標
                "SAR",
                # パターン認識系指標
                "CDL_DOJI",
                "CDL_HAMMER",
                "CDL_HANGING_MAN",
                "CDL_SHOOTING_STAR",
                "CDL_ENGULFING",
                "CDL_HARAMI",
                "CDL_PIERCING",
                "CDL_THREE_BLACK_CROWS",
                "CDL_THREE_WHITE_SOLDIERS",
                "CDL_DARK_CLOUD_COVER",
            ]

    def _get_valid_operators(self) -> List[str]:
        """有効な演算子のリストを取得"""
        return [
            ">",
            "<",
            ">=",
            "<=",
            "==",
            "!=",  # 基本比較演算子
            "above",
            "below",  # フロントエンド用演算子
        ]

    def _get_valid_data_sources(self) -> List[str]:
        """有効なデータソースのリストを取得"""
        return [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "OpenInterest",
            "FundingRate",
        ]

    def validate_indicator_gene(self, indicator_gene) -> bool:
        """
        指標遺伝子の妥当性を検証

        Args:
            indicator_gene: 指標遺伝子オブジェクト

        Returns:
            妥当性（True/False）
        """
        try:
            if not indicator_gene.type or not isinstance(indicator_gene.type, str):
                return False

            if not isinstance(indicator_gene.parameters, dict):
                return False

            # 有効な指標タイプの確認
            if indicator_gene.type not in self.valid_indicator_types:
                return False

            # パラメータの妥当性確認
            if "period" in indicator_gene.parameters:
                period = indicator_gene.parameters["period"]
                if not isinstance(period, (int, float)) or period <= 0:
                    return False

            return True

        except Exception as e:
            logger.error(f"指標遺伝子バリデーションエラー: {e}")
            return False

    def validate_condition(self, condition) -> tuple[bool, str]:
        """
        条件の妥当性を検証

        Args:
            condition: 条件オブジェクト

        Returns:
            (妥当性（True/False）, エラー詳細)
        """
        try:
            # 条件オブジェクトの基本検証
            if (
                not hasattr(condition, "operator")
                or not hasattr(condition, "left_operand")
                or not hasattr(condition, "right_operand")
            ):
                error_msg = f"条件オブジェクトに必要な属性がありません: {condition}"
                logger.warning(error_msg)
                return False, error_msg

            # オペレーターの検証
            if condition.operator not in self.valid_operators:
                error_msg = f"無効な演算子: {condition.operator}, 有効な演算子: {self.valid_operators}"
                logger.warning(error_msg)
                return False, error_msg

            # オペランドの検証
            left_valid, left_error = self._is_valid_operand_detailed(
                condition.left_operand
            )
            if not left_valid:
                error_msg = f"無効な左オペランド: {condition.left_operand} (型: {type(condition.left_operand)}) - {left_error}"
                logger.warning(error_msg)
                return False, error_msg

            right_valid, right_error = self._is_valid_operand_detailed(
                condition.right_operand
            )
            if not right_valid:
                error_msg = f"無効な右オペランド: {condition.right_operand} (型: {type(condition.right_operand)}) - {right_error}"
                logger.warning(error_msg)
                return False, error_msg

            return True, ""

        except Exception as e:
            error_msg = f"条件バリデーションエラー: {e}"
            logger.error(error_msg)
            return False, error_msg

    def _is_valid_operand(self, operand) -> bool:
        """
        オペランドの妥当性を検証

        Args:
            operand: 検証するオペランド

        Returns:
            妥当性（True/False）
        """
        valid, _ = self._is_valid_operand_detailed(operand)
        return valid

    def _is_valid_operand_detailed(self, operand) -> tuple[bool, str]:
        """
        オペランドの妥当性を詳細に検証

        Args:
            operand: 検証するオペランド

        Returns:
            (妥当性（True/False）, エラー詳細)
        """
        try:
            # None値は無効
            if operand is None:
                return False, "オペランドがNoneです"

            # 数値の場合は常に有効（NaNや無限大も含む）
            if isinstance(operand, (int, float)):
                return True, ""

            # 文字列の場合
            if isinstance(operand, str):
                # 空文字列は無効
                if not operand or not operand.strip():
                    return False, "オペランドが空文字列です"

                # 前後の空白を除去して処理
                operand = operand.strip()

                # 数値文字列の場合は有効
                try:
                    float(operand)
                    return True, ""
                except ValueError:
                    pass

                # 指標名またはデータソース名の場合は有効
                if (
                    self._is_indicator_name(operand)
                    or operand in self.valid_data_sources
                ):
                    return True, ""

                return (
                    False,
                    f"無効な文字列オペランド: '{operand}' (指標名でもデータソースでもありません)",
                )

            # 辞書形式のオペランド（GA生成時に発生する可能性）
            if isinstance(operand, dict):
                valid, error = self._validate_dict_operand_detailed(operand)
                return valid, error

            return False, f"サポートされていないオペランド型: {type(operand)}"

        except Exception as e:
            error_msg = f"オペランド検証エラー: {e}"
            logger.error(error_msg)
            return False, error_msg

    def _validate_dict_operand(self, operand: dict) -> bool:
        """
        辞書形式のオペランドを検証

        Args:
            operand: 辞書形式のオペランド

        Returns:
            妥当性（True/False）
        """
        valid, _ = self._validate_dict_operand_detailed(operand)
        return valid

    def _validate_dict_operand_detailed(self, operand: dict) -> tuple[bool, str]:
        """
        辞書形式のオペランドを詳細に検証

        Args:
            operand: 辞書形式のオペランド

        Returns:
            (妥当性（True/False）, エラー詳細)
        """
        try:
            # 指標タイプの辞書
            if operand.get("type") == "indicator":
                indicator_name = operand.get("name")
                if not indicator_name:
                    return False, "指標タイプの辞書にnameが設定されていません"
                if not isinstance(indicator_name, str):
                    return (
                        False,
                        f"指標名が文字列ではありません: {type(indicator_name)}",
                    )
                if self._is_indicator_name(indicator_name.strip()):
                    return True, ""
                else:
                    return False, f"無効な指標名: '{indicator_name}'"

            # 価格データタイプの辞書
            elif operand.get("type") == "price":
                price_name = operand.get("name")
                if not price_name:
                    return False, "価格タイプの辞書にnameが設定されていません"
                if not isinstance(price_name, str):
                    return False, f"価格名が文字列ではありません: {type(price_name)}"
                if price_name.strip() in self.valid_data_sources:
                    return True, ""
                else:
                    return False, f"無効な価格データソース: '{price_name}'"

            # 数値タイプの辞書
            elif operand.get("type") == "value":
                value = operand.get("value")
                if value is None:
                    return False, "数値タイプの辞書にvalueが設定されていません"
                if isinstance(value, (int, float)):
                    return True, ""
                elif isinstance(value, str):
                    try:
                        float(value.strip())
                        return True, ""
                    except ValueError:
                        return False, f"数値に変換できない文字列: '{value}'"
                else:
                    return False, f"無効な数値型: {type(value)}"

            else:
                return False, f"無効な辞書タイプ: '{operand.get('type')}'"

        except Exception as e:
            error_msg = f"辞書オペランド検証エラー: {e}"
            logger.error(error_msg)
            return False, error_msg

        # 後方互換API: 公開メソッドとして提供
        def is_valid_indicator_name(self, name: str) -> bool:
            """公開APIラッパー（旧テスト互換）"""
            return self._is_indicator_name(name)

    def _is_indicator_name(self, name: str) -> bool:
        """
        指標名かどうかを判定

        Args:
            name: 判定する名前

        Returns:
            指標名の場合True
        """
        try:
            # 空文字列や空白のみの場合は無効
            if not name or not name.strip():
                return False

            # 前後の空白を除去
            name = name.strip()

            # 基本的な指標タイプの確認
            if name in self.valid_indicator_types:
                return True

            # 指標名_パラメータ形式の確認（例: SMA_20, RSI_14, HT_DCPHASE_14）
            if "_" in name:
                # 最後のアンダースコアで分割して、パラメータ部分を除去
                parts = name.rsplit("_", 1)
                if len(parts) == 2:
                    potential_indicator = parts[0].strip()
                    potential_param = parts[1].strip()

                    # パラメータ部分が数値の場合、指標名として判定
                    try:
                        float(potential_param)
                        if potential_indicator in self.valid_indicator_types:
                            return True
                    except ValueError:
                        pass

                # 従来の方式もサポート（最初のアンダースコアで分割）
                indicator_type = name.split("_")[0].strip()
                if indicator_type in self.valid_indicator_types:
                    return True

            # 指標インデックス形式の確認（例: SMA_0, RSI_1）
            if name.endswith(("_0", "_1", "_2", "_3", "_4")):
                indicator_type = name.rsplit("_", 1)[0].strip()
                if indicator_type in self.valid_indicator_types:
                    return True

            return False

        except Exception as e:
            logger.error(f"指標名判定エラー: {e}")
            return False

    def clean_condition(self, condition) -> bool:
        """
        条件をクリーニングして修正可能な問題を自動修正

        Args:
            condition: クリーニング対象の条件

        Returns:
            クリーニング成功の場合True
        """
        try:
            # 文字列オペランドの前後空白を除去
            if isinstance(condition.left_operand, str):
                condition.left_operand = condition.left_operand.strip()

            if isinstance(condition.right_operand, str):
                condition.right_operand = condition.right_operand.strip()

            # 辞書形式のオペランドから文字列を抽出
            if isinstance(condition.left_operand, dict):
                condition.left_operand = self._extract_operand_from_dict(
                    condition.left_operand
                )

            if isinstance(condition.right_operand, dict):
                condition.right_operand = self._extract_operand_from_dict(
                    condition.right_operand
                )

            # フロントエンド用演算子を標準形式に変換
            if condition.operator == "above":
                condition.operator = ">"
            elif condition.operator == "below":
                condition.operator = "<"

            return True

        except Exception as e:
            logger.error(f"条件クリーニングエラー: {e}")
            return False

    def _extract_operand_from_dict(self, operand_dict: dict) -> str:
        """
        辞書形式のオペランドから文字列を抽出

        Args:
            operand_dict: 辞書形式のオペランド

        Returns:
            抽出された文字列オペランド
        """
        try:
            if operand_dict.get("type") == "indicator":
                return operand_dict.get("name", "")
            elif operand_dict.get("type") == "price":
                return operand_dict.get("name", "")
            elif operand_dict.get("type") == "value":
                value = operand_dict.get("value")
                return str(value) if value is not None else ""
            else:
                return str(operand_dict.get("name", ""))

        except Exception as e:
            logger.error(f"辞書オペランド抽出エラー: {e}")
            return ""

    def validate_strategy_gene(self, strategy_gene) -> Tuple[bool, List[str]]:
        """
        戦略遺伝子の妥当性を検証

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        try:
            # 指標数の制約チェック
            max_indicators = getattr(strategy_gene, "MAX_INDICATORS", 5)
            if len(strategy_gene.indicators) > max_indicators:
                errors.append(
                    f"指標数が上限({max_indicators})を超えています: {len(strategy_gene.indicators)}"
                )

            # 指標の妥当性チェック
            for i, indicator in enumerate(strategy_gene.indicators):
                if not self.validate_indicator_gene(indicator):
                    errors.append(f"指標{i}が無効です: {indicator.type}")

            # 条件の妥当性チェック（クリーニング付き）
            from .condition_group import ConditionGroup

            def _validate_mixed_conditions(cond_list, label_prefix: str):
                for i, condition in enumerate(cond_list):
                    if isinstance(condition, ConditionGroup):
                        # ORグループ内の各条件を検証
                        for j, c in enumerate(condition.conditions):
                            self.clean_condition(c)
                            is_valid, error_detail = self.validate_condition(c)
                            if not is_valid:
                                errors.append(
                                    f"{label_prefix}OR子条件{j}が無効です: {error_detail}"
                                )
                    else:
                        self.clean_condition(condition)
                        is_valid, error_detail = self.validate_condition(condition)
                        if not is_valid:
                            errors.append(
                                f"{label_prefix}{i}が無効です: {error_detail}"
                            )

            _validate_mixed_conditions(strategy_gene.entry_conditions, "エントリー条件")
            # ロング・ショート条件の妥当性チェック（クリーニング付き）
            _validate_mixed_conditions(
                strategy_gene.long_entry_conditions, "ロングエントリー条件"
            )
            _validate_mixed_conditions(
                strategy_gene.short_entry_conditions, "ショートエントリー条件"
            )

            for i, condition in enumerate(strategy_gene.exit_conditions):
                # 条件をクリーニング
                self.clean_condition(condition)
                is_valid, error_detail = self.validate_condition(condition)
                if not is_valid:
                    errors.append(f"イグジット条件{i}が無効です: {error_detail}")

            # 最低限の条件チェック（ロング・ショート条件も考慮）
            has_entry_conditions = (
                bool(strategy_gene.entry_conditions)
                or bool(strategy_gene.long_entry_conditions)
                or bool(strategy_gene.short_entry_conditions)
            )
            if not has_entry_conditions:
                errors.append(
                    "エントリー条件が設定されていません（entry_conditions、long_entry_conditions、short_entry_conditionsのいずれかが必要）"
                )

            # TP/SL遺伝子が有効な場合はイグジット条件は不要
            if not strategy_gene.exit_conditions:
                # TP/SL遺伝子が有効でない場合のみエラーとする
                if not (strategy_gene.tpsl_gene and strategy_gene.tpsl_gene.enabled):
                    errors.append("イグジット条件が設定されていません")

            # 有効な指標の存在チェック
            enabled_indicators = [
                ind for ind in strategy_gene.indicators if ind.enabled
            ]
            if not enabled_indicators:
                errors.append("有効な指標が設定されていません")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"戦略遺伝子バリデーションエラー: {e}")
            errors.append(f"バリデーション処理エラー: {e}")
            return False, errors

    def validate_risk_management(self, risk_management: dict) -> Tuple[bool, List[str]]:
        """
        リスク管理設定の妥当性を検証

        Args:
            risk_management: リスク管理設定

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        try:
            # ストップロスの検証
            if "stop_loss" in risk_management:
                stop_loss = risk_management["stop_loss"]
                if (
                    not isinstance(stop_loss, (int, float))
                    or stop_loss <= 0
                    or stop_loss >= 1
                ):
                    errors.append("ストップロスは0-1の範囲で設定してください")

            # テイクプロフィットの検証
            if "take_profit" in risk_management:
                take_profit = risk_management["take_profit"]
                if not isinstance(take_profit, (int, float)) or take_profit <= 0:
                    errors.append("テイクプロフィットは正の値で設定してください")

            # ポジションサイズの検証
            if "position_size" in risk_management:
                position_size = risk_management["position_size"]
                if (
                    not isinstance(position_size, (int, float))
                    or position_size <= 0
                    or position_size > 1
                ):
                    errors.append("ポジションサイズは0-1の範囲で設定してください")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"リスク管理バリデーションエラー: {e}")
            errors.append(f"リスク管理バリデーション処理エラー: {e}")
            return False, errors

    def validate_metadata(self, metadata: dict) -> Tuple[bool, List[str]]:
        """
        メタデータの妥当性を検証

        Args:
            metadata: メタデータ

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        try:
            # フィットネススコアの検証
            if "fitness_score" in metadata:
                fitness_score = metadata["fitness_score"]
                if not isinstance(fitness_score, (int, float)):
                    errors.append("フィットネススコアは数値で設定してください")

            # 生成日時の検証
            if "created_at" in metadata:
                created_at = metadata["created_at"]
                if not isinstance(created_at, str):
                    errors.append("生成日時は文字列で設定してください")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"メタデータバリデーションエラー: {e}")
            errors.append(f"メタデータバリデーション処理エラー: {e}")
            return False, errors

    def get_validation_summary(self, strategy_gene) -> dict:
        """
        バリデーション結果のサマリーを取得

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            バリデーション結果のサマリー
        """
        try:
            is_valid, errors = self.validate_strategy_gene(strategy_gene)

            # リスク管理とメタデータの検証
            risk_valid, risk_errors = self.validate_risk_management(
                strategy_gene.risk_management
            )
            metadata_valid, metadata_errors = self.validate_metadata(
                strategy_gene.metadata
            )

            return {
                "overall_valid": is_valid and risk_valid and metadata_valid,
                "strategy_valid": is_valid,
                "risk_management_valid": risk_valid,
                "metadata_valid": metadata_valid,
                "errors": errors + risk_errors + metadata_errors,
                "indicator_count": len(strategy_gene.indicators),
                "enabled_indicator_count": len(
                    [ind for ind in strategy_gene.indicators if ind.enabled]
                ),
                "entry_condition_count": len(strategy_gene.entry_conditions),
                "exit_condition_count": len(strategy_gene.exit_conditions),
            }

        except Exception as e:
            logger.error(f"バリデーションサマリー作成エラー: {e}")
            return {
                "overall_valid": False,
                "strategy_valid": False,
                "risk_management_valid": False,
                "metadata_valid": False,
                "errors": [f"バリデーションサマリー作成エラー: {e}"],
                "indicator_count": 0,
                "enabled_indicator_count": 0,
                "entry_condition_count": 0,
                "exit_condition_count": 0,
            }
