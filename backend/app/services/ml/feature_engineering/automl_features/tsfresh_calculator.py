"""
TSFresh特徴量計算クラス

時系列データから100以上の統計的特徴量を自動生成し、
仮説検定による特徴量選択を実行します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings

from .....utils.unified_error_handler import safe_ml_operation

from .automl_config import TSFreshConfig
from .feature_settings import FinancialFeatureSettings, MarketRegime
from .feature_selector import AdvancedFeatureSelector
from .performance_optimizer import PerformanceOptimizer


from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


logger = logging.getLogger(__name__)


class TSFreshFeatureCalculator:
    """
    TSFresh特徴量計算クラス

    時系列データから統計的特徴量を自動生成します。
    """

    def __init__(self, config: Optional[TSFreshConfig] = None):
        """
        初期化

        Args:
            config: TSFresh設定
        """
        self.config = config or TSFreshConfig()
        self.feature_cache = {}  # 使用しないが互換性のため保持
        self.selected_features = None
        self.last_extraction_info = {}
        self.feature_settings = FinancialFeatureSettings()
        self.current_market_regime = MarketRegime.TRENDING  # デフォルト
        self.feature_selector = AdvancedFeatureSelector()
        self.performance_optimizer = PerformanceOptimizer()
        self.feature_settings = FinancialFeatureSettings()
        self.current_market_regime = MarketRegime.TRENDING  # デフォルト

    @safe_ml_operation(
        default_return=None, context="TSFresh特徴量計算でエラーが発生しました"
    )
    def calculate_tsfresh_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        feature_selection: Optional[bool] = None,
        custom_settings: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        TSFresh特徴量を計算

        Args:
            df: OHLCV価格データ
            target: ターゲット変数（特徴量選択用）
            feature_selection: 特徴量選択を実行するか
            custom_settings: カスタム特徴量設定

        Returns:
            TSFresh特徴量が追加されたDataFrame
        """

        if df is None or df.empty:
            logger.warning("空のデータが提供されました")
            return df

        try:
            # 設定の決定
            use_feature_selection = (
                feature_selection
                if feature_selection is not None
                else self.config.feature_selection
            )

            logger.info("TSFresh特徴量抽出を開始...")

            # 特徴量設定（適応的設定を使用）
            if custom_settings is None:
                # データに適応した設定を取得
                performance_mode = getattr(self.config, "performance_mode", "balanced")
                settings = self.get_adaptive_settings(df, performance_mode)
                logger.info(f"適応的設定を使用: {len(settings)}種類の特徴量")
            else:
                settings = custom_settings
                logger.info("カスタム設定を使用")

            # データを時系列形式に変換（キャッシュの有無に関わらず必要）
            ts_data = self._prepare_timeseries_data(df)

            if ts_data.empty:
                logger.warning("時系列データの変換に失敗しました")
                return df

            # キャッシュ機能を無効化（毎回新鮮な特徴量を生成）
            logger.info("キャッシュを使用せず、新しい特徴量を生成します")

            # ランダム性を追加して毎回異なる特徴量を生成
            import random

            random.seed()  # 現在時刻でシードをリセット

            # システムリソースを監視
            self.performance_optimizer.monitor_system_resources()

            # 最適化提案を取得
            optimization_suggestions = self.performance_optimizer.suggest_optimization(
                len(df), len(settings)
            )

            # 並列処理数を調整
            n_jobs = min(
                optimization_suggestions.get(
                    "parallel_jobs", self.config.parallel_jobs
                ),
                self.config.parallel_jobs,
            )

            # 常に新しい特徴量を抽出
            # 警告を抑制
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 特徴量抽出
                extracted_features = extract_features(
                    ts_data,
                    column_id="id",
                    column_sort="time",
                    default_fc_parameters=settings,
                    impute_function=impute,
                    n_jobs=n_jobs,
                    disable_progressbar=True,
                )

                # キャッシュ保存を無効化（毎回新鮮な特徴量生成のため）
                logger.debug("キャッシュ保存をスキップしました")

                # メモリ最適化
                if optimization_suggestions.get("memory_optimization", False):
                    extracted_features = (
                        self.performance_optimizer.optimize_dataframe_memory(
                            extracted_features
                        )
                    )

                # ガベージコレクション
                self.performance_optimizer.force_garbage_collection()

            logger.info(f"TSFresh特徴量抽出完了: {len(extracted_features.columns)}個")

            # 特徴量選択（ターゲットがある場合）
            if use_feature_selection and target is not None:
                logger.info("高度特徴量選択を開始...")

                # ターゲットのインデックスを調整
                target_aligned = self._align_target_with_features(
                    target, extracted_features
                )

                if target_aligned is not None:
                    # 高度特徴量選択を実行
                    selected_features, selection_info = (
                        self.feature_selector.select_features_comprehensive(
                            extracted_features,
                            target_aligned,
                            max_features=self.config.feature_count_limit,
                            selection_methods=[
                                "statistical_test",
                                "correlation_filter",
                                "importance_based",
                            ],
                        )
                    )

                    self.selected_features = selected_features.columns.tolist()
                    result_features = selected_features

                    # 選択情報を抽出情報に追加
                    self.last_extraction_info["selection_info"] = selection_info

                    logger.info(
                        f"高度特徴量選択完了: {len(selected_features.columns)}個を選択"
                    )
                else:
                    logger.warning(
                        "ターゲット変数の調整に失敗しました。選択なしで続行します"
                    )
                    result_features = extracted_features
            else:
                result_features = extracted_features

            # 特徴量数制限
            if len(result_features.columns) > self.config.feature_count_limit:
                logger.info(
                    f"特徴量数を{self.config.feature_count_limit}個に制限します"
                )
                # 分散の大きい特徴量を優先的に選択
                feature_variances = result_features.var().sort_values(ascending=False)
                selected_cols = feature_variances.head(
                    self.config.feature_count_limit
                ).index
                result_features = result_features[selected_cols]

            # 元のDataFrameに結合
            result_df = self._merge_features_with_original(df, result_features)

            # 抽出情報を保存
            self.last_extraction_info = {
                "total_extracted": len(extracted_features.columns),
                "final_count": len(result_features.columns),
                "feature_selection_used": use_feature_selection and target is not None,
                "settings_used": len(settings),
            }

            logger.info(
                f"TSFresh特徴量生成完了: 総計{len(result_features.columns)}個の特徴量"
            )
            return result_df

        except Exception as e:
            logger.error(f"TSFresh特徴量計算エラー: {e}")
            return df

    def _prepare_timeseries_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """時系列データをTSFresh形式に変換"""
        try:
            ts_data = []

            # 各価格系列を個別の時系列として扱う
            price_columns = ["Open", "High", "Low", "Close", "Volume"]
            available_columns = [col for col in price_columns if col in df.columns]

            if not available_columns:
                logger.warning("利用可能な価格データ列が見つかりません")
                return pd.DataFrame()

            for col in available_columns:
                series_data = df[col].dropna()
                for i, (timestamp, value) in enumerate(series_data.items()):
                    if pd.notna(value) and np.isfinite(value):
                        ts_data.append({"id": col, "time": i, "value": float(value)})

            result_df = pd.DataFrame(ts_data)
            logger.debug(f"時系列データ変換完了: {len(result_df)}行")
            return result_df

        except Exception as e:
            logger.error(f"時系列データ変換エラー: {e}")
            return pd.DataFrame()

    def _get_financial_feature_settings(self) -> Dict:
        """金融データ用の特徴量設定"""
        # 金融時系列に適した特徴量のみを選択
        # TSFreshで利用可能で安定した特徴量のみを使用
        financial_settings = {
            # 基本統計的特徴量（大幅拡張）
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "variance": None,
            "minimum": None,
            "maximum": None,
            "sum_values": None,
            "abs_energy": None,
            "mean_abs_change": None,
            "mean_change": None,
            "mean_second_derivative_central": None,
            "root_mean_square": None,
            # 分位点（大幅拡張）
            "quantile": [
                {"q": 0.1},
                {"q": 0.2},
                {"q": 0.25},
                {"q": 0.3},
                {"q": 0.4},
                {"q": 0.6},
                {"q": 0.7},
                {"q": 0.75},
                {"q": 0.8},
                {"q": 0.9},
            ],
            # 自己相関（大幅拡張）
            "autocorrelation": [
                {"lag": i} for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
            ],
            # 部分自己相関
            "partial_autocorrelation": [{"lag": i} for i in [1, 2, 3, 4, 5, 10]],
            # トレンド分析（拡張）
            "linear_trend": [
                {"attr": "slope"},
                {"attr": "intercept"},
                {"attr": "rvalue"},
                {"attr": "stderr"},
            ],
            # 周波数領域特徴量
            "fft_coefficient": [{"coeff": i, "attr": "real"} for i in range(8)]
            + [{"coeff": i, "attr": "imag"} for i in range(8)]
            + [{"coeff": i, "attr": "abs"} for i in range(8)],
            # エネルギー・パワー特徴量
            "energy_ratio_by_chunks": [
                {"num_segments": 10, "segment_focus": i} for i in range(5)
            ],
            "fft_aggregated": [
                {"aggtype": "centroid"},
                {"aggtype": "variance"},
                {"aggtype": "skew"},
                {"aggtype": "kurtosis"},
            ],
            # 長さ・カウント統計（拡張）
            "count_above_mean": None,
            "count_below_mean": None,
            "longest_strike_above_mean": None,
            "longest_strike_below_mean": None,
            "count_above": [{"t": 0}],
            "count_below": [{"t": 0}],
            # 範囲統計（拡張）
            "range_count": [
                {"min": -2, "max": 2},
                {"min": -1.5, "max": 1.5},
                {"min": -1, "max": 1},
                {"min": -0.5, "max": 0.5},
            ],
            # 変化点・極値検出
            "number_peaks": [{"n": i} for i in [1, 3, 5, 10]],
            "ratio_beyond_r_sigma": [{"r": i} for i in [0.5, 1, 1.5, 2, 2.5, 3]],
            # 複雑性・エントロピー
            "sample_entropy": None,
            "approximate_entropy": [{"m": 2, "r": 0.1}, {"m": 2, "r": 0.3}],
            # 対称性・歪み
            "symmetry_looking": [{"r": 0.1}, {"r": 0.2}],
            "c3": [{"lag": i} for i in [1, 2, 3]],
            "cid_ce": [{"normalize": True}, {"normalize": False}],
            # 時系列の特性
            "has_duplicate_max": None,
            "has_duplicate_min": None,
            "has_duplicate": None,
            "ratio_value_number_to_time_series_length": None,
            # AR係数
            "ar_coefficient": [{"coeff": i, "k": 10} for i in range(3)],
            "agg_autocorrelation": [
                {"f_agg": "mean", "maxlag": 20},
                {"f_agg": "median", "maxlag": 20},
                {"f_agg": "var", "maxlag": 20},
            ],
        }

        return financial_settings

    def _align_target_with_features(
        self, target: pd.Series, features: pd.DataFrame
    ) -> Optional[pd.Series]:
        """ターゲット変数を特徴量のインデックスに合わせる"""
        try:
            if len(target) != len(features):
                logger.warning(
                    f"ターゲット変数の長さ({len(target)})と特徴量の長さ({len(features)})が一致しません"
                )
                # 短い方に合わせる
                min_length = min(len(target), len(features))
                target_aligned = target.iloc[:min_length]
                target_aligned.index = features.index[:min_length]
            else:
                target_aligned = target.copy()
                target_aligned.index = features.index

            # NaNを除去
            valid_mask = target_aligned.notna() & features.notna().all(axis=1)
            target_aligned = target_aligned[valid_mask]

            if len(target_aligned) == 0:
                logger.warning("有効なターゲット変数が見つかりません")
                return None

            return target_aligned

        except Exception as e:
            logger.error(f"ターゲット変数の調整エラー: {e}")
            return None

    def _merge_features_with_original(
        self, original_df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """特徴量を元のDataFrameに結合"""
        try:
            result_df = original_df.copy()

            if features.empty:
                logger.warning("TSFresh特徴量が空です")
                return result_df

            # インデックスを合わせて結合
            if len(features) == len(original_df):
                for col in features.columns:
                    # TSFreshプレフィックスを追加
                    new_col_name = f"TSF_{col}"
                    result_df[new_col_name] = features[col].values
                logger.info(
                    f"TSFresh特徴量 {len(features.columns)}個を正常に結合しました"
                )
            else:
                logger.warning(
                    f"特徴量の長さ({len(features)})と元データの長さ({len(original_df)})が一致しません"
                )

                # 特徴量が短い場合の対処
                if len(features) < len(original_df):
                    logger.info(
                        f"特徴量を{len(features)}から{len(original_df)}に拡張します"
                    )

                    # 各列について処理
                    for col in features.columns:
                        new_col_name = f"TSF_{col}"

                        # 元の特徴量の値を取得
                        feature_values = features[col].values

                        # 最後の値で埋める
                        if len(feature_values) > 0:
                            last_value = feature_values[-1]
                        else:
                            last_value = 0

                        # 全長のデータを作成
                        full_data = np.full(len(original_df), last_value, dtype=float)

                        # 元の特徴量の値を先頭に配置
                        full_data[: len(feature_values)] = feature_values

                        # 結果DataFrameに追加
                        result_df[new_col_name] = full_data

                    logger.info(
                        f"TSFresh特徴量 {len(features.columns)}個を拡張して結合しました"
                    )
                else:
                    # 特徴量が長い場合は切り詰める
                    for col in features.columns:
                        new_col_name = f"TSF_{col}"
                        result_df[new_col_name] = (
                            features[col].iloc[: len(original_df)].values
                        )

                    logger.info(
                        f"TSFresh特徴量 {len(features.columns)}個を切り詰めて結合しました"
                    )

            return result_df

        except Exception as e:
            logger.error(f"特徴量結合エラー: {e}")
            # エラーが発生した場合は元のDataFrameを返す
            return original_df

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        if self.selected_features:
            return [f"TSF_{name}" for name in self.selected_features]
        else:
            # デフォルトの特徴量名（推定）
            return [
                "TSF_mean",
                "TSF_std",
                "TSF_skewness",
                "TSF_kurtosis",
                "TSF_autocorr_1",
                "TSF_autocorr_5",
                "TSF_autocorr_10",
                "TSF_fft_coeff_0_real",
                "TSF_fft_coeff_1_real",
                "TSF_sample_entropy",
                "TSF_linear_trend_slope",
                "TSF_number_peaks_3",
                "TSF_change_quantiles_low",
                "TSF_range_count",
                "TSF_count_above_mean",
            ]

    def get_extraction_info(self) -> Dict[str, Any]:
        """最後の抽出情報を取得"""
        return self.last_extraction_info.copy()

    def clear_cache(self):
        """キャッシュをクリア"""
        self.feature_cache.clear()
        logger.debug("TSFresh特徴量キャッシュをクリアしました")

    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """市場レジームを検出"""
        try:
            if "Close" not in df.columns or len(df) < 20:
                return MarketRegime.TRENDING

            # 価格変動の統計を計算
            returns = df["Close"].pct_change().dropna()
            volatility = returns.std()
            trend_strength = abs(returns.mean()) / volatility if volatility > 0 else 0

            # ボラティリティ閾値
            vol_threshold_high = 0.03  # 3%
            vol_threshold_low = 0.01  # 1%

            # トレンド強度閾値
            trend_threshold = 0.1

            if volatility > vol_threshold_high:
                return MarketRegime.VOLATILE
            elif volatility < vol_threshold_low:
                return MarketRegime.CALM
            elif trend_strength > trend_threshold:
                return MarketRegime.TRENDING
            else:
                return MarketRegime.RANGING

        except Exception as e:
            logger.warning(f"市場レジーム検出エラー: {e}")
            return MarketRegime.TRENDING

    def get_adaptive_settings(
        self, df: pd.DataFrame, performance_mode: str = "balanced"
    ) -> Dict:
        """データに適応した設定を取得"""
        try:
            # 市場レジームを検出
            regime = self.detect_market_regime(df)
            self.current_market_regime = regime

            logger.info(f"検出された市場レジーム: {regime.value}")

            # パフォーマンスモードに応じた設定選択
            if performance_mode == "fast":
                return self.feature_settings.get_lightweight_settings()
            elif performance_mode == "comprehensive":
                return self.feature_settings.get_comprehensive_settings()
            elif performance_mode == "financial_optimized":
                return self.feature_settings.get_financial_optimized_settings()
            else:  # balanced
                # 市場レジームに適したプロファイルを選択
                # get_market_regime_profilesメソッドが存在しない場合のフォールバック
                try:
                    suitable_profiles = (
                        self.feature_settings.get_market_regime_profiles(regime)
                    )
                except AttributeError:
                    # メソッドが存在しない場合は基本設定を使用
                    logger.warning(
                        "get_market_regime_profilesメソッドが見つかりません。基本設定を使用します。"
                    )
                    return self._get_financial_feature_settings()

                # 計算コストを考慮して選択
                selected_profiles = []
                total_cost = 0
                max_cost = 25  # 最大計算コスト

                # プロファイルが存在するかチェック
                if not hasattr(self.feature_settings, "profiles"):
                    logger.warning(
                        "プロファイル情報が見つかりません。基本設定を使用します。"
                    )
                    return self._get_financial_feature_settings()

                # コストの低い順にソート
                profile_costs = [
                    (name, self.feature_settings.profiles[name].computational_cost)
                    for name in suitable_profiles
                    if name in self.feature_settings.profiles
                ]
                profile_costs.sort(key=lambda x: x[1])

                for profile_name, cost in profile_costs:
                    if total_cost + cost <= max_cost:
                        selected_profiles.append(profile_name)
                        total_cost += cost

                if selected_profiles:
                    logger.info(
                        f"選択されたプロファイル: {selected_profiles} (総コスト: {total_cost})"
                    )
                    try:
                        return self.feature_settings.get_combined_settings(
                            selected_profiles
                        )
                    except AttributeError:
                        logger.warning(
                            "get_combined_settingsメソッドが見つかりません。基本設定を使用します。"
                        )
                        return self._get_financial_feature_settings()
                else:
                    logger.info(
                        "適切なプロファイルが見つかりません。基本設定を使用します。"
                    )
                    return self._get_financial_feature_settings()

        except Exception as e:
            logger.error(f"適応設定取得エラー: {e}")
            return self._get_financial_feature_settings()

    def set_market_regime(self, regime: MarketRegime):
        """市場レジームを手動設定"""
        self.current_market_regime = regime
        logger.info(f"市場レジームを手動設定: {regime.value}")

    def get_regime_info(self) -> Dict[str, Any]:
        """現在の市場レジーム情報を取得"""
        return {
            "current_regime": self.current_market_regime.value,
            "suitable_profiles": self.feature_settings.get_market_regime_profiles(
                self.current_market_regime
            ),
            "regime_description": {
                MarketRegime.TRENDING: "明確なトレンドが存在する市場",
                MarketRegime.RANGING: "レンジ相場、横ばい市場",
                MarketRegime.VOLATILE: "高ボラティリティ市場",
                MarketRegime.CALM: "低ボラティリティ、安定市場",
            }.get(self.current_market_regime, "不明"),
        }
