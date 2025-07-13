"""
データバリデーションユーティリティ

特徴量データの妥当性チェックとクリーンアップ機能を提供します。
無限大値、NaN値、異常に大きな値の検出と処理を行います。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class DataValidator:
    """
    データバリデーションクラス
    
    特徴量データの妥当性チェックとクリーンアップを行います。
    """
    
    # 異常値の閾値設定（金融データに適した範囲に調整）
    MAX_VALUE_THRESHOLD = 1e6   # 最大値の閾値
    MIN_VALUE_THRESHOLD = -1e6  # 最小値の閾値
    
    def __init__(self):
        """初期化"""
        pass
    
    @staticmethod
    def safe_divide(numerator: Union[pd.Series, np.ndarray, float], 
                   denominator: Union[pd.Series, np.ndarray, float], 
                   default_value: float = 0.0,
                   epsilon: float = 1e-8) -> Union[pd.Series, np.ndarray, float]:
        """
        安全な除算処理
        
        Args:
            numerator: 分子
            denominator: 分母
            default_value: 分母が0の場合のデフォルト値
            epsilon: 分母に加える小さな値
            
        Returns:
            除算結果
        """
        try:
            # 分母にepsilonを加えて0除算を防ぐ
            safe_denominator = denominator + epsilon
            result = numerator / safe_denominator
            
            # 無限大値をデフォルト値に置換
            if isinstance(result, (pd.Series, np.ndarray)):
                result = np.where(np.isinf(result), default_value, result)
                result = np.where(np.isnan(result), default_value, result)
            elif np.isinf(result) or np.isnan(result):
                result = default_value
                
            return result
            
        except Exception as e:
            logger.warning(f"除算処理でエラーが発生しました: {e}")
            if isinstance(numerator, (pd.Series, np.ndarray)):
                return np.full_like(numerator, default_value, dtype=float)
            else:
                return default_value
    
    @staticmethod
    def safe_correlation(x: pd.Series, y: pd.Series,
                        window: int, default_value: float = 0.0) -> pd.Series:
        """
        安全な相関計算

        Args:
            x: 系列1
            y: 系列2
            window: ウィンドウサイズ
            default_value: NaN/無限大の場合のデフォルト値

        Returns:
            相関係数の系列
        """
        try:
            correlation = x.rolling(window=window).corr(y)

            # NaN値と無限大値をデフォルト値に置換
            correlation = correlation.fillna(default_value)
            correlation = np.where(np.isinf(correlation), default_value, correlation)

            return correlation

        except Exception as e:
            logger.warning(f"相関計算でエラーが発生しました: {e}")
            return pd.Series(np.full(len(x), default_value), index=x.index)

    @staticmethod
    def safe_multiply(a: Union[pd.Series, float],
                     b: Union[pd.Series, float],
                     default_value: float = 0.0) -> Union[pd.Series, float]:
        """
        安全な乗算処理（無限値やNaN値を避ける）

        Args:
            a: 第1オペランド
            b: 第2オペランド
            default_value: 無限値やNaN値の場合のデフォルト値

        Returns:
            乗算結果
        """
        try:
            result = a * b

            if isinstance(result, pd.Series):
                # 無限値とNaN値を置換
                result = result.replace([np.inf, -np.inf], default_value)
                result = result.fillna(default_value)
            else:
                # スカラーの場合
                if np.isinf(result) or np.isnan(result):
                    result = default_value

            return result

        except Exception as e:
            logger.warning(f"乗算処理でエラーが発生しました: {e}")
            if isinstance(a, pd.Series):
                return pd.Series(np.full(len(a), default_value), index=a.index)
            else:
                return default_value
    
    @staticmethod
    def safe_pct_change(data: pd.Series, periods: int = 1,
                       fill_value: float = 0.0, min_threshold: float = 1e-10) -> pd.Series:
        """
        安全な変化率計算（0除算を避ける）

        Args:
            data: 変化率を計算するデータ
            periods: 期間
            fill_value: 無限値やNaN値の置換値
            min_threshold: 除算の最小閾値

        Returns:
            安全に計算された変化率
        """
        try:
            # 前の値を取得
            prev_data = data.shift(periods)

            # 0や極小値を閾値で置換
            prev_data_safe = prev_data.where(
                np.abs(prev_data) >= min_threshold,
                min_threshold * np.sign(prev_data).replace(0, 1)
            )

            # 変化率を計算
            pct_change = (data - prev_data) / prev_data_safe

            # 無限値とNaN値を置換
            pct_change = pct_change.replace([np.inf, -np.inf], fill_value)
            pct_change = pct_change.fillna(fill_value)

            return pct_change

        except Exception as e:
            logger.warning(f"変化率計算でエラーが発生しました: {e}")
            return pd.Series(np.full(len(data), fill_value), index=data.index)

    @staticmethod
    def safe_rolling_mean(data: pd.Series, window: int,
                         min_periods: int = 1, fill_value: float = 0.0) -> pd.Series:
        """
        安全なローリング平均計算

        Args:
            data: 計算対象データ
            window: ウィンドウサイズ
            min_periods: 最小期間
            fill_value: NaN値の置換値

        Returns:
            安全に計算されたローリング平均
        """
        try:
            rolling_mean = data.rolling(window=window, min_periods=min_periods).mean()
            rolling_mean = rolling_mean.fillna(fill_value)
            return rolling_mean

        except Exception as e:
            logger.warning(f"ローリング平均計算でエラーが発生しました: {e}")
            return pd.Series(np.full(len(data), fill_value), index=data.index)

    @staticmethod
    def safe_rolling_std(data: pd.Series, window: int,
                        min_periods: int = 1, fill_value: float = 0.0) -> pd.Series:
        """
        安全なローリング標準偏差計算

        Args:
            data: 計算対象データ
            window: ウィンドウサイズ
            min_periods: 最小期間
            fill_value: NaN値の置換値

        Returns:
            安全に計算されたローリング標準偏差
        """
        try:
            rolling_std = data.rolling(window=window, min_periods=min_periods).std()
            rolling_std = rolling_std.fillna(fill_value)
            return rolling_std

        except Exception as e:
            logger.warning(f"ローリング標準偏差計算でエラーが発生しました: {e}")
            return pd.Series(np.full(len(data), fill_value), index=data.index)

    @staticmethod
    def safe_normalize(data: pd.Series, window: int,
                      default_value: float = 0.0) -> pd.Series:
        """
        安全な正規化処理（Z-score）

        Args:
            data: 正規化するデータ
            window: 計算ウィンドウ
            default_value: 標準偏差が0の場合のデフォルト値

        Returns:
            正規化されたデータ
        """
        try:
            mean = DataValidator.safe_rolling_mean(data, window)
            std = DataValidator.safe_rolling_std(data, window)

            # 標準偏差が0の場合を考慮
            normalized = DataValidator.safe_divide(
                data - mean, std, default_value=default_value
            )

            return normalized

        except Exception as e:
            logger.warning(f"正規化処理でエラーが発生しました: {e}")
            return pd.Series(np.full(len(data), default_value), index=data.index)
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, 
                          column_names: Optional[List[str]] = None) -> Tuple[bool, Dict[str, List[str]]]:
        """
        DataFrameの妥当性チェック
        
        Args:
            df: チェック対象のDataFrame
            column_names: チェック対象のカラム名（Noneの場合は全カラム）
            
        Returns:
            (妥当性フラグ, 問題のあるカラムの詳細)
        """
        if df is None or df.empty:
            return False, {"empty": ["DataFrame is empty or None"]}
        
        issues = {
            "infinite": [],
            "nan": [],
            "too_large": [],
            "too_small": []
        }
        
        check_columns = column_names if column_names else df.select_dtypes(include=[np.number]).columns
        
        for col in check_columns:
            if col not in df.columns:
                continue
                
            series = df[col]
            
            # 無限大値のチェック
            if np.isinf(series).any():
                issues["infinite"].append(col)
            
            # NaN値のチェック
            if series.isna().any():
                issues["nan"].append(col)
            
            # 異常に大きな値のチェック
            if (series > cls.MAX_VALUE_THRESHOLD).any():
                issues["too_large"].append(col)
            
            # 異常に小さな値のチェック
            if (series < cls.MIN_VALUE_THRESHOLD).any():
                issues["too_small"].append(col)
        
        is_valid = not any(issues.values())
        return is_valid, issues
    
    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame, 
                       column_names: Optional[List[str]] = None,
                       fill_method: str = "median") -> pd.DataFrame:
        """
        DataFrameのクリーンアップ
        
        Args:
            df: クリーンアップ対象のDataFrame
            column_names: クリーンアップ対象のカラム名（Noneの場合は全カラム）
            fill_method: 欠損値の補完方法（median, mean, zero, forward_fill）
            
        Returns:
            クリーンアップされたDataFrame
        """
        if df is None or df.empty:
            return df
        
        result_df = df.copy()
        check_columns = column_names if column_names else df.select_dtypes(include=[np.number]).columns
        
        for col in check_columns:
            if col not in result_df.columns:
                continue
            
            series = result_df[col]
            
            # 無限大値を NaN に変換
            series = series.replace([np.inf, -np.inf], np.nan)
            
            # 異常に大きな値を NaN に変換
            series = series.where(
                (series <= cls.MAX_VALUE_THRESHOLD) & (series >= cls.MIN_VALUE_THRESHOLD),
                np.nan
            )
            
            # 欠損値の補完
            if fill_method == "median":
                fill_value = series.median()
            elif fill_method == "mean":
                fill_value = series.mean()
            elif fill_method == "zero":
                fill_value = 0.0
            elif fill_method == "forward_fill":
                series = series.ffill()
                fill_value = 0.0  # 最初の値がNaNの場合のフォールバック
            else:
                fill_value = 0.0
            
            # まだNaNが残っている場合は指定された値で補完
            if fill_method != "forward_fill":
                series = series.fillna(fill_value)
            else:
                series = series.fillna(fill_value)
            
            result_df[col] = series
        
        return result_df
    
    @classmethod
    def log_validation_results(cls, validation_results: Tuple[bool, Dict[str, List[str]]], 
                              dataframe_name: str = "DataFrame") -> None:
        """
        バリデーション結果をログ出力
        
        Args:
            validation_results: validate_dataframeの結果
            dataframe_name: DataFrameの名前（ログ用）
        """
        is_valid, issues = validation_results
        
        if is_valid:
            logger.debug(f"{dataframe_name}: データは正常です")
        else:
            logger.warning(f"{dataframe_name}: データに問題があります")
            
            for issue_type, columns in issues.items():
                if columns:
                    logger.warning(f"  {issue_type}: {columns}")
    
    @classmethod
    def validate_and_clean(cls, df: pd.DataFrame, 
                          column_names: Optional[List[str]] = None,
                          fill_method: str = "median",
                          log_name: str = "DataFrame") -> pd.DataFrame:
        """
        バリデーションとクリーンアップを一括実行
        
        Args:
            df: 対象のDataFrame
            column_names: 対象カラム名
            fill_method: 補完方法
            log_name: ログ用の名前
            
        Returns:
            クリーンアップされたDataFrame
        """
        # バリデーション実行
        validation_results = cls.validate_dataframe(df, column_names)
        cls.log_validation_results(validation_results, log_name)
        
        # クリーンアップ実行
        cleaned_df = cls.clean_dataframe(df, column_names, fill_method)
        
        # クリーンアップ後の再バリデーション
        post_validation = cls.validate_dataframe(cleaned_df, column_names)
        if not post_validation[0]:
            logger.error(f"{log_name}: クリーンアップ後もデータに問題があります")
            cls.log_validation_results(post_validation, f"{log_name} (クリーンアップ後)")
        
        return cleaned_df
