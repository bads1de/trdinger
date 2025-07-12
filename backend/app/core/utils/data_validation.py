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
    
    # 異常値の閾値設定
    MAX_VALUE_THRESHOLD = 1e10  # 最大値の閾値
    MIN_VALUE_THRESHOLD = -1e10  # 最小値の閾値
    
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
            mean = data.rolling(window=window).mean()
            std = data.rolling(window=window).std()
            
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
