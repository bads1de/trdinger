"""
データドリフト検出システム

統計的手法を使用してデータ分布の変化を検出し、
モデル性能劣化の早期警告を提供します。
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """ドリフトタイプ"""
    NO_DRIFT = "no_drift"
    MILD_DRIFT = "mild_drift"
    MODERATE_DRIFT = "moderate_drift"
    SEVERE_DRIFT = "severe_drift"


@dataclass
class DriftDetectionResult:
    """ドリフト検出結果"""
    feature_name: str
    drift_type: DriftType
    drift_score: float
    p_value: float
    threshold: float
    test_method: str
    timestamp: datetime
    reference_period: str
    current_period: str
    recommendation: str


@dataclass
class DriftMonitoringConfig:
    """ドリフト監視設定"""
    reference_window_days: int = 30
    detection_window_days: int = 7
    mild_threshold: float = 0.05
    moderate_threshold: float = 0.01
    severe_threshold: float = 0.001
    min_samples: int = 100
    statistical_tests: List[str] = None
    
    def __post_init__(self):
        if self.statistical_tests is None:
            self.statistical_tests = ["ks_test", "wasserstein", "psi"]


class DataDriftDetector:
    """
    データドリフト検出器
    
    複数の統計的手法を使用してデータ分布の変化を検出し、
    モデル再学習の必要性を判定します。
    """
    
    def __init__(self, config: Optional[DriftMonitoringConfig] = None):
        """
        初期化
        
        Args:
            config: ドリフト監視設定
        """
        self.config = config or DriftMonitoringConfig()
        self.drift_history: List[DriftDetectionResult] = []
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_timestamp: Optional[datetime] = None
        
    def set_reference_data(self, data: pd.DataFrame, timestamp: Optional[datetime] = None):
        """
        参照データを設定
        
        Args:
            data: 参照データ
            timestamp: データのタイムスタンプ
        """
        self.reference_data = data.copy()
        self.reference_timestamp = timestamp or datetime.now()
        logger.info(f"参照データ設定完了: {len(data)}行, {len(data.columns)}列")
    
    def detect_drift(self, current_data: pd.DataFrame, 
                    timestamp: Optional[datetime] = None) -> List[DriftDetectionResult]:
        """
        データドリフトを検出
        
        Args:
            current_data: 現在のデータ
            timestamp: データのタイムスタンプ
            
        Returns:
            ドリフト検出結果のリスト
        """
        if self.reference_data is None:
            logger.warning("参照データが設定されていません")
            return []
        
        timestamp = timestamp or datetime.now()
        results = []
        
        # 数値カラムのみを対象
        numeric_columns = current_data.select_dtypes(include=[np.number]).columns
        reference_numeric = self.reference_data.select_dtypes(include=[np.number])
        
        for column in numeric_columns:
            if column not in reference_numeric.columns:
                continue
                
            try:
                # データ準備
                reference_values = reference_numeric[column].dropna()
                current_values = current_data[column].dropna()
                
                if len(reference_values) < self.config.min_samples or \
                   len(current_values) < self.config.min_samples:
                    continue
                
                # 各統計テストを実行
                for test_method in self.config.statistical_tests:
                    result = self._perform_drift_test(
                        reference_values, current_values, column, test_method, timestamp
                    )
                    if result:
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"ドリフト検出エラー (カラム: {column}): {e}")
        
        # 結果を履歴に追加
        self.drift_history.extend(results)
        
        # 履歴サイズ制限
        if len(self.drift_history) > 10000:
            self.drift_history = self.drift_history[-5000:]
        
        logger.info(f"ドリフト検出完了: {len(results)}個の結果")
        return results
    
    def _perform_drift_test(self, reference_values: pd.Series, current_values: pd.Series,
                           column: str, test_method: str, timestamp: datetime) -> Optional[DriftDetectionResult]:
        """統計テストを実行"""
        try:
            if test_method == "ks_test":
                return self._kolmogorov_smirnov_test(
                    reference_values, current_values, column, timestamp
                )
            elif test_method == "wasserstein":
                return self._wasserstein_test(
                    reference_values, current_values, column, timestamp
                )
            elif test_method == "psi":
                return self._population_stability_index(
                    reference_values, current_values, column, timestamp
                )
            else:
                logger.warning(f"未知のテスト方法: {test_method}")
                return None
                
        except Exception as e:
            logger.error(f"統計テストエラー ({test_method}, {column}): {e}")
            return None
    
    def _kolmogorov_smirnov_test(self, reference_values: pd.Series, current_values: pd.Series,
                                column: str, timestamp: datetime) -> DriftDetectionResult:
        """Kolmogorov-Smirnov検定"""
        statistic, p_value = stats.ks_2samp(reference_values, current_values)
        
        # ドリフトタイプの判定
        drift_type = self._classify_drift_by_pvalue(p_value)
        
        # 推奨事項の生成
        recommendation = self._generate_recommendation(drift_type, "KS検定")
        
        return DriftDetectionResult(
            feature_name=column,
            drift_type=drift_type,
            drift_score=statistic,
            p_value=p_value,
            threshold=self.config.severe_threshold,
            test_method="ks_test",
            timestamp=timestamp,
            reference_period=f"{self.config.reference_window_days}日間",
            current_period=f"{self.config.detection_window_days}日間",
            recommendation=recommendation
        )
    
    def _wasserstein_test(self, reference_values: pd.Series, current_values: pd.Series,
                         column: str, timestamp: datetime) -> DriftDetectionResult:
        """Wasserstein距離テスト"""
        distance = wasserstein_distance(reference_values, current_values)
        
        # 正規化された距離を計算
        reference_range = reference_values.max() - reference_values.min()
        normalized_distance = distance / reference_range if reference_range > 0 else 0
        
        # ドリフトタイプの判定（距離ベース）
        if normalized_distance < 0.1:
            drift_type = DriftType.NO_DRIFT
        elif normalized_distance < 0.2:
            drift_type = DriftType.MILD_DRIFT
        elif normalized_distance < 0.4:
            drift_type = DriftType.MODERATE_DRIFT
        else:
            drift_type = DriftType.SEVERE_DRIFT
        
        recommendation = self._generate_recommendation(drift_type, "Wasserstein距離")
        
        return DriftDetectionResult(
            feature_name=column,
            drift_type=drift_type,
            drift_score=normalized_distance,
            p_value=1.0 - normalized_distance,  # 疑似p値
            threshold=0.2,
            test_method="wasserstein",
            timestamp=timestamp,
            reference_period=f"{self.config.reference_window_days}日間",
            current_period=f"{self.config.detection_window_days}日間",
            recommendation=recommendation
        )
    
    def _population_stability_index(self, reference_values: pd.Series, current_values: pd.Series,
                                   column: str, timestamp: datetime) -> DriftDetectionResult:
        """Population Stability Index (PSI)"""
        try:
            # ビンを作成（参照データに基づく）
            bins = np.histogram_bin_edges(reference_values, bins=10)
            
            # 各データセットのヒストグラムを計算
            ref_hist, _ = np.histogram(reference_values, bins=bins)
            cur_hist, _ = np.histogram(current_values, bins=bins)
            
            # 比率に変換（ゼロ除算を避ける）
            ref_pct = (ref_hist + 1e-6) / (len(reference_values) + 1e-5)
            cur_pct = (cur_hist + 1e-6) / (len(current_values) + 1e-5)
            
            # PSIを計算
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            
            # ドリフトタイプの判定（PSI基準）
            if psi < 0.1:
                drift_type = DriftType.NO_DRIFT
            elif psi < 0.2:
                drift_type = DriftType.MILD_DRIFT
            elif psi < 0.25:
                drift_type = DriftType.MODERATE_DRIFT
            else:
                drift_type = DriftType.SEVERE_DRIFT
            
            recommendation = self._generate_recommendation(drift_type, "PSI")
            
            return DriftDetectionResult(
                feature_name=column,
                drift_type=drift_type,
                drift_score=psi,
                p_value=max(0.001, 1.0 - psi),  # 疑似p値
                threshold=0.2,
                test_method="psi",
                timestamp=timestamp,
                reference_period=f"{self.config.reference_window_days}日間",
                current_period=f"{self.config.detection_window_days}日間",
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"PSI計算エラー: {e}")
            return None
    
    def _classify_drift_by_pvalue(self, p_value: float) -> DriftType:
        """p値によるドリフト分類"""
        if p_value >= self.config.mild_threshold:
            return DriftType.NO_DRIFT
        elif p_value >= self.config.moderate_threshold:
            return DriftType.MILD_DRIFT
        elif p_value >= self.config.severe_threshold:
            return DriftType.MODERATE_DRIFT
        else:
            return DriftType.SEVERE_DRIFT
    
    def _generate_recommendation(self, drift_type: DriftType, test_method: str) -> str:
        """推奨事項を生成"""
        recommendations = {
            DriftType.NO_DRIFT: "継続監視",
            DriftType.MILD_DRIFT: "データ品質確認を推奨",
            DriftType.MODERATE_DRIFT: "特徴量エンジニアリング見直しを検討",
            DriftType.SEVERE_DRIFT: "モデル再学習を強く推奨"
        }
        
        base_rec = recommendations.get(drift_type, "詳細調査が必要")
        return f"{base_rec} ({test_method}による検出)"
    
    def get_drift_summary(self, hours: int = 24) -> Dict[str, Any]:
        """ドリフト検出サマリーを取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_results = [r for r in self.drift_history if r.timestamp >= cutoff_time]
        
        if not recent_results:
            return {"status": "no_recent_data", "period_hours": hours}
        
        # 統計情報を計算
        drift_counts = {}
        for drift_type in DriftType:
            drift_counts[drift_type.value] = sum(
                1 for r in recent_results if r.drift_type == drift_type
            )
        
        # 最も深刻なドリフト
        severe_drifts = [r for r in recent_results if r.drift_type == DriftType.SEVERE_DRIFT]
        
        return {
            "period_hours": hours,
            "total_detections": len(recent_results),
            "drift_counts": drift_counts,
            "severe_drift_features": [r.feature_name for r in severe_drifts],
            "requires_attention": len(severe_drifts) > 0,
            "last_detection": recent_results[-1].timestamp.isoformat() if recent_results else None
        }
