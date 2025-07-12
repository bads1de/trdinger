"""
ML信号生成器

LightGBMを用いた3クラス分類モデル（上昇・下落・レンジ）による
価格予測信号を生成します。
"""

import logging
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """
    ML信号生成器
    
    特徴量エンジニアリングサービスが生成した特徴量を使用して、
    未来の価格動向（上昇・下落・レンジ）を予測するモデルを構築・運用します。
    """

    def __init__(self, model_save_path: str = "backend/ml_models/"):
        """
        初期化

        Args:
            model_save_path: モデル保存パス
        """
        self.model_save_path = model_save_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
        # モデル保存ディレクトリを作成
        os.makedirs(model_save_path, exist_ok=True)

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        prediction_horizon: int = 24,
        threshold_up: float = 0.02,
        threshold_down: float = -0.02
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        学習用データを準備

        Args:
            df: 特徴量付きDataFrame
            prediction_horizon: 予測期間（時間）
            threshold_up: 上昇判定閾値
            threshold_down: 下落判定閾値

        Returns:
            特徴量DataFrame、ラベルSeries
        """
        try:
            # ターゲットカラムの存在チェック
            required_target_columns = ['target_up', 'target_down', 'target_range']
            if not all(col in df.columns for col in required_target_columns):
                raise ValueError("Target columns not found in features_df")

            # 未来の価格変化率を計算
            future_returns = (
                df['close'].shift(-prediction_horizon) / df['close'] - 1
            )

            # 3クラス分類のラベルを作成
            labels = pd.Series(index=df.index, dtype=int)
            labels[future_returns >= threshold_up] = 2  # 上昇
            labels[future_returns <= threshold_down] = 0  # 下落
            labels[(future_returns > threshold_down) & (future_returns < threshold_up)] = 1  # レンジ

            # NaNを除去
            valid_mask = ~(future_returns.isna() | labels.isna())
            df_clean = df[valid_mask].copy()
            labels_clean = labels[valid_mask]

            # 特徴量カラムを選択（数値カラムのみ）
            feature_columns = []
            for col in df_clean.columns:
                if col not in ['timestamp', 'open', 'high', 'low', 'close'] and col not in required_target_columns:
                    if df_clean[col].dtype in ['int64', 'float64']:
                        feature_columns.append(col)

            self.feature_columns = feature_columns
            features_df = df_clean[feature_columns].fillna(0)

            logger.info(f"学習データ準備完了: {len(features_df)}サンプル, {len(feature_columns)}特徴量")
            logger.info(f"ラベル分布: 下落={sum(labels_clean==0)}, レンジ={sum(labels_clean==1)}, 上昇={sum(labels_clean==2)}")

            return features_df, labels_clean

        except Exception as e:
            logger.error(f"学習データ準備エラー: {e}")
            raise

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        モデルを学習

        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries
            test_size: テストデータの割合
            random_state: 乱数シード

        Returns:
            学習結果の辞書
        """
        try:
            # 時系列分割（リークを防ぐため）
            split_index = int(len(features) * (1 - test_size))
            X_train = features.iloc[:split_index]
            X_test = features.iloc[split_index:]
            y_train = labels.iloc[:split_index]
            y_test = labels.iloc[split_index:]

            # 特徴量の標準化
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # LightGBMデータセットを作成
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

            # LightGBMパラメータ
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': random_state
            }

            # モデル学習
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )

            # 予測と評価
            y_pred = self.model.predict(X_test_scaled, num_iteration=self.model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1)

            accuracy = accuracy_score(y_test, y_pred_class)
            class_report = classification_report(y_test, y_pred_class, output_dict=True, zero_division=0)

            # 特徴量重要度
            feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importance(importance_type='gain')
            ))

            self.is_trained = True

            result = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'feature_importance': feature_importance,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'best_iteration': self.model.best_iteration
            }

            logger.info(f"モデル学習完了: 精度={accuracy:.4f}")
            return result

        except Exception as e:
            logger.error(f"モデル学習エラー: {e}")
            raise

    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        予測を実行

        Args:
            features: 特徴量DataFrame

        Returns:
            予測確率の辞書 {"up": float, "down": float, "range": float}
        """
        try:
            if not self.is_trained or self.model is None:
                # モデル未学習時は警告レベルでログ出力（エラーレベルから変更）
                logger.warning("モデルが学習されていません。デフォルト値を返します。")
                return {"down": 0.33, "range": 0.34, "up": 0.33}  # 早期リターン

            if self.feature_columns is None:
                # 特徴量カラムが設定されていない場合、利用可能な全カラムを使用
                logger.warning("特徴量カラムが設定されていません。利用可能な全カラムを使用します。")
                features_selected = features.fillna(0)
            else:
                # 特徴量を選択・整形
                available_columns = [col for col in self.feature_columns if col in features.columns]
                if not available_columns:
                    logger.warning("指定された特徴量カラムが見つかりません。利用可能な全カラムを使用します。")
                    features_selected = features.fillna(0)
                else:
                    features_selected = features[available_columns].fillna(0)
            
            # 標準化
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_selected)
            else:
                logger.warning("スケーラーが設定されていません。標準化をスキップします。")
                features_scaled = features_selected.values

            # 予測（モデルタイプに応じて適切なメソッドを使用）
            if hasattr(self.model, 'predict_proba'):
                # 確率予測が可能な場合（RandomForest等）
                predictions = self.model.predict_proba(features_scaled)
            elif hasattr(self.model, 'predict') and hasattr(self.model, 'best_iteration'):
                # LightGBM等の場合
                predictions = self.model.predict(
                    features_scaled,
                    num_iteration=self.model.best_iteration
                )
            else:
                # その他のモデルの場合
                predictions = self.model.predict(features_scaled)

            # 最新の予測結果を取得
            if len(predictions.shape) == 2:
                latest_pred = predictions[-1]  # 最後の行
            else:
                latest_pred = predictions

            # 予測結果を3クラス（down, range, up）の確率に変換
            if len(latest_pred) == 3:
                return {
                    "down": float(latest_pred[0]),
                    "range": float(latest_pred[1]),
                    "up": float(latest_pred[2])
                }
            elif len(latest_pred) == 2:
                # 2クラス分類の場合、rangeを中間値として設定
                return {
                    "down": float(latest_pred[0]),
                    "range": 0.34,
                    "up": float(latest_pred[1])
                }
            else:
                # 予期しない形式の場合、デフォルト値を返す
                logger.warning(f"予期しない予測結果の形式: {latest_pred.shape}")
                return {"down": 0.33, "range": 0.34, "up": 0.33}

        except Exception as e:
            logger.warning(f"予測エラー: {e}")
            return {"down": 0.33, "range": 0.34, "up": 0.33}  # デフォルト値

    def save_model(self, model_name: str = "ml_signal_model") -> str:
        """
        モデルを保存

        Args:
            model_name: モデル名

        Returns:
            保存パス
        """
        try:
            if not self.is_trained:
                raise ValueError("学習済みモデルがありません")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_save_path, f"{model_name}_{timestamp}")

            # モデルとメタデータを保存
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'timestamp': timestamp
            }

            joblib.dump(model_data, f"{model_path}.pkl")
            logger.info(f"モデル保存完了: {model_path}.pkl")

            return f"{model_path}.pkl"

        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            raise

    def load_model(self, model_path: str) -> bool:
        """
        モデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

            model_data = joblib.load(model_path)

            # モデルデータの構造を確認して適切に読み込み
            if isinstance(model_data, dict):
                # 新しい形式（辞書形式）
                if 'model' in model_data:
                    self.model = model_data['model']
                    self.scaler = model_data.get('scaler')
                    self.feature_columns = model_data.get('feature_columns')
                else:
                    # 古い形式の可能性があるため、直接モデルとして扱う
                    self.model = model_data
                    self.scaler = None
                    self.feature_columns = None
            else:
                # 直接モデルオブジェクトの場合
                self.model = model_data
                self.scaler = None
                self.feature_columns = None

            self.is_trained = True

            logger.info(f"モデル読み込み完了: {model_path}")
            return True

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        try:
            if not self.is_trained or self.model is None:
                return {}

            importance = dict(zip(
                self.feature_columns,
                self.model.feature_importance(importance_type='gain')
            ))

            # 重要度順にソート
            sorted_importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            )

            return sorted_importance

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return {}
