"""
ロバストで多角的なMLモデル評価テスト

複数の評価軸、異なるデータ条件、時系列安定性、
実取引シミュレーションを通じて包括的に精度を検証します。
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import RobustScaler, StandardScaler
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer
from app.services.ml.models.ensemble_models import EnsembleModelManager

logger = logging.getLogger(__name__)


class RobustMultifacetedEvaluation:
    """ロバストで多角的なMLモデル評価クラス"""

    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble_manager = EnsembleModelManager()
        self.evaluation_results = {}

    def create_diverse_test_scenarios(self):
        """多様なテストシナリオを作成"""
        scenarios = {}
        
        # シナリオ1: 標準的な市場条件
        scenarios['normal_market'] = self._create_market_data(
            n_samples=1500, volatility=0.015, trend_strength=0.02, noise_level=0.01
        )
        
        # シナリオ2: 高ボラティリティ市場
        scenarios['high_volatility'] = self._create_market_data(
            n_samples=1500, volatility=0.035, trend_strength=0.01, noise_level=0.02
        )
        
        # シナリオ3: 強いトレンド市場
        scenarios['strong_trend'] = self._create_market_data(
            n_samples=1500, volatility=0.012, trend_strength=0.05, noise_level=0.008
        )
        
        # シナリオ4: レンジ相場
        scenarios['range_bound'] = self._create_market_data(
            n_samples=1500, volatility=0.008, trend_strength=0.001, noise_level=0.015
        )
        
        # シナリオ5: 極端な市場条件
        scenarios['extreme_market'] = self._create_market_data(
            n_samples=1500, volatility=0.05, trend_strength=0.03, noise_level=0.025
        )
        
        return scenarios

    def _create_market_data(self, n_samples, volatility, trend_strength, noise_level):
        """指定されたパラメータで市場データを作成"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
        base_price = 50000
        
        # トレンドコンポーネント
        trend = np.cumsum(np.random.normal(0, trend_strength, n_samples))
        
        # ボラティリティクラスタリング
        vol_series = np.zeros(n_samples)
        vol_series[0] = volatility
        for i in range(1, n_samples):
            vol_series[i] = 0.9 * vol_series[i-1] + 0.1 * volatility + np.random.normal(0, volatility * 0.1)
        
        # 価格生成
        returns = trend / n_samples + np.random.normal(0, vol_series) + np.random.normal(0, noise_level)
        prices = base_price * np.cumprod(1 + returns)
        
        # OHLCV データ
        ohlcv_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.003, n_samples))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_samples))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 0.8, n_samples),
        }, index=dates)
        
        # 基本技術指標
        ohlcv_data['SMA_10'] = ohlcv_data['Close'].rolling(10).mean()
        ohlcv_data['SMA_20'] = ohlcv_data['Close'].rolling(20).mean()
        ohlcv_data['RSI'] = self._calculate_rsi(ohlcv_data['Close'])
        ohlcv_data['MACD'] = ohlcv_data['Close'].ewm(span=12).mean() - ohlcv_data['Close'].ewm(span=26).mean()
        
        # 外部データ
        fr_data = pd.DataFrame({
            'timestamp': dates[::8],
            'funding_rate': np.random.normal(0.0001, 0.0003, len(dates[::8]))
        })
        
        oi_data = pd.DataFrame({
            'timestamp': dates,
            'open_interest': np.random.lognormal(15, 0.3, n_samples)
        })
        
        # ターゲット生成
        future_returns = ohlcv_data['Close'].pct_change(12).shift(-12)
        rolling_vol = ohlcv_data['Close'].pct_change().rolling(24).std()
        dynamic_threshold = rolling_vol * 0.8
        
        y = pd.Series(1, index=dates)  # Hold
        y[future_returns > dynamic_threshold] = 2   # Up
        y[future_returns < -dynamic_threshold] = 0  # Down
        
        # データクリーニング
        valid_mask = ohlcv_data.notna().all(axis=1) & y.notna() & rolling_vol.notna()
        ohlcv_data = ohlcv_data[valid_mask]
        y = y[valid_mask]
        
        return {
            'ohlcv': ohlcv_data,
            'funding_rate': fr_data,
            'open_interest': oi_data,
            'target': y,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'noise_level': noise_level
        }

    def _calculate_rsi(self, prices, window=14):
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def evaluate_time_series_stability(self, X, y, model, n_splits=5):
        """時系列安定性評価"""
        logger.info("⏰ 時系列安定性評価中...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metrics = {
            'accuracy': [],
            'balanced_accuracy': [],
            'f1_weighted': [],
            'precision_weighted': [],
            'recall_weighted': []
        }
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            # スケーリング
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_fold),
                columns=X_train_fold.columns,
                index=X_train_fold.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_fold),
                columns=X_test_fold.columns,
                index=X_test_fold.index
            )
            
            # モデル学習・予測
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_scaled, y_train_fold)
                y_pred = model.predict(X_test_scaled)
            
            # 評価指標計算
            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_test_fold, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test_fold, y_pred),
                'f1_weighted': f1_score(y_test_fold, y_pred, average='weighted'),
                'precision_weighted': precision_score(y_test_fold, y_pred, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_test_fold, y_pred, average='weighted', zero_division=0),
                'train_period': f"{X_train_fold.index[0]} to {X_train_fold.index[-1]}",
                'test_period': f"{X_test_fold.index[0]} to {X_test_fold.index[-1]}"
            }
            
            fold_results.append(fold_result)
            
            for metric in metrics:
                metrics[metric].append(fold_result[metric])
        
        # 統計サマリー
        stability_summary = {}
        for metric, values in metrics.items():
            stability_summary[f'{metric}_mean'] = np.mean(values)
            stability_summary[f'{metric}_std'] = np.std(values)
            stability_summary[f'{metric}_min'] = np.min(values)
            stability_summary[f'{metric}_max'] = np.max(values)
            stability_summary[f'{metric}_cv'] = np.std(values) / np.mean(values) if np.mean(values) > 0 else np.inf
        
        return stability_summary, fold_results

    def evaluate_class_imbalance_robustness(self, X, y, model):
        """クラス不均衡に対するロバストネス評価"""
        logger.info("⚖️ クラス不均衡ロバストネス評価中...")
        
        # 元のクラス分布
        original_distribution = y.value_counts(normalize=True).to_dict()
        
        # 時系列分割
        split_point = int(len(X) * 0.7)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # モデル学習・予測
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # クラス別詳細評価
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # 不均衡対応指標
        imbalance_metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
            'macro_precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        }
        
        # ROC-AUC（多クラス対応）
        if y_proba is not None and len(np.unique(y_test)) > 2:
            try:
                imbalance_metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                imbalance_metrics['roc_auc_ovo'] = roc_auc_score(y_test, y_proba, multi_class='ovo', average='weighted')
            except:
                pass
        
        return {
            'original_distribution': original_distribution,
            'test_distribution': y_test.value_counts(normalize=True).to_dict(),
            'pred_distribution': pd.Series(y_pred).value_counts(normalize=True).to_dict(),
            'imbalance_metrics': imbalance_metrics,
            'class_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }

    def evaluate_feature_importance_stability(self, X, y, model, n_iterations=5):
        """特徴量重要度の安定性評価"""
        logger.info("🔍 特徴量重要度安定性評価中...")
        
        importance_results = []
        
        for i in range(n_iterations):
            # ブートストラップサンプリング
            sample_indices = np.random.choice(len(X), size=int(len(X) * 0.8), replace=True)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            
            # 時系列分割
            split_point = int(len(X_sample) * 0.7)
            X_train = X_sample.iloc[:split_point]
            y_train = y_sample.iloc[:split_point]
            
            # スケーリング
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            # モデル学習
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_scaled, y_train)
            
            # 特徴量重要度取得
            if hasattr(model, 'feature_importances_'):
                importance = pd.Series(model.feature_importances_, index=X_train_scaled.columns)
                importance_results.append(importance)
        
        if importance_results:
            # 重要度の統計
            importance_df = pd.DataFrame(importance_results)
            importance_stats = {
                'mean': importance_df.mean(),
                'std': importance_df.std(),
                'cv': importance_df.std() / importance_df.mean(),
                'top_10_stable': importance_df.mean().nlargest(10).index.tolist()
            }
            
            return importance_stats
        
        return None

    def simulate_trading_performance(self, X, y, model, initial_capital=100000):
        """取引パフォーマンスシミュレーション"""
        logger.info("💰 取引パフォーマンスシミュレーション中...")
        
        # 時系列分割
        split_point = int(len(X) * 0.7)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # モデル学習・予測
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # 取引シミュレーション
        capital = initial_capital
        positions = []
        returns = []
        
        # 価格データを取得（テスト期間）
        test_prices = X_test.index.to_series().apply(lambda x: 50000 + np.random.normal(0, 1000))  # 簡易価格
        
        for i in range(len(y_pred)):
            prediction = y_pred[i]
            actual = y_test.iloc[i]
            
            # 取引戦略
            if prediction == 2:  # Up予測
                position = 'long'
                if actual == 2:  # 正解
                    return_rate = 0.02  # 2%利益
                elif actual == 0:  # 不正解
                    return_rate = -0.02  # 2%損失
                else:  # Hold
                    return_rate = 0.005  # 0.5%利益
            elif prediction == 0:  # Down予測
                position = 'short'
                if actual == 0:  # 正解
                    return_rate = 0.02  # 2%利益
                elif actual == 2:  # 不正解
                    return_rate = -0.02  # 2%損失
                else:  # Hold
                    return_rate = 0.005  # 0.5%利益
            else:  # Hold予測
                position = 'hold'
                return_rate = 0  # 手数料考慮で0
            
            capital *= (1 + return_rate)
            positions.append(position)
            returns.append(return_rate)
        
        # パフォーマンス指標
        total_return = (capital - initial_capital) / initial_capital
        returns_array = np.array(returns)
        
        trading_metrics = {
            'total_return': total_return,
            'final_capital': capital,
            'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns_array),
            'win_rate': np.sum(np.array(returns) > 0) / len(returns),
            'avg_return_per_trade': np.mean(returns_array),
            'volatility': np.std(returns_array) * np.sqrt(252),
            'total_trades': len(returns),
            'long_trades': positions.count('long'),
            'short_trades': positions.count('short'),
            'hold_trades': positions.count('hold')
        }
        
        return trading_metrics

    def _calculate_max_drawdown(self, returns):
        """最大ドローダウン計算"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def run_comprehensive_evaluation(self):
        """包括的評価を実行"""
        logger.info("=" * 80)
        logger.info("🔬 ロバストで多角的なMLモデル評価")
        logger.info("=" * 80)
        
        # 多様なテストシナリオ作成
        scenarios = self.create_diverse_test_scenarios()
        
        # 最適なモデルを作成
        best_model = self.ensemble_manager.create_optimized_ensemble(
            scenarios['normal_market']['ohlcv'], 
            scenarios['normal_market']['target'],
            optimize=False
        )
        
        evaluation_results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            logger.info(f"\n📊 シナリオ評価: {scenario_name}")
            
            # 高度な特徴量生成
            features = self.feature_engineer.create_advanced_features(
                scenario_data['ohlcv'],
                scenario_data['funding_rate'],
                scenario_data['open_interest']
            )
            features = self.feature_engineer.clean_features(features)
            
            # インデックス合わせ
            common_index = features.index.intersection(scenario_data['target'].index)
            X = features.loc[common_index]
            y = scenario_data['target'].loc[common_index]
            
            if len(X) < 100:
                logger.warning(f"  データ不足: {len(X)}サンプル")
                continue
            
            # 特徴量選択
            from sklearn.ensemble import RandomForestClassifier
            temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
            split_point = int(len(X) * 0.7)
            X_temp_train = X.iloc[:split_point]
            y_temp_train = y.iloc[:split_point]
            
            scaler_temp = RobustScaler()
            X_temp_scaled = pd.DataFrame(
                scaler_temp.fit_transform(X_temp_train),
                columns=X_temp_train.columns,
                index=X_temp_train.index
            )
            
            temp_model.fit(X_temp_scaled, y_temp_train)
            feature_importance = pd.Series(
                temp_model.feature_importances_,
                index=X_temp_scaled.columns
            ).sort_values(ascending=False)
            
            top_features = feature_importance.head(30).index
            X_selected = X[top_features]
            
            scenario_results = {}
            
            # 1. 時系列安定性評価
            stability_summary, fold_results = self.evaluate_time_series_stability(
                X_selected, y, best_model
            )
            scenario_results['time_series_stability'] = {
                'summary': stability_summary,
                'fold_details': fold_results
            }
            
            # 2. クラス不均衡ロバストネス
            imbalance_results = self.evaluate_class_imbalance_robustness(
                X_selected, y, best_model
            )
            scenario_results['class_imbalance_robustness'] = imbalance_results
            
            # 3. 特徴量重要度安定性
            importance_stability = self.evaluate_feature_importance_stability(
                X_selected, y, best_model
            )
            scenario_results['feature_importance_stability'] = importance_stability
            
            # 4. 取引パフォーマンス
            trading_performance = self.simulate_trading_performance(
                X_selected, y, best_model
            )
            scenario_results['trading_performance'] = trading_performance
            
            # シナリオ情報
            scenario_results['scenario_info'] = {
                'volatility': scenario_data['volatility'],
                'trend_strength': scenario_data['trend_strength'],
                'noise_level': scenario_data['noise_level'],
                'samples': len(X),
                'features': len(top_features),
                'class_distribution': y.value_counts(normalize=True).to_dict()
            }
            
            evaluation_results[scenario_name] = scenario_results
            
            # 結果サマリー表示
            logger.info(f"  安定性 (CV): {stability_summary['balanced_accuracy_cv']:.4f}")
            logger.info(f"  バランス精度: {stability_summary['balanced_accuracy_mean']:.4f} ± {stability_summary['balanced_accuracy_std']:.4f}")
            logger.info(f"  取引リターン: {trading_performance['total_return']:.2%}")
            logger.info(f"  シャープレシオ: {trading_performance['sharpe_ratio']:.4f}")
        
        self.evaluation_results = evaluation_results
        return evaluation_results


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    
    # 包括的評価実行
    evaluator = RobustMultifacetedEvaluation()
    results = evaluator.run_comprehensive_evaluation()
    
    logger.info("\n🎉 ロバストで多角的な評価完了")
