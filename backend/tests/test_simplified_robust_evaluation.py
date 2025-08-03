"""
簡略化されたロバスト評価テスト

複数のシナリオでモデルの堅牢性を検証し、
最終レポート用のデータを収集します。
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import RobustScaler
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)


class SimplifiedRobustEvaluation:
    """簡略化されたロバスト評価クラス"""

    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.evaluation_results = {}

    def create_test_scenarios(self):
        """テストシナリオを作成"""
        scenarios = {}
        
        # シナリオ1: 標準市場
        scenarios['normal'] = self._create_market_data(1200, 0.015, 0.02, 0.01)
        
        # シナリオ2: 高ボラティリティ
        scenarios['high_vol'] = self._create_market_data(1200, 0.035, 0.01, 0.02)
        
        # シナリオ3: 強トレンド
        scenarios['strong_trend'] = self._create_market_data(1200, 0.012, 0.05, 0.008)
        
        return scenarios

    def _create_market_data(self, n_samples, volatility, trend_strength, noise_level):
        """市場データを作成"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
        base_price = 50000
        
        # 価格生成
        trend = np.cumsum(np.random.normal(0, trend_strength, n_samples))
        vol_series = np.full(n_samples, volatility)
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
            'params': {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'noise_level': noise_level
            }
        }

    def _calculate_rsi(self, prices, window=14):
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def evaluate_model_performance(self, X, y, model_name="Model"):
        """モデル性能を評価"""
        logger.info(f"📊 {model_name}の性能評価中...")
        
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
        
        # 複数モデルでテスト
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                class_weight='balanced', random_state=42, max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # 評価指標
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': X.shape[1]
                }
                
                logger.info(f"  {name}: 精度={results[name]['accuracy']:.4f}, "
                           f"バランス精度={results[name]['balanced_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"{name}でエラー: {e}")
                continue
        
        return results

    def evaluate_time_series_stability(self, X, y, n_splits=3):
        """時系列安定性評価"""
        logger.info("⏰ 時系列安定性評価中...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
        )
        
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
            
            # 評価
            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_test_fold, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test_fold, y_pred),
                'f1_score': f1_score(y_test_fold, y_pred, average='weighted')
            }
            
            fold_results.append(fold_result)
        
        # 統計サマリー
        metrics = ['accuracy', 'balanced_accuracy', 'f1_score']
        stability_summary = {}
        
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            stability_summary[f'{metric}_mean'] = np.mean(values)
            stability_summary[f'{metric}_std'] = np.std(values)
            stability_summary[f'{metric}_cv'] = np.std(values) / np.mean(values) if np.mean(values) > 0 else np.inf
        
        return stability_summary, fold_results

    def simulate_trading_performance(self, X, y):
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
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # 取引シミュレーション
        initial_capital = 100000
        capital = initial_capital
        returns = []
        
        for i in range(len(y_pred)):
            prediction = y_pred[i]
            actual = y_test.iloc[i]
            
            # 取引戦略
            if prediction == 2:  # Up予測
                if actual == 2:  # 正解
                    return_rate = 0.02
                elif actual == 0:  # 不正解
                    return_rate = -0.02
                else:  # Hold
                    return_rate = 0.005
            elif prediction == 0:  # Down予測
                if actual == 0:  # 正解
                    return_rate = 0.02
                elif actual == 2:  # 不正解
                    return_rate = -0.02
                else:  # Hold
                    return_rate = 0.005
            else:  # Hold予測
                return_rate = 0
            
            capital *= (1 + return_rate)
            returns.append(return_rate)
        
        # パフォーマンス指標
        total_return = (capital - initial_capital) / initial_capital
        returns_array = np.array(returns)
        
        trading_metrics = {
            'total_return': total_return,
            'final_capital': capital,
            'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
            'win_rate': np.sum(np.array(returns) > 0) / len(returns),
            'avg_return_per_trade': np.mean(returns_array),
            'total_trades': len(returns)
        }
        
        return trading_metrics

    def run_comprehensive_evaluation(self):
        """包括的評価を実行"""
        logger.info("=" * 80)
        logger.info("🔬 簡略化ロバスト評価")
        logger.info("=" * 80)
        
        # テストシナリオ作成
        scenarios = self.create_test_scenarios()
        
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
            
            # 1. モデル性能評価
            model_results = self.evaluate_model_performance(X_selected, y, scenario_name)
            scenario_results['model_performance'] = model_results
            
            # 2. 時系列安定性評価
            stability_summary, fold_results = self.evaluate_time_series_stability(X_selected, y)
            scenario_results['time_series_stability'] = {
                'summary': stability_summary,
                'fold_details': fold_results
            }
            
            # 3. 取引パフォーマンス
            trading_performance = self.simulate_trading_performance(X_selected, y)
            scenario_results['trading_performance'] = trading_performance
            
            # シナリオ情報
            scenario_results['scenario_info'] = {
                'params': scenario_data['params'],
                'samples': len(X),
                'features': len(top_features),
                'class_distribution': y.value_counts(normalize=True).to_dict()
            }
            
            evaluation_results[scenario_name] = scenario_results
            
            # 結果サマリー表示
            best_model = max(model_results.keys(), key=lambda k: model_results[k]['balanced_accuracy'])
            logger.info(f"  最高性能モデル: {best_model}")
            logger.info(f"  バランス精度: {model_results[best_model]['balanced_accuracy']:.4f}")
            logger.info(f"  安定性 (CV): {stability_summary['balanced_accuracy_cv']:.4f}")
            logger.info(f"  取引リターン: {trading_performance['total_return']:.2%}")
        
        self.evaluation_results = evaluation_results
        return evaluation_results


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    
    # 簡略化評価実行
    evaluator = SimplifiedRobustEvaluation()
    results = evaluator.run_comprehensive_evaluation()
    
    logger.info("\n🎉 簡略化ロバスト評価完了")
