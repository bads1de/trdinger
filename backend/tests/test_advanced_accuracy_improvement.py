"""
高度な精度改善効果テスト

新しい高度な特徴量エンジニアリングとアンサンブル学習により、
40.55%から60%以上への精度向上を検証します。
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer

logger = logging.getLogger(__name__)


class AdvancedAccuracyImprovementTest:
    """高度な精度改善効果テストクラス"""

    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        # EnsembleTrainerの設定
        ensemble_config = {
            "method": "stacking",
            "stacking_params": {
                "base_models": ["lightgbm", "random_forest", "xgboost"],
                "meta_model": "lightgbm",
                "cv_folds": 3
            }
        }
        self.ensemble_trainer = EnsembleTrainer(ensemble_config=ensemble_config)

    def create_enhanced_dataset(self, n_samples=1000):
        """拡張されたデータセットを作成"""
        np.random.seed(42)
        
        # 時系列インデックス
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
        
        # より現実的な価格データ（複数のトレンドパターン）
        base_price = 50000
        
        # 複数の市場サイクルを模擬
        trend_component = np.sin(np.arange(n_samples) * 2 * np.pi / 168) * 0.05  # 週次サイクル
        volatility_component = np.sin(np.arange(n_samples) * 2 * np.pi / 24) * 0.02  # 日次サイクル
        random_walk = np.cumsum(np.random.normal(0, 0.01, n_samples))
        
        price_pattern = trend_component + volatility_component + random_walk
        prices = base_price * np.cumprod(1 + price_pattern)
        
        # OHLCV データ
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.003, n_samples))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_samples))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 0.8, n_samples),
        }, index=dates)
        
        # 基本的な技術指標（TALibが使用できない場合のフォールバック）
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        
        # ファンディングレートと建玉残高
        fr_data = pd.DataFrame({
            'timestamp': dates[::8],  # 8時間間隔
            'funding_rate': np.random.normal(0.0001, 0.0003, len(dates[::8]))
        })
        
        oi_data = pd.DataFrame({
            'timestamp': dates,
            'open_interest': np.random.lognormal(15, 0.3, n_samples)
        })
        
        # ターゲット生成（より予測可能なパターンを含む）
        future_returns = data['Close'].pct_change(12).shift(-12)
        
        # 動的閾値
        rolling_vol = data['Close'].pct_change().rolling(24).std()
        dynamic_threshold = rolling_vol * 0.8
        
        # 3クラス分類
        y = pd.Series(1, index=dates)  # Hold
        y[future_returns > dynamic_threshold] = 2   # Up
        y[future_returns < -dynamic_threshold] = 0  # Down
        
        # データクリーニング
        valid_mask = data.notna().all(axis=1) & y.notna() & rolling_vol.notna()
        data = data[valid_mask]
        y = y[valid_mask]
        
        return data, fr_data, oi_data, y

    def _calculate_rsi(self, prices, window=14):
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def test_baseline_performance(self, X, y):
        """ベースライン性能（改善前）"""
        logger.info("🔴 ベースライン性能テスト（基本特徴量 + RandomForest）")
        
        # 時系列分割
        split_point = int(len(X) * 0.7)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        # 基本的なスケーリング
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
        
        # RandomForest
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        results = {
            'method': 'ベースライン（基本特徴量 + RandomForest）',
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'features': X.shape[1]
        }
        
        logger.info(f"  精度: {results['accuracy']:.4f}")
        logger.info(f"  バランス精度: {results['balanced_accuracy']:.4f}")
        logger.info(f"  F1スコア: {results['f1_score']:.4f}")
        logger.info(f"  特徴量数: {results['features']}")
        
        return results

    def test_advanced_features_performance(self, ohlcv_data, fr_data, oi_data, y):
        """高度な特徴量での性能テスト"""
        logger.info("🟡 高度な特徴量性能テスト（高度特徴量 + RandomForest）")
        
        # 高度な特徴量生成
        advanced_features = self.feature_engineer.create_advanced_features(
            ohlcv_data, fr_data, oi_data
        )
        
        # データクリーニング
        advanced_features = self.feature_engineer.clean_features(advanced_features)
        
        # インデックスを合わせる
        common_index = advanced_features.index.intersection(y.index)
        X = advanced_features.loc[common_index]
        y_aligned = y.loc[common_index]
        
        # 時系列分割
        split_point = int(len(X) * 0.7)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y_aligned.iloc[:split_point]
        y_test = y_aligned.iloc[split_point:]
        
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
        
        # 特徴量選択（重要度ベース）
        from sklearn.ensemble import RandomForestClassifier
        temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
        temp_model.fit(X_train_scaled, y_train)
        
        feature_importance = pd.Series(
            temp_model.feature_importances_,
            index=X_train_scaled.columns
        ).sort_values(ascending=False)
        
        # 上位特徴量を選択
        top_features = feature_importance.head(min(30, len(feature_importance))).index
        X_train_selected = X_train_scaled[top_features]
        X_test_selected = X_test_scaled[top_features]
        
        # RandomForest（改良版）
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_selected, y_train)
        
        y_pred = model.predict(X_test_selected)
        
        results = {
            'method': '高度特徴量 + 改良RandomForest',
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'features': len(top_features),
            'total_features_generated': X.shape[1]
        }
        
        logger.info(f"  精度: {results['accuracy']:.4f}")
        logger.info(f"  バランス精度: {results['balanced_accuracy']:.4f}")
        logger.info(f"  F1スコア: {results['f1_score']:.4f}")
        logger.info(f"  選択特徴量数: {results['features']}")
        logger.info(f"  生成特徴量総数: {results['total_features_generated']}")
        
        return results, X_train_selected, X_test_selected, y_train, y_test

    def test_ensemble_performance(self, X_train, X_test, y_train, y_test):
        """アンサンブル学習での性能テスト"""
        logger.info("🟢 アンサンブル学習性能テスト（高度特徴量 + アンサンブル）")
        
        # アンサンブルモデルの学習・評価
        training_result = self.ensemble_trainer._train_model_impl(
            X_train, X_test, y_train, y_test
        )
        
        # 結果を整形（EnsembleTrainerの結果をensemble_resultsの形式に変換）
        ensemble_results = {
            "stacking_ensemble": {
                "accuracy": training_result.get("accuracy", 0),
                "balanced_accuracy": training_result.get("balanced_accuracy", 0),
                "f1_score": training_result.get("f1_score", 0)
            }
        }
        
        # 最高性能モデルを特定
        best_method = "stacking_ensemble"
        best_model = ensemble_results["stacking_ensemble"]
        
        logger.info(f"🏆 最高性能モデル: {best_method}")
        logger.info(f"  精度: {best_model['accuracy']:.4f}")
        logger.info(f"  バランス精度: {best_model['balanced_accuracy']:.4f}")
        logger.info(f"  F1スコア: {best_model['f1_score']:.4f}")
        
        return ensemble_results, best_method, best_model

    def run_comprehensive_accuracy_test(self):
        """包括的な精度改善テストを実行"""
        logger.info("=" * 80)
        logger.info("🚀 高度な精度改善効果の包括的検証")
        logger.info("=" * 80)
        
        # 拡張データセット作成
        ohlcv_data, fr_data, oi_data, y = self.create_enhanced_dataset(n_samples=1200)
        
        logger.info(f"データセット: {len(ohlcv_data)}サンプル")
        logger.info(f"ラベル分布: {y.value_counts().to_dict()}")
        
        # 1. ベースライン性能
        baseline_results = self.test_baseline_performance(ohlcv_data, y)
        
        # 2. 高度な特徴量での性能
        advanced_results, X_train, X_test, y_train, y_test = self.test_advanced_features_performance(
            ohlcv_data, fr_data, oi_data, y
        )
        
        # 3. アンサンブル学習での性能
        ensemble_results, best_method, best_model = self.test_ensemble_performance(
            X_train, X_test, y_train, y_test
        )
        
        # 結果分析
        self._analyze_comprehensive_results(baseline_results, advanced_results, best_model, ensemble_results)
        
        return baseline_results, advanced_results, ensemble_results

    def _analyze_comprehensive_results(self, baseline, advanced, best_ensemble, all_ensemble):
        """包括的結果分析"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 包括的精度改善効果分析")
        logger.info("=" * 80)
        
        # 段階的改善効果
        baseline_acc = baseline['balanced_accuracy']
        advanced_acc = advanced['balanced_accuracy']
        ensemble_acc = best_ensemble['balanced_accuracy']
        
        # 改善率計算
        advanced_improvement = (advanced_acc - baseline_acc) / baseline_acc * 100
        ensemble_improvement = (ensemble_acc - baseline_acc) / baseline_acc * 100
        final_improvement = (ensemble_acc - advanced_acc) / advanced_acc * 100
        
        logger.info("🎯 段階的改善効果:")
        logger.info(f"  ベースライン: {baseline_acc:.4f}")
        logger.info(f"  高度特徴量: {advanced_acc:.4f} ({advanced_improvement:+.1f}%)")
        logger.info(f"  アンサンブル: {ensemble_acc:.4f} ({ensemble_improvement:+.1f}%)")
        
        logger.info(f"\n📈 改善段階:")
        logger.info(f"  特徴量改善効果: {advanced_improvement:+.1f}%")
        logger.info(f"  アンサンブル効果: {final_improvement:+.1f}%")
        logger.info(f"  総合改善効果: {ensemble_improvement:+.1f}%")
        
        # 特徴量効率性
        baseline_efficiency = baseline['accuracy'] / baseline['features']
        advanced_efficiency = advanced['accuracy'] / advanced['features']
        
        logger.info(f"\n🔧 特徴量効率性:")
        logger.info(f"  ベースライン: {baseline_efficiency:.6f} ({baseline['features']}特徴量)")
        logger.info(f"  高度特徴量: {advanced_efficiency:.6f} ({advanced['features']}特徴量)")
        
        # 全アンサンブル結果
        logger.info(f"\n🤖 アンサンブルモデル比較:")
        for method, scores in all_ensemble.items():
            logger.info(f"  {method}: {scores['balanced_accuracy']:.4f}")
        logger.info("  注: EnsembleTrainerはスタッキングアンサンブルのみを実装しています")
        
        # 目標達成度
        target_accuracy = 0.55  # 55%目標
        current_accuracy = ensemble_acc
        
        logger.info(f"\n🎯 目標達成度:")
        logger.info(f"  目標精度: {target_accuracy:.1%}")
        logger.info(f"  現在精度: {current_accuracy:.1%}")
        logger.info(f"  達成率: {current_accuracy/target_accuracy:.1%}")
        
        if current_accuracy >= target_accuracy:
            logger.info("  🎉 目標精度を達成！")
        elif current_accuracy >= target_accuracy * 0.9:
            logger.info("  ✅ 目標に近い精度を達成")
        else:
            logger.info("  ⚠️ さらなる改善が必要")
        
        # 次のステップ提案
        logger.info(f"\n🚀 次のステップ提案:")
        if current_accuracy < target_accuracy:
            logger.info("  1. 外部データ統合（マクロ経済指標、センチメント）")
            logger.info("  2. ディープラーニングモデル（LSTM、Transformer）")
            logger.info("  3. ハイパーパラメータ最適化（Optuna）")
            logger.info("  4. より長期間のデータでの学習")
        else:
            logger.info("  1. 本番環境での実証実験")
            logger.info("  2. リアルタイム予測システムの構築")
            logger.info("  3. リスク管理システムの統合")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    
    # 包括的精度改善テスト実行
    test = AdvancedAccuracyImprovementTest()
    baseline, advanced, ensemble = test.run_comprehensive_accuracy_test()
    
    logger.info("\n🎉 高度な精度改善効果検証完了")
