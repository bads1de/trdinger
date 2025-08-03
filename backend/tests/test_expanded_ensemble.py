"""
拡張アンサンブル学習テスト

11個のアルゴリズムを使用した拡張アンサンブルの性能を検証します。
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer
from app.services.ml.models.ensemble_models import EnsembleModelManager

logger = logging.getLogger(__name__)


class ExpandedEnsembleTest:
    """拡張アンサンブル学習テストクラス"""

    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble_manager = EnsembleModelManager()

    def create_test_dataset(self, n_samples=1000):
        """テストデータセットを作成"""
        np.random.seed(42)
        
        # 時系列インデックス
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
        
        # より複雑な価格パターン
        base_price = 50000
        
        # 複数の周期的パターン
        daily_cycle = np.sin(np.arange(n_samples) * 2 * np.pi / 24) * 0.01
        weekly_cycle = np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 7)) * 0.02
        monthly_trend = np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 30)) * 0.03
        
        # ランダムウォーク + 周期的パターン
        random_walk = np.cumsum(np.random.normal(0, 0.01, n_samples))
        price_pattern = daily_cycle + weekly_cycle + monthly_trend + random_walk
        
        # 価格生成
        prices = base_price * np.cumprod(1 + price_pattern)
        
        # OHLCV データ
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.003, n_samples))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_samples))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 0.8, n_samples),
        }, index=dates)
        
        # 基本技術指標
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        
        # 外部データ
        fr_data = pd.DataFrame({
            'timestamp': dates[::8],
            'funding_rate': np.random.normal(0.0001, 0.0003, len(dates[::8]))
        })
        
        oi_data = pd.DataFrame({
            'timestamp': dates,
            'open_interest': np.random.lognormal(15, 0.3, n_samples)
        })
        
        # ターゲット生成（より予測可能なパターン）
        future_returns = data['Close'].pct_change(12).shift(-12)
        
        # 動的閾値
        rolling_vol = data['Close'].pct_change().rolling(24).std()
        dynamic_threshold = rolling_vol * 0.7
        
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

    def test_expanded_ensemble_performance(self):
        """拡張アンサンブルの性能テスト"""
        logger.info("=" * 80)
        logger.info("🚀 拡張アンサンブル学習性能テスト（11アルゴリズム）")
        logger.info("=" * 80)
        
        # データセット作成
        ohlcv_data, fr_data, oi_data, y = self.create_test_dataset(n_samples=1200)
        
        logger.info(f"データセット: {len(ohlcv_data)}サンプル")
        logger.info(f"ラベル分布: {y.value_counts().to_dict()}")
        
        # 高度な特徴量生成
        features = self.feature_engineer.create_advanced_features(
            ohlcv_data, fr_data, oi_data
        )
        features = self.feature_engineer.clean_features(features)
        
        # インデックス合わせ
        common_index = features.index.intersection(y.index)
        X = features.loc[common_index]
        y_aligned = y.loc[common_index]
        
        logger.info(f"特徴量: {X.shape[1]}個")
        
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
        
        # 特徴量選択
        from sklearn.ensemble import RandomForestClassifier
        temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
        temp_model.fit(X_train_scaled, y_train)
        
        feature_importance = pd.Series(
            temp_model.feature_importances_,
            index=X_train_scaled.columns
        ).sort_values(ascending=False)
        
        # 上位特徴量を選択
        top_features = feature_importance.head(25).index
        X_train_selected = X_train_scaled[top_features]
        X_test_selected = X_test_scaled[top_features]
        
        logger.info(f"選択特徴量: {len(top_features)}個")
        
        # 拡張アンサンブル学習・評価
        ensemble_results = self.ensemble_manager.train_and_evaluate_models(
            X_train_selected, X_test_selected, y_train, y_test
        )
        
        # 結果分析
        self._analyze_expanded_results(ensemble_results)
        
        return ensemble_results

    def _analyze_expanded_results(self, results):
        """拡張アンサンブル結果分析"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 拡張アンサンブル結果分析（11アルゴリズム）")
        logger.info("=" * 80)
        
        # 結果をソート
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['balanced_accuracy'],
            reverse=True
        )
        
        logger.info("🏆 アルゴリズム性能ランキング:")
        for i, (method, scores) in enumerate(sorted_results, 1):
            logger.info(f"  {i:2d}. {method:20s}: "
                       f"精度={scores['accuracy']:.4f}, "
                       f"バランス精度={scores['balanced_accuracy']:.4f}, "
                       f"F1={scores['f1_score']:.4f}")
        
        # 最高性能モデル
        best_method, best_scores = sorted_results[0]
        logger.info(f"\n🥇 最高性能モデル: {best_method}")
        logger.info(f"  精度: {best_scores['accuracy']:.4f}")
        logger.info(f"  バランス精度: {best_scores['balanced_accuracy']:.4f}")
        logger.info(f"  F1スコア: {best_scores['f1_score']:.4f}")
        
        # アルゴリズムカテゴリ別分析
        tree_based = ['rf', 'extra_trees', 'xgb', 'lgb', 'gb', 'ada']
        linear_based = ['lr', 'ridge', 'svm']
        other_based = ['nb', 'knn']
        ensemble_based = ['voting_ensemble', 'stacking_ensemble']
        
        logger.info(f"\n📊 カテゴリ別性能分析:")
        
        # ツリー系
        tree_results = {k: v for k, v in results.items() if k in tree_based}
        if tree_results:
            avg_tree_acc = np.mean([v['balanced_accuracy'] for v in tree_results.values()])
            logger.info(f"  🌳 ツリー系平均: {avg_tree_acc:.4f}")
        
        # 線形系
        linear_results = {k: v for k, v in results.items() if k in linear_based}
        if linear_results:
            avg_linear_acc = np.mean([v['balanced_accuracy'] for v in linear_results.values()])
            logger.info(f"  📏 線形系平均: {avg_linear_acc:.4f}")
        
        # その他
        other_results = {k: v for k, v in results.items() if k in other_based}
        if other_results:
            avg_other_acc = np.mean([v['balanced_accuracy'] for v in other_results.values()])
            logger.info(f"  🔮 その他平均: {avg_other_acc:.4f}")
        
        # アンサンブル
        ensemble_results = {k: v for k, v in results.items() if k in ensemble_based}
        if ensemble_results:
            avg_ensemble_acc = np.mean([v['balanced_accuracy'] for v in ensemble_results.values()])
            logger.info(f"  🤝 アンサンブル平均: {avg_ensemble_acc:.4f}")
        
        # 多様性分析
        all_accuracies = [v['balanced_accuracy'] for v in results.values()]
        accuracy_std = np.std(all_accuracies)
        accuracy_mean = np.mean(all_accuracies)
        
        logger.info(f"\n📈 アルゴリズム多様性:")
        logger.info(f"  平均精度: {accuracy_mean:.4f}")
        logger.info(f"  標準偏差: {accuracy_std:.4f}")
        logger.info(f"  変動係数: {accuracy_std/accuracy_mean:.4f}")
        
        if accuracy_std > 0.05:
            logger.info("  ✅ 高い多様性 - アンサンブル効果期待大")
        elif accuracy_std > 0.02:
            logger.info("  ⚠️ 中程度の多様性 - アンサンブル効果あり")
        else:
            logger.info("  ❌ 低い多様性 - アンサンブル効果限定的")
        
        # 推奨事項
        logger.info(f"\n💡 推奨事項:")
        if best_method in ensemble_based:
            logger.info("  🎉 アンサンブル手法が最高性能を達成！")
        else:
            logger.info(f"  🔍 {best_method}が最高性能 - アンサンブルでさらなる向上の余地")
        
        if accuracy_std > 0.03:
            logger.info("  📈 高い多様性を活用したスタッキングアンサンブルを推奨")
        
        logger.info("  🚀 上位3-5アルゴリズムでの軽量アンサンブルも検討価値あり")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    
    # 拡張アンサンブルテスト実行
    test = ExpandedEnsembleTest()
    results = test.test_expanded_ensemble_performance()
    
    logger.info("\n🎉 拡張アンサンブル学習テスト完了")
