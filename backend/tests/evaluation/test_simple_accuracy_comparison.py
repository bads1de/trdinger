"""
シンプルなMLモデル精度比較テスト

改善前後のMLモデル性能を直接比較し、
分析報告書で予測された改善効果を検証します。
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class SimpleAccuracyComparison:
    """シンプルなMLモデル精度比較クラス"""

    def __init__(self):
        self.results = {}

    def create_trading_dataset(self, n_samples=1000):
        """取引データセットを作成"""
        np.random.seed(42)
        
        # 時系列インデックス
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
        
        # 価格データ（リアルな価格動向を模擬）
        base_price = 50000
        trend = np.cumsum(np.random.normal(0, 0.001, n_samples))
        noise = np.random.normal(0, 0.02, n_samples)
        price_changes = trend + noise
        prices = base_price * np.cumprod(1 + price_changes)
        
        # 基本的なOHLCVデータ
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 1, n_samples),
        }, index=dates)
        
        # 技術指標を追加
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'] = self._calculate_macd(data['Close'])
        data['BB_upper'], data['BB_lower'] = self._calculate_bollinger_bands(data['Close'])
        data['Volume_SMA'] = data['Volume'].rolling(10).mean()
        
        # ファンディングレート（8時間間隔を1時間に補間）
        fr_8h = np.random.normal(0.0001, 0.0005, n_samples // 8)
        fr_1h = np.repeat(fr_8h, 8)[:n_samples]
        data['Funding_Rate'] = fr_1h
        
        # 建玉残高（1時間間隔）
        data['Open_Interest'] = np.random.lognormal(15, 0.5, n_samples)
        data['OI_Change'] = data['Open_Interest'].pct_change()
        
        # ターゲット生成（24時間後の価格変動）
        future_returns = data['Close'].pct_change(24).shift(-24)
        
        # 3クラス分類
        y = pd.Series(1, index=dates)  # Hold
        y[future_returns > 0.02] = 2   # Up (2%以上上昇)
        y[future_returns < -0.02] = 0  # Down (2%以上下落)
        
        # 有効なデータのみ
        valid_mask = data.notna().all(axis=1) & y.notna()
        data = data[valid_mask]
        y = y[valid_mask]
        
        return data, y

    def _calculate_rsi(self, prices, window=14):
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_macd(self, prices, fast=12, slow=26):
        """MACD計算"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """ボリンジャーバンド計算"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper.fillna(prices), lower.fillna(prices)

    def test_baseline_model(self, X, y):
        """改善前のベースラインモデル"""
        logger.info("🔴 改善前のベースラインモデル")
        
        # 問題1: ランダム分割（データリークあり）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 問題2: スケーリングなし
        # 問題3: 特徴量選択なし
        
        # モデル学習
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
        model.fit(X_train, y_train)
        
        # 予測と評価
        y_pred = model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X.shape[1]
        }
        
        logger.info(f"  精度: {results['accuracy']:.4f}")
        logger.info(f"  バランス精度: {results['balanced_accuracy']:.4f}")
        logger.info(f"  F1スコア: {results['f1_score']:.4f}")
        logger.info(f"  特徴量数: {results['features']}")
        
        return results

    def test_improved_model(self, X, y):
        """改善後のモデル"""
        logger.info("🟢 改善後のモデル")
        
        # 改善1: 時系列分割（データリーク防止）
        split_point = int(len(X) * 0.7)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        # 改善2: RobustScaler適用
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
        
        # 改善3: 特徴量選択（重要度ベース）
        temp_model = RandomForestClassifier(n_estimators=30, random_state=42)
        temp_model.fit(X_train_scaled, y_train)
        
        # 重要度上位の特徴量を選択
        feature_importance = pd.Series(
            temp_model.feature_importances_,
            index=X_train_scaled.columns
        ).sort_values(ascending=False)
        
        top_features = feature_importance.head(min(15, len(feature_importance))).index
        X_train_selected = X_train_scaled[top_features]
        X_test_selected = X_test_scaled[top_features]
        
        # 改善4: より良いハイパーパラメータ
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_selected, y_train)
        
        # 予測と評価
        y_pred = model.predict(X_test_selected)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(top_features),
            'selected_features': top_features.tolist()
        }
        
        logger.info(f"  精度: {results['accuracy']:.4f}")
        logger.info(f"  バランス精度: {results['balanced_accuracy']:.4f}")
        logger.info(f"  F1スコア: {results['f1_score']:.4f}")
        logger.info(f"  特徴量数: {results['features']} (選択後)")
        
        return results

    def run_comparison(self):
        """比較テストを実行"""
        logger.info("=" * 60)
        logger.info("📊 MLモデル精度改善効果の検証")
        logger.info("=" * 60)
        
        # データセット作成
        X, y = self.create_trading_dataset(n_samples=1200)
        
        logger.info(f"データセット: {len(X)}サンプル, {X.shape[1]}特徴量")
        logger.info(f"ラベル分布: {y.value_counts().to_dict()}")
        
        # ベースラインモデル
        baseline_results = self.test_baseline_model(X, y)
        
        # 改善モデル
        improved_results = self.test_improved_model(X, y)
        
        # 改善効果の分析
        self._analyze_improvement(baseline_results, improved_results)
        
        return baseline_results, improved_results

    def _analyze_improvement(self, baseline, improved):
        """改善効果を分析"""
        logger.info("\n" + "=" * 60)
        logger.info("📈 改善効果の分析")
        logger.info("=" * 60)
        
        # 改善率計算
        accuracy_improvement = (improved['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100
        balanced_acc_improvement = (improved['balanced_accuracy'] - baseline['balanced_accuracy']) / baseline['balanced_accuracy'] * 100
        f1_improvement = (improved['f1_score'] - baseline['f1_score']) / baseline['f1_score'] * 100
        
        logger.info("🎯 主要指標の改善効果:")
        logger.info(f"  精度: {baseline['accuracy']:.4f} → {improved['accuracy']:.4f} ({accuracy_improvement:+.1f}%)")
        logger.info(f"  バランス精度: {baseline['balanced_accuracy']:.4f} → {improved['balanced_accuracy']:.4f} ({balanced_acc_improvement:+.1f}%)")
        logger.info(f"  F1スコア: {baseline['f1_score']:.4f} → {improved['f1_score']:.4f} ({f1_improvement:+.1f}%)")
        
        # 特徴量効率性
        baseline_efficiency = baseline['accuracy'] / baseline['features']
        improved_efficiency = improved['accuracy'] / improved['features']
        efficiency_improvement = (improved_efficiency - baseline_efficiency) / baseline_efficiency * 100
        
        logger.info(f"\n🔧 特徴量効率性:")
        logger.info(f"  改善前: {baseline_efficiency:.6f} (精度/特徴量数)")
        logger.info(f"  改善後: {improved_efficiency:.6f} (精度/特徴量数)")
        logger.info(f"  効率性改善: {efficiency_improvement:+.1f}%")
        
        # 分析報告書との比較
        logger.info(f"\n📋 分析報告書予測との比較:")
        logger.info(f"  予測改善率: 20-30%")
        logger.info(f"  実際の精度改善: {accuracy_improvement:+.1f}%")
        logger.info(f"  実際のバランス精度改善: {balanced_acc_improvement:+.1f}%")
        
        # 総合評価
        avg_improvement = (accuracy_improvement + balanced_acc_improvement + f1_improvement) / 3
        
        if avg_improvement >= 20:
            logger.info("  ✅ 予測を上回る優秀な改善効果！")
        elif avg_improvement >= 10:
            logger.info("  ✅ 有意な改善効果を確認")
        elif avg_improvement >= 5:
            logger.info("  ⚠️ 軽微な改善効果")
        else:
            logger.info("  ❌ 改善効果が限定的")
        
        logger.info(f"\n🏆 総合改善率: {avg_improvement:+.1f}%")
        
        # 改善要因の詳細
        logger.info(f"\n🔍 実装された改善要因:")
        logger.info(f"  ✅ データリーク防止: 時系列分割採用")
        logger.info(f"  ✅ 特徴量スケーリング: RobustScaler適用")
        logger.info(f"  ✅ 特徴量選択: {baseline['features']} → {improved['features']}特徴量")
        logger.info(f"  ✅ ハイパーパラメータ最適化: 実施")
        
        return {
            'accuracy_improvement': accuracy_improvement,
            'balanced_accuracy_improvement': balanced_acc_improvement,
            'f1_improvement': f1_improvement,
            'average_improvement': avg_improvement,
            'efficiency_improvement': efficiency_improvement
        }


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    
    # 比較テスト実行
    comparison = SimpleAccuracyComparison()
    baseline_results, improved_results = comparison.run_comparison()
    
    logger.info("\n🎉 精度改善効果検証完了")
