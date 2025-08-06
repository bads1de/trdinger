"""
データリーク分析テスト

改善前後の精度差がデータリーク防止によるものかを検証し、
真の改善効果を測定します。
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class DataLeakageAnalysis:
    """データリーク分析クラス"""

    def __init__(self):
        self.results = {}

    def create_trading_dataset_with_leakage_test(self, n_samples=1000):
        """データリークテスト用のデータセットを作成"""
        np.random.seed(42)
        
        # 時系列インデックス
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
        
        # より現実的な価格データ（強いトレンドを含む）
        base_price = 50000
        
        # 明確なトレンドパターンを作成
        trend_periods = n_samples // 4
        trends = []
        for i in range(4):
            if i % 2 == 0:
                # 上昇トレンド
                trend = np.linspace(0, 0.1, trend_periods)
            else:
                # 下降トレンド
                trend = np.linspace(0, -0.1, trend_periods)
            trends.extend(trend)
        
        # 残りの期間を埋める
        while len(trends) < n_samples:
            trends.append(trends[-1])
        trends = trends[:n_samples]
        
        # 価格生成
        price_changes = np.array(trends) / n_samples + np.random.normal(0, 0.01, n_samples)
        prices = base_price * np.cumprod(1 + price_changes)
        
        # OHLCV データ
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 1, n_samples),
        }, index=dates)
        
        # 技術指標
        data['SMA_5'] = data['Close'].rolling(5).mean()
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'] = self._calculate_macd(data['Close'])
        data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(10).mean()
        
        # 将来の情報を含む特徴量（データリークの原因）
        data['Future_Price_Hint'] = data['Close'].shift(-12)  # 12時間後の価格
        data['Future_Volume_Hint'] = data['Volume'].shift(-6)  # 6時間後の出来高
        
        # ターゲット生成（12時間後の価格変動）
        future_returns = data['Close'].pct_change(12).shift(-12)
        
        # 3クラス分類（より明確な閾値）
        y = pd.Series(1, index=dates)  # Hold
        y[future_returns > 0.015] = 2   # Up (1.5%以上上昇)
        y[future_returns < -0.015] = 0  # Down (1.5%以上下落)
        
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

    def test_with_data_leakage(self, X, y):
        """データリークありのテスト（改善前の問題のある方法）"""
        logger.info("🔴 データリークありのテスト（改善前）")
        
        # 問題のある方法：ランダム分割（将来の情報が学習に混入）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 将来の情報を含む特徴量も使用
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'method': 'データリークあり（ランダム分割）'
        }
        
        # 特徴量重要度を確認
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        logger.info(f"  精度: {results['accuracy']:.4f}")
        logger.info(f"  バランス精度: {results['balanced_accuracy']:.4f}")
        logger.info(f"  F1スコア: {results['f1_score']:.4f}")
        logger.info(f"  最重要特徴量: {feature_importance.head(3).to_dict()}")
        
        return results, feature_importance

    def test_without_data_leakage_basic(self, X, y):
        """データリークなしのテスト（基本的な改善）"""
        logger.info("🟡 データリークなし（基本改善）")
        
        # 将来の情報を含む特徴量を除去
        leak_features = ['Future_Price_Hint', 'Future_Volume_Hint']
        X_clean = X.drop(columns=[col for col in leak_features if col in X.columns])
        
        # 時系列分割
        split_point = int(len(X_clean) * 0.7)
        X_train = X_clean.iloc[:split_point]
        X_test = X_clean.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'method': 'データリークなし（基本改善）'
        }
        
        logger.info(f"  精度: {results['accuracy']:.4f}")
        logger.info(f"  バランス精度: {results['balanced_accuracy']:.4f}")
        logger.info(f"  F1スコア: {results['f1_score']:.4f}")
        
        return results

    def test_with_full_improvements(self, X, y):
        """完全改善版のテスト"""
        logger.info("🟢 完全改善版（全改善適用）")
        
        # 将来の情報を含む特徴量を除去
        leak_features = ['Future_Price_Hint', 'Future_Volume_Hint']
        X_clean = X.drop(columns=[col for col in leak_features if col in X.columns])
        
        # 時系列分割
        split_point = int(len(X_clean) * 0.7)
        X_train = X_clean.iloc[:split_point]
        X_test = X_clean.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        # RobustScaler適用
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
        temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
        temp_model.fit(X_train_scaled, y_train)
        
        feature_importance = pd.Series(
            temp_model.feature_importances_,
            index=X_train_scaled.columns
        ).sort_values(ascending=False)
        
        # 上位特徴量を選択
        top_features = feature_importance.head(min(8, len(feature_importance))).index
        X_train_selected = X_train_scaled[top_features]
        X_test_selected = X_test_scaled[top_features]
        
        # 改善されたモデル
        model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight='balanced'  # クラス不均衡対応
        )
        model.fit(X_train_selected, y_train)
        
        y_pred = model.predict(X_test_selected)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'method': '完全改善版',
            'selected_features': top_features.tolist()
        }
        
        logger.info(f"  精度: {results['accuracy']:.4f}")
        logger.info(f"  バランス精度: {results['balanced_accuracy']:.4f}")
        logger.info(f"  F1スコア: {results['f1_score']:.4f}")
        logger.info(f"  選択特徴量: {len(top_features)}個")
        
        return results

    def test_time_series_cv(self, X, y):
        """時系列クロスバリデーションテスト"""
        logger.info("🔵 時系列クロスバリデーション")
        
        # 将来の情報を含む特徴量を除去
        leak_features = ['Future_Price_Hint', 'Future_Volume_Hint']
        X_clean = X.drop(columns=[col for col in leak_features if col in X.columns])
        
        # RobustScaler適用
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        # 時系列クロスバリデーション
        tscv = TimeSeriesSplit(n_splits=3)
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        cv_scores = []
        cv_balanced_scores = []
        cv_f1_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train_cv = X_scaled.iloc[train_idx]
            X_test_cv = X_scaled.iloc[test_idx]
            y_train_cv = y.iloc[train_idx]
            y_test_cv = y.iloc[test_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_test_cv)
            
            cv_scores.append(accuracy_score(y_test_cv, y_pred_cv))
            cv_balanced_scores.append(balanced_accuracy_score(y_test_cv, y_pred_cv))
            cv_f1_scores.append(f1_score(y_test_cv, y_pred_cv, average='weighted'))
        
        results = {
            'accuracy': np.mean(cv_scores),
            'accuracy_std': np.std(cv_scores),
            'balanced_accuracy': np.mean(cv_balanced_scores),
            'balanced_accuracy_std': np.std(cv_balanced_scores),
            'f1_score': np.mean(cv_f1_scores),
            'f1_std': np.std(cv_f1_scores),
            'method': '時系列クロスバリデーション'
        }
        
        logger.info(f"  精度: {results['accuracy']:.4f} ± {results['accuracy_std']:.4f}")
        logger.info(f"  バランス精度: {results['balanced_accuracy']:.4f} ± {results['balanced_accuracy_std']:.4f}")
        logger.info(f"  F1スコア: {results['f1_score']:.4f} ± {results['f1_std']:.4f}")
        
        return results

    def run_comprehensive_analysis(self):
        """包括的な分析を実行"""
        logger.info("=" * 80)
        logger.info("🔍 データリーク分析とMLモデル改善効果の検証")
        logger.info("=" * 80)
        
        # データセット作成
        X, y = self.create_trading_dataset_with_leakage_test(n_samples=1000)
        
        logger.info(f"データセット: {len(X)}サンプル, {X.shape[1]}特徴量")
        logger.info(f"ラベル分布: {y.value_counts().to_dict()}")
        
        # 各手法でテスト
        results = {}
        
        # 1. データリークありのテスト
        results['leakage'], feature_importance = self.test_with_data_leakage(X, y)
        
        # 2. データリークなし（基本改善）
        results['basic_improvement'] = self.test_without_data_leakage_basic(X, y)
        
        # 3. 完全改善版
        results['full_improvement'] = self.test_with_full_improvements(X, y)
        
        # 4. 時系列クロスバリデーション
        results['time_series_cv'] = self.test_time_series_cv(X, y)
        
        # 分析結果をまとめる
        self._analyze_comprehensive_results(results, feature_importance)
        
        return results

    def _analyze_comprehensive_results(self, results, feature_importance):
        """包括的な結果分析"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 包括的分析結果")
        logger.info("=" * 80)
        
        # データリークの影響を分析
        leakage_accuracy = results['leakage']['accuracy']
        basic_accuracy = results['basic_improvement']['accuracy']
        full_accuracy = results['full_improvement']['accuracy']
        cv_accuracy = results['time_series_cv']['accuracy']
        
        logger.info("🎯 精度比較:")
        logger.info(f"  データリークあり: {leakage_accuracy:.4f}")
        logger.info(f"  基本改善: {basic_accuracy:.4f}")
        logger.info(f"  完全改善: {full_accuracy:.4f}")
        logger.info(f"  時系列CV: {cv_accuracy:.4f}")
        
        # データリークの影響度
        leakage_impact = (leakage_accuracy - basic_accuracy) / leakage_accuracy * 100
        logger.info(f"\n🚨 データリークの影響: {leakage_impact:.1f}%の精度水増し")
        
        # 真の改善効果
        true_improvement = (full_accuracy - basic_accuracy) / basic_accuracy * 100
        logger.info(f"🎉 真の改善効果: {true_improvement:+.1f}%")
        
        # 特徴量重要度分析
        logger.info(f"\n🔍 特徴量重要度分析（データリークありモデル）:")
        top_features = feature_importance.head(5)
        for feature, importance in top_features.items():
            leak_indicator = "🚨" if "Future" in feature else "✅"
            logger.info(f"  {leak_indicator} {feature}: {importance:.4f}")
        
        # 安定性評価
        cv_stability = results['time_series_cv']['accuracy_std'] / results['time_series_cv']['accuracy']
        logger.info(f"\n📈 モデル安定性 (CV): {cv_stability:.4f}")
        
        if cv_stability < 0.1:
            stability_rating = "非常に安定"
        elif cv_stability < 0.2:
            stability_rating = "安定"
        else:
            stability_rating = "不安定"
        
        logger.info(f"  評価: {stability_rating}")
        
        # 総合評価
        logger.info(f"\n🏆 総合評価:")
        logger.info(f"  ✅ データリーク防止: 成功（{leakage_impact:.1f}%の水増し除去）")
        logger.info(f"  ✅ 真の改善効果: {true_improvement:+.1f}%")
        logger.info(f"  ✅ モデル安定性: {stability_rating}")
        
        if true_improvement > 5:
            logger.info(f"  🎉 有意な改善効果を確認！")
        elif true_improvement > 0:
            logger.info(f"  ⚠️ 軽微な改善効果")
        else:
            logger.info(f"  ❌ 改善効果が見られません")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    
    # 包括的分析実行
    analysis = DataLeakageAnalysis()
    results = analysis.run_comprehensive_analysis()
    
    logger.info("\n🎉 データリーク分析完了")
