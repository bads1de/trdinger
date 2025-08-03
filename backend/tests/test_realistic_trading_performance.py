"""
リアルな取引環境でのMLモデル性能テスト

実際の取引環境に近い条件でMLモデルの改善効果を検証し、
実用的な精度改善を測定します。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import RobustScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class RealisticTradingPerformanceTest:
    """リアルな取引環境でのMLモデル性能テストクラス"""

    def __init__(self):
        self.results = {}

    def create_realistic_market_data(self, n_samples=2000):
        """リアルな市場データを作成"""
        np.random.seed(42)

        # 時系列インデックス（1時間間隔）
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="1h")

        # より現実的な価格動向（複数の市場サイクルを含む）
        base_price = 50000

        # 複数の周期的パターンを組み合わせ
        daily_cycle = (
            np.sin(np.arange(n_samples) * 2 * np.pi / 24) * 0.005
        )  # 日次サイクル
        weekly_cycle = (
            np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 7)) * 0.01
        )  # 週次サイクル
        monthly_trend = (
            np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 30)) * 0.02
        )  # 月次トレンド

        # ランダムウォーク + 周期的パターン
        random_walk = np.cumsum(np.random.normal(0, 0.008, n_samples))
        price_pattern = daily_cycle + weekly_cycle + monthly_trend + random_walk

        # ボラティリティクラスタリング
        volatility = np.abs(np.random.normal(0.015, 0.005, n_samples))
        for i in range(1, n_samples):
            volatility[i] = 0.7 * volatility[i - 1] + 0.3 * volatility[i]

        # 価格生成
        price_changes = price_pattern + np.random.normal(0, volatility)
        prices = base_price * np.cumprod(1 + price_changes)

        # OHLCV データ
        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.001, n_samples)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.003, n_samples))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.003, n_samples))),
                "Close": prices,
                "Volume": np.random.lognormal(10, 1, n_samples),
            },
            index=dates,
        )

        # 現実的な技術指標
        data["Returns"] = data["Close"].pct_change()
        data["SMA_5"] = data["Close"].rolling(5).mean()
        data["SMA_10"] = data["Close"].rolling(10).mean()
        data["SMA_20"] = data["Close"].rolling(20).mean()
        data["EMA_12"] = data["Close"].ewm(span=12).mean()
        data["EMA_26"] = data["Close"].ewm(span=26).mean()

        # 技術指標
        data["RSI"] = self._calculate_rsi(data["Close"])
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["MACD_Signal"] = data["MACD"].ewm(span=9).mean()
        data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]

        # ボリンジャーバンド
        data["BB_Middle"] = data["Close"].rolling(20).mean()
        data["BB_Std"] = data["Close"].rolling(20).std()
        data["BB_Upper"] = data["BB_Middle"] + (data["BB_Std"] * 2)
        data["BB_Lower"] = data["BB_Middle"] - (data["BB_Std"] * 2)
        data["BB_Position"] = (data["Close"] - data["BB_Lower"]) / (
            data["BB_Upper"] - data["BB_Lower"]
        )

        # 出来高指標
        data["Volume_SMA"] = data["Volume"].rolling(10).mean()
        data["Volume_Ratio"] = data["Volume"] / data["Volume_SMA"]
        data["VWAP"] = (data["Close"] * data["Volume"]).rolling(20).sum() / data[
            "Volume"
        ].rolling(20).sum()

        # ボラティリティ指標
        data["Volatility"] = data["Returns"].rolling(20).std()
        data["ATR"] = self._calculate_atr(data)

        # モメンタム指標
        data["ROC"] = data["Close"].pct_change(10)
        data["Williams_R"] = self._calculate_williams_r(data)

        # ファンディングレート（8時間間隔を1時間に補間）
        fr_base = np.random.normal(0.0001, 0.0003, n_samples // 8)
        fr_1h = np.repeat(fr_base, 8)[:n_samples]
        data["Funding_Rate"] = fr_1h

        # 建玉残高
        oi_trend = np.cumsum(np.random.normal(0, 0.01, n_samples))
        data["Open_Interest"] = np.exp(
            15 + oi_trend + np.random.normal(0, 0.1, n_samples)
        )
        data["OI_Change"] = data["Open_Interest"].pct_change()

        # ターゲット生成（より現実的な予測期間）
        prediction_horizon = 6  # 6時間後の価格変動を予測
        future_returns = (
            data["Close"].pct_change(prediction_horizon).shift(-prediction_horizon)
        )

        # 動的閾値（ボラティリティベース）
        rolling_vol = data["Returns"].rolling(24).std()
        dynamic_threshold = rolling_vol * 1.0  # 1σの変動

        # 3クラス分類
        y = pd.Series(1, index=dates)  # Hold
        y[future_returns > dynamic_threshold] = 2  # Up
        y[future_returns < -dynamic_threshold] = 0  # Down

        # データクリーニング
        # 無限値を除去
        data = data.replace([np.inf, -np.inf], np.nan)

        # 有効なデータのみ
        valid_mask = data.notna().all(axis=1) & y.notna() & rolling_vol.notna()
        data = data[valid_mask]
        y = y[valid_mask]

        # 最終的なデータクリーニング
        data = data.fillna(data.median())

        return data, y

    def _calculate_rsi(self, prices, window=14):
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_atr(self, data, window=14):
        """ATR計算"""
        high_low = data["High"] - data["Low"]
        high_close = np.abs(data["High"] - data["Close"].shift())
        low_close = np.abs(data["Low"] - data["Close"].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr.fillna(true_range)

    def _calculate_williams_r(self, data, window=14):
        """Williams %R計算"""
        highest_high = data["High"].rolling(window=window).max()
        lowest_low = data["Low"].rolling(window=window).min()
        williams_r = -100 * (highest_high - data["Close"]) / (highest_high - lowest_low)
        return williams_r.fillna(-50)

    def _clean_data(self, X):
        """データクリーニング"""
        X_clean = X.copy()

        # 無限値を除去
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

        # 異常に大きな値を除去
        for col in X_clean.select_dtypes(include=[np.number]).columns:
            Q1 = X_clean[col].quantile(0.01)
            Q99 = X_clean[col].quantile(0.99)
            X_clean[col] = X_clean[col].clip(lower=Q1, upper=Q99)

        # 欠損値を補完
        X_clean = X_clean.fillna(X_clean.median())

        return X_clean

    def test_baseline_trading_model(self, X, y):
        """ベースライン取引モデル（改善前）"""
        logger.info("🔴 ベースライン取引モデル（改善前）")

        # データクリーニング
        X_clean = self._clean_data(X)

        # 時系列分割（改善前でもデータリークは防ぐ）
        split_point = int(len(X_clean) * 0.8)  # 学習期間を長く
        X_train = X_clean.iloc[:split_point]
        X_test = X_clean.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        # 基本的なモデル
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # 詳細な評価
        results = self._calculate_detailed_metrics(
            y_test, y_pred, y_proba, "ベースライン"
        )
        results.update(
            {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": X.shape[1],
            }
        )

        return results

    def test_improved_trading_model(self, X, y):
        """改善された取引モデル"""
        logger.info("🟢 改善された取引モデル")

        # データクリーニング
        X_clean = self._clean_data(X)

        # 時系列分割
        split_point = int(len(X_clean) * 0.8)
        X_train = X_clean.iloc[:split_point]
        X_test = X_clean.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        # RobustScaler適用
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # 特徴量選択
        temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
        temp_model.fit(X_train_scaled, y_train)

        feature_importance = pd.Series(
            temp_model.feature_importances_, index=X_train_scaled.columns
        ).sort_values(ascending=False)

        # 上位特徴量を選択
        top_features = feature_importance.head(min(15, len(feature_importance))).index
        X_train_selected = X_train_scaled[top_features]
        X_test_selected = X_test_scaled[top_features]

        # 改善されたモデル
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            max_features="sqrt",
        )
        model.fit(X_train_selected, y_train)

        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)

        # 詳細な評価
        results = self._calculate_detailed_metrics(y_test, y_pred, y_proba, "改善版")
        results.update(
            {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": len(top_features),
                "selected_features": top_features.tolist(),
            }
        )

        return results

    def test_time_series_cv_performance(self, X, y):
        """時系列クロスバリデーション性能テスト"""
        logger.info("🔵 時系列クロスバリデーション性能テスト")

        # データクリーニング
        X_clean = self._clean_data(X)

        # RobustScaler適用
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index
        )

        # 時系列クロスバリデーション
        tscv = TimeSeriesSplit(n_splits=5)
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=8, class_weight="balanced"
        )

        cv_results = {
            "accuracy": [],
            "balanced_accuracy": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
        }

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train_cv = X_scaled.iloc[train_idx]
            X_test_cv = X_scaled.iloc[test_idx]
            y_train_cv = y.iloc[train_idx]
            y_test_cv = y.iloc[test_idx]

            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_test_cv)

            # 各フォールドの結果
            cv_results["accuracy"].append(accuracy_score(y_test_cv, y_pred_cv))
            cv_results["balanced_accuracy"].append(
                balanced_accuracy_score(y_test_cv, y_pred_cv)
            )
            cv_results["f1_score"].append(
                f1_score(y_test_cv, y_pred_cv, average="weighted")
            )

            precision, recall, _, _ = precision_recall_fscore_support(
                y_test_cv, y_pred_cv, average="weighted", zero_division=0
            )
            cv_results["precision"].append(precision)
            cv_results["recall"].append(recall)

            logger.info(f"  フォールド {fold+1}: 精度={cv_results['accuracy'][-1]:.4f}")

        # 統計サマリー
        results = {}
        for metric, values in cv_results.items():
            results[f"{metric}_mean"] = np.mean(values)
            results[f"{metric}_std"] = np.std(values)
            results[f"{metric}_min"] = np.min(values)
            results[f"{metric}_max"] = np.max(values)

        results["method"] = "時系列クロスバリデーション"
        results["n_folds"] = len(cv_results["accuracy"])

        return results

    def _calculate_detailed_metrics(self, y_true, y_pred, y_proba, method_name):
        """詳細な評価指標を計算"""
        results = {
            "method": method_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
        }

        # クラス別の詳細指標
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        class_names = ["Down", "Hold", "Up"]
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                results[f"{class_name}_precision"] = precision[i]
                results[f"{class_name}_recall"] = recall[i]
                results[f"{class_name}_f1"] = f1[i]
                results[f"{class_name}_support"] = support[i]

        # 取引シグナルの精度（Up/Downクラスのみ）
        trading_mask = (y_true != 1) & (y_pred != 1)  # Holdを除外
        if trading_mask.sum() > 0:
            trading_accuracy = accuracy_score(
                y_true[trading_mask], y_pred[trading_mask]
            )
            results["trading_signal_accuracy"] = trading_accuracy
        else:
            results["trading_signal_accuracy"] = 0.0

        logger.info(f"  {method_name}結果:")
        logger.info(f"    精度: {results['accuracy']:.4f}")
        logger.info(f"    バランス精度: {results['balanced_accuracy']:.4f}")
        logger.info(f"    F1スコア: {results['f1_score']:.4f}")
        logger.info(f"    取引シグナル精度: {results['trading_signal_accuracy']:.4f}")

        return results

    def run_realistic_performance_test(self):
        """リアルな性能テストを実行"""
        logger.info("=" * 80)
        logger.info("🚀 リアルな取引環境でのMLモデル性能テスト")
        logger.info("=" * 80)

        # リアルなデータセット作成
        X, y = self.create_realistic_market_data(n_samples=2000)

        logger.info(f"データセット: {len(X)}サンプル, {X.shape[1]}特徴量")
        logger.info(f"ラベル分布: {y.value_counts().to_dict()}")

        # 各手法でテスト
        results = {}

        # 1. ベースラインモデル
        results["baseline"] = self.test_baseline_trading_model(X, y)

        # 2. 改善モデル
        results["improved"] = self.test_improved_trading_model(X, y)

        # 3. 時系列クロスバリデーション
        results["time_series_cv"] = self.test_time_series_cv_performance(X, y)

        # 結果分析
        self._analyze_realistic_performance(results)

        return results

    def _analyze_realistic_performance(self, results):
        """リアルな性能結果を分析"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 リアルな取引環境での性能分析結果")
        logger.info("=" * 80)

        baseline = results["baseline"]
        improved = results["improved"]
        cv_results = results["time_series_cv"]

        # 主要指標の改善効果
        accuracy_improvement = (
            (improved["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100
        )
        balanced_acc_improvement = (
            (improved["balanced_accuracy"] - baseline["balanced_accuracy"])
            / baseline["balanced_accuracy"]
            * 100
        )
        f1_improvement = (
            (improved["f1_score"] - baseline["f1_score"]) / baseline["f1_score"] * 100
        )
        trading_signal_improvement = (
            (improved["trading_signal_accuracy"] - baseline["trading_signal_accuracy"])
            / baseline["trading_signal_accuracy"]
            * 100
        )

        logger.info("🎯 主要指標の改善効果:")
        logger.info(
            f"  精度: {baseline['accuracy']:.4f} → {improved['accuracy']:.4f} ({accuracy_improvement:+.1f}%)"
        )
        logger.info(
            f"  バランス精度: {baseline['balanced_accuracy']:.4f} → {improved['balanced_accuracy']:.4f} ({balanced_acc_improvement:+.1f}%)"
        )
        logger.info(
            f"  F1スコア: {baseline['f1_score']:.4f} → {improved['f1_score']:.4f} ({f1_improvement:+.1f}%)"
        )
        logger.info(
            f"  取引シグナル精度: {baseline['trading_signal_accuracy']:.4f} → {improved['trading_signal_accuracy']:.4f} ({trading_signal_improvement:+.1f}%)"
        )

        # クロスバリデーション結果
        logger.info(f"\n📈 時系列クロスバリデーション結果:")
        logger.info(
            f"  平均精度: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}"
        )
        logger.info(
            f"  平均バランス精度: {cv_results['balanced_accuracy_mean']:.4f} ± {cv_results['balanced_accuracy_std']:.4f}"
        )
        logger.info(
            f"  平均F1スコア: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}"
        )

        # 安定性評価
        cv_stability = cv_results["accuracy_std"] / cv_results["accuracy_mean"]
        logger.info(f"  モデル安定性 (CV): {cv_stability:.4f}")

        # 総合評価
        avg_improvement = (
            accuracy_improvement + balanced_acc_improvement + f1_improvement
        ) / 3

        logger.info(f"\n🏆 総合評価:")
        logger.info(f"  平均改善率: {avg_improvement:+.1f}%")
        logger.info(f"  取引シグナル改善: {trading_signal_improvement:+.1f}%")
        logger.info(
            f"  モデル安定性: {'高' if cv_stability < 0.1 else '中' if cv_stability < 0.2 else '低'}"
        )

        if avg_improvement > 10:
            logger.info("  🎉 優秀な改善効果を確認！")
        elif avg_improvement > 5:
            logger.info("  ✅ 有意な改善効果を確認")
        elif avg_improvement > 0:
            logger.info("  ⚠️ 軽微な改善効果")
        else:
            logger.info("  ❌ 改善効果が見られません")

        # 実用性評価
        if improved["trading_signal_accuracy"] > 0.55:
            logger.info("  💰 実用的な取引シグナル精度")
        elif improved["trading_signal_accuracy"] > 0.50:
            logger.info("  📈 取引に使用可能な精度")
        else:
            logger.info("  ⚠️ 取引シグナル精度が不十分")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    # リアルな性能テスト実行
    test = RealisticTradingPerformanceTest()
    results = test.run_realistic_performance_test()

    logger.info("\n🎉 リアルな取引環境での性能テスト完了")
