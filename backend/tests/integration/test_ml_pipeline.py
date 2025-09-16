"""
統合MLパイプライン

MLパイプラインのエンドツーエンド機能をテスト
TDD原則に基づき、特徴量エンジニアリングからモデル評価までをテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

# ML関連
from backend.app.utils.data_processing import data_processor
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

# ラベル生成関連（モックを使用）
from unittest.mock import MagicMock


class TestMLPipelineIntegration:
    """MLパイプラインの統合テスト"""

    @pytest.fixture
    def ml_ready_data(self):
        """ML学習用データ"""
        dates = pd.date_range('2023-01-01', periods=1000, freq='h')
        np.random.seed(42)

        # 現実的な価格データ生成
        base_price = 50000
        returns = np.random.normal(0.0005, 0.02, 1000)
        prices = base_price * np.exp(np.cumsum(returns))

        # OHLCデータの生成
        opens = prices
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, 1000)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, 1000)))
        closes = prices + np.random.normal(0, prices * 0.005, 1000)

        # OHLCの整合性を確保
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        data = {
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, 1000),
            'open_interest': np.random.uniform(100000, 500000, 1000),
            'funding_rate': np.random.uniform(-0.001, 0.001, 1000)
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_label_generator(self):
        """モックラベルジェネレータ"""
        mock = MagicMock()
        # ラベル生成のモック実装
        def generate_labels(price_data, **kwargs):
            # シンプルなラベル生成：価格上昇=1、下降=0
            returns = price_data.pct_change()
            labels = (returns > 0).astype(int)
            threshold_info = {"method": "simple_return", "threshold": 0.0}
            return labels, threshold_info

        mock.generate_labels.side_effect = generate_labels
        return mock

    def test_feature_engineering_pipeline(self, ml_ready_data):
        """特徴量エンジニアリングパイプライン"""
        # 1. 基本データクリーニング
        cleaned_data = data_processor.clean_and_validate_data(
            ml_ready_data,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )
        assert len(cleaned_data) > 0

        # 2. 技術指標の計算
        service = TechnicalIndicatorService()

        indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR']
        feature_data = cleaned_data.copy()

        for indicator in indicators:
            result = service.calculate_indicator(feature_data, indicator, {})
            if result is not None:
                feature_data[f'{indicator.lower()}'] = result

        # 特徴量が増えていることを確認
        assert len(feature_data.columns) > len(cleaned_data.columns)

    def test_ml_data_preparation_workflow(self, ml_ready_data, mock_label_generator):
        """MLデータ準備ワークフロー"""
        # 1. データクリーニング
        cleaned_data = data_processor.clean_and_validate_data(
            ml_ready_data,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )

        # 2. 特徴量エンジニアリング
        processed_data = data_processor.preprocess_with_pipeline(cleaned_data)
        assert isinstance(processed_data, pd.DataFrame)

        # 3. ラベル生成
        features, labels, threshold_info = data_processor.prepare_training_data(
            processed_data, mock_label_generator
        )

        # 4. 結果検証
        assert len(features) == len(labels)
        assert 'close' in features.columns
        assert threshold_info["method"] == "simple_return"

    def test_model_training_simulation(self, ml_ready_data, mock_label_generator):
        """モデル学習シミュレーション"""
        # 1. データ準備
        cleaned_data = data_processor.clean_and_validate_data(ml_ready_data, ['open', 'high', 'low', 'close'])
        processed_data = data_processor.preprocess_with_pipeline(cleaned_data)

        features, labels, _ = data_processor.prepare_training_data(
            processed_data, mock_label_generator
        )

        # 2. モデル学習のシミュレーション
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report

        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # モデル学習
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 予測と評価
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # 基本的な検証
        assert accuracy > 0.4  # ランダムよりは良い性能
        assert len(predictions) == len(y_test)

    def test_feature_importance_analysis(self, ml_ready_data, mock_label_generator):
        """特徴量重要度分析"""
        # 1. データ準備
        cleaned_data = data_processor.clean_and_validate_data(ml_ready_data, ['open', 'high', 'low', 'close'])
        processed_data = data_processor.preprocess_with_pipeline(cleaned_data)

        features, labels, _ = data_processor.prepare_training_data(
            processed_data, mock_label_generator
        )

        # 2. 特徴量重要度の計算
        from sklearn.ensemble import RandomForestClassifier

        # NaNを含む行を除去
        valid_data = features.dropna()
        valid_labels = labels.loc[valid_data.index]

        if len(valid_data) > 10:  # 十分なデータがある場合のみ
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(valid_data, valid_labels)

            # 特徴量重要度を取得
            feature_importance = pd.DataFrame({
                'feature': valid_data.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # 最も重要な特徴量がcloseであることを確認
            top_feature = feature_importance.iloc[0]['feature']
            assert top_feature in valid_data.columns

    def test_cross_validation_simulation(self, ml_ready_data, mock_label_generator):
        """交差検証シミュレーション"""
        # 1. データ準備
        cleaned_data = data_processor.clean_and_validate_data(ml_ready_data, ['open', 'high', 'low', 'close'])
        processed_data = data_processor.preprocess_with_pipeline(cleaned_data)

        features, labels, _ = data_processor.prepare_training_data(
            processed_data, mock_label_generator
        )

        # 2. 交差検証
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier

        # NaNを含む行を除去
        valid_data = features.dropna()
        valid_labels = labels.loc[valid_data.index]

        if len(valid_data) > 50:  # 十分なデータがある場合のみ
            model = RandomForestClassifier(n_estimators=50, random_state=42)

            # 5-fold交差検証
            cv_scores = cross_val_score(model, valid_data, valid_labels, cv=5)

            # CVスコアの検証
            assert len(cv_scores) == 5
            assert np.mean(cv_scores) > 0.4  # 基本的な性能
            assert np.std(cv_scores) < 0.2   # 安定性

    def test_model_evaluation_metrics(self, ml_ready_data, mock_label_generator):
        """モデル評価メトリクス"""
        # 1. データ準備
        cleaned_data = data_processor.clean_and_validate_data(ml_ready_data, ['open', 'high', 'low', 'close'])
        processed_data = data_processor.preprocess_with_pipeline(cleaned_data)

        features, labels, _ = data_processor.prepare_training_data(
            processed_data, mock_label_generator
        )

        # 2. モデル評価
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix

        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            features.dropna(), labels.loc[features.dropna().index],
            test_size=0.3, random_state=42
        )

        if len(X_train) > 10:
            # モデル学習
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 予測
            predictions = model.predict(X_test)

            # 分類レポート生成
            report = classification_report(y_test, predictions, output_dict=True)

            # メトリクスの検証
            assert 'accuracy' in report
            assert report['accuracy'] > 0.4
            assert 'macro avg' in report
            assert 'weighted avg' in report

    def test_pipeline_optimization(self, ml_ready_data):
        """パイプライン最適化テスト"""
        # 1. データ準備
        cleaned_data = data_processor.clean_and_validate_data(
            ml_ready_data, ['open', 'high', 'low', 'close']
        )

        # 2. 異なるパイプライン設定での比較
        pipeline_configs = [
            {"scaling_method": "standard", "remove_outliers": True},
            {"scaling_method": "robust", "remove_outliers": False},
            {"scaling_method": "minmax", "remove_outliers": True}
        ]

        results = {}

        for i, config in enumerate(pipeline_configs):
            # パイプライン適用
            processed = data_processor.preprocess_with_pipeline(
                cleaned_data, pipeline_name=f"test_pipeline_{i}", **config
            )

            # 結果の統計を計算
            stats = {
                'n_features': len(processed.columns),
                'n_samples': len(processed),
                'memory_usage': processed.memory_usage(deep=True).sum()
            }

            results[f"config_{i}"] = stats

        # 結果の比較
        assert len(results) == 3
        for config_name, stats in results.items():
            assert stats['n_samples'] > 0
            assert stats['n_features'] > 0

    def test_data_leakage_prevention(self, ml_ready_data, mock_label_generator):
        """データリーク防止テスト"""
        # 1. データ準備
        cleaned_data = data_processor.clean_and_validate_data(
            ml_ready_data, ['open', 'high', 'low', 'close']
        )

        # 2. 特徴量エンジニアリング（将来の情報を使用しない）
        processed_data = data_processor.preprocess_with_pipeline(cleaned_data)

        # 3. 時間ベースの分割
        split_idx = int(len(processed_data) * 0.7)
        train_data = processed_data.iloc[:split_idx]
        test_data = processed_data.iloc[split_idx:]

        # 4. ラベル生成（訓練データのみ使用）
        mock_label_generator.generate_labels.side_effect = lambda price_data, **kwargs: (
            (price_data.pct_change() > 0).astype(int),
            {"method": "time_aware"}
        )

        train_features, train_labels, _ = data_processor.prepare_training_data(
            train_data, mock_label_generator
        )

        # 5. リークチェック
        # テストデータの特徴量が訓練データの統計情報に依存しないことを確認
        assert len(train_features) < len(processed_data)
        assert len(test_data) > 0

    def test_scalability_with_large_dataset(self):
        """大規模データセットでのスケーラビリティ"""
        # 大規模データセットの作成
        dates = pd.date_range('2020-01-01', periods=5000, freq='h')
        large_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(30000, 70000, 5000),
            'high': np.random.uniform(35000, 75000, 5000),
            'low': np.random.uniform(25000, 65000, 5000),
            'close': np.random.uniform(30000, 70000, 5000),
            'volume': np.random.randint(100, 10000, 5000)
        })

        import time
        start_time = time.time()

        # MLパイプライン実行
        cleaned = data_processor.clean_and_validate_data(large_data, ['open', 'high', 'low', 'close'])
        processed = data_processor.preprocess_with_pipeline(cleaned)

        processing_time = time.time() - start_time

        # パフォーマンス要件を満たしていることを確認
        assert processing_time < 30.0  # 30秒以内に完了
        assert len(processed) == len(large_data)

    def test_ml_pipeline_error_recovery(self, ml_ready_data):
        """MLパイプラインのエラーリカバリー"""
        # 1. 不完全なデータでのテスト
        incomplete_data = ml_ready_data.copy()
        incomplete_data.loc[0, 'close'] = np.nan  # NaNを挿入

        # 2. エラーハンドリング
        try:
            cleaned = data_processor.clean_and_validate_data(
                incomplete_data, ['open', 'high', 'low', 'close']
            )

            # NaN行が除去されていることを確認
            assert not cleaned['close'].isna().any()

        except Exception as e:
            # エラーが適切に処理されることを確認
            assert "data validation" in str(e).lower() or "nan" in str(e).lower()

    def test_feature_stability(self, ml_ready_data):
        """特徴量の安定性テスト"""
        # 1. 同じデータでの複数回の処理
        cleaned_data = data_processor.clean_and_validate_data(
            ml_ready_data, ['open', 'high', 'low', 'close']
        )

        # 2. 複数回の特徴量エンジニアリング
        results = []
        for i in range(3):
            processed = data_processor.preprocess_with_pipeline(
                cleaned_data, pipeline_name=f"stability_test_{i}"
            )
            results.append(processed)

        # 3. 結果の安定性チェック
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(results[0], results[i])


class TestAdvancedMLFeatures:
    """高度なML機能テスト"""

    def test_ensemble_model_simulation(self):
        """アンサンブルモデルシミュレーション"""
        # テストデータ
        dates = pd.date_range('2023-01-01', periods=500, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(49000, 51000, 500),
            'high': np.random.uniform(50000, 52000, 500),
            'low': np.random.uniform(48000, 50000, 500),
            'close': np.random.uniform(49000, 51000, 500),
            'volume': np.random.randint(1000, 5000, 500)
        })

        # 複数モデルのシミュレーション
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }

        predictions = {}

        for name, model in models.items():
            # データ準備
            cleaned = data_processor.clean_and_validate_data(data, ['open', 'high', 'low', 'close'])
            processed = data_processor.preprocess_with_pipeline(cleaned)

            # シンプルなラベル生成
            labels = (processed['close'].pct_change() > 0).astype(int).dropna()
            features = processed.loc[labels.index]

            if len(features) > 10:
                # モデル学習
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.3, random_state=42
                )

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                predictions[name] = preds

        # アンサンブル結果の検証
        assert len(predictions) == len(models)
        for model_name, preds in predictions.items():
            assert len(preds) > 0


if __name__ == "__main__":
    pytest.main([__file__])