"""
AutoML有効でのモデル学習テスト

TSFreshやAutoFeat特徴量が正しく生成されるかをテストします。
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.ml.ml_training_service import MLTrainingService
from app.services.ml.model_manager import model_manager
from app.services.ml.feature_engineering.automl_feature_analyzer import AutoMLFeatureAnalyzer

logger = logging.getLogger(__name__)


def test_automl_enabled_training():
    """AutoML有効でのモデル学習をテスト"""
    print("=" * 80)
    print("AutoML有効モデル学習テスト")
    print("=" * 80)
    
    try:
        # AutoML設定を作成
        automl_config = {
            "tsfresh": {
                "enabled": True,
                "feature_selection": True,
                "max_features": 30
            },
            "autofeat": {
                "enabled": True,
                "max_features": 20,
                "feateng_steps": 2
            }
        }
        
        print("AutoML設定:")
        print(f"  TSFresh: 有効, 最大特徴量数: {automl_config['tsfresh']['max_features']}")
        print(f"  AutoFeat: 有効, 最大特徴量数: {automl_config['autofeat']['max_features']}")
        
        # サンプルデータを作成（AutoMLのため少し小さくする）
        np.random.seed(42)
        n_samples = 200
        
        data = {
            "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="h"),
            "Open": 50000 + np.random.randn(n_samples) * 1000,
            "High": 50000 + np.random.randn(n_samples) * 1000 + 500,
            "Low": 50000 + np.random.randn(n_samples) * 1000 - 500,
            "Close": 50000 + np.random.randn(n_samples) * 1000,
            "Volume": np.random.uniform(100, 1000, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # 価格の整合性を保つ
        for i in range(len(df)):
            prices = [
                df.loc[i, "Open"],
                df.loc[i, "High"],
                df.loc[i, "Low"],
                df.loc[i, "Close"],
            ]
            df.loc[i, "High"] = max(prices)
            df.loc[i, "Low"] = min(prices)
        
        print(f"サンプルデータ作成: {len(df)}行")
        
        # AutoML有効でMLTrainingServiceを初期化
        ml_service = MLTrainingService(
            trainer_type="ensemble",
            automl_config=automl_config
        )
        
        print("AutoML有効でモデル学習を開始...")
        
        # モデルを学習
        result = ml_service.train_model(
            training_data=df,
            save_model=True,
            model_name="automl_test_model"
        )
        
        print("✅ AutoMLモデル学習完了")
        
        # 特徴量重要度を確認
        trainer = ml_service.trainer
        feature_importance = trainer.get_feature_importance(100)
        
        if feature_importance:
            print(f"生成された特徴量数: {len(feature_importance)}")
            
            # AutoML特徴量の確認
            ts_features = [name for name in feature_importance.keys() if name.startswith("TS_")]
            af_features = [name for name in feature_importance.keys() if name.startswith("AF_")]
            manual_features = [name for name in feature_importance.keys() if not (name.startswith("TS_") or name.startswith("AF_"))]
            
            print(f"TSFresh特徴量: {len(ts_features)}個")
            print(f"AutoFeat特徴量: {len(af_features)}個")
            print(f"手動特徴量: {len(manual_features)}個")
            
            if ts_features:
                print(f"TSFresh例: {ts_features[:3]}")
            if af_features:
                print(f"AutoFeat例: {af_features[:3]}")
            
            # AutoML特徴量分析を実行
            analyzer = AutoMLFeatureAnalyzer()
            analysis_result = analyzer.analyze_feature_importance(feature_importance, top_n=20)
            
            print("\nAutoML特徴量分析結果:")
            if "error" not in analysis_result:
                # タイプ別統計
                type_stats = analysis_result.get("type_statistics", {})
                print("\nタイプ別統計:")
                for type_name, stats in type_stats.items():
                    count = stats.get('count', 0)
                    ratio = stats.get('importance_ratio', 0)
                    print(f"  {type_name}: {count}個 ({ratio:.1f}%)")
                
                # AutoML効果
                automl_impact = analysis_result.get("automl_impact", {})
                print("\nAutoML効果:")
                print(f"  AutoML特徴量比率: {automl_impact.get('automl_feature_ratio', 0):.1f}%")
                print(f"  AutoML重要度比率: {automl_impact.get('automl_importance_ratio', 0):.1f}%")
                
                # 上位特徴量
                top_features = analysis_result.get("top_features", [])
                print(f"\n上位特徴量 (上位{min(10, len(top_features))}個):")
                for i, feature in enumerate(top_features[:10]):
                    print(f"  {i+1}. {feature['feature_name']} ({feature['feature_type']}) - {feature['importance']:.4f}")
                
                # 成功判定
                if ts_features or af_features:
                    print("\n✅ AutoML特徴量が正常に生成されました")
                    return True
                else:
                    print("\n⚠️ AutoML特徴量が生成されませんでした")
                    return False
            else:
                print(f"分析エラー: {analysis_result['error']}")
                return False
        else:
            print("❌ 特徴量重要度が取得できませんでした")
            return False
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_automl_feature_generation_debug():
    """AutoML特徴量生成のデバッグ"""
    print("\n" + "=" * 80)
    print("AutoML特徴量生成デバッグ")
    print("=" * 80)
    
    try:
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
        from app.services.ml.feature_engineering.automl_features.automl_config import AutoMLConfig
        
        # AutoML設定を作成
        automl_config = AutoMLConfig.get_financial_optimized_config()
        
        print("AutoML設定:")
        print(f"  TSFresh有効: {automl_config.tsfresh.enabled}")
        print(f"  AutoFeat有効: {automl_config.autofeat.enabled}")
        
        # サンプルデータを作成
        np.random.seed(42)
        n_samples = 100  # 小さなデータセットでテスト
        
        data = {
            "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="h"),
            "Open": 50000 + np.random.randn(n_samples) * 1000,
            "High": 50000 + np.random.randn(n_samples) * 1000 + 500,
            "Low": 50000 + np.random.randn(n_samples) * 1000 - 500,
            "Close": 50000 + np.random.randn(n_samples) * 1000,
            "Volume": np.random.uniform(100, 1000, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # 価格の整合性を保つ
        for i in range(len(df)):
            prices = [
                df.loc[i, "Open"],
                df.loc[i, "High"],
                df.loc[i, "Low"],
                df.loc[i, "Close"],
            ]
            df.loc[i, "High"] = max(prices)
            df.loc[i, "Low"] = min(prices)
        
        print(f"サンプルデータ作成: {len(df)}行")
        
        # ターゲット変数を作成
        target = (df['Close'].shift(-1) / df['Close'] - 1) * 100  # 次の期間のリターン
        target = target.dropna()
        
        print(f"ターゲット変数作成: {len(target)}行")
        
        # FeatureEngineeringServiceを初期化
        feature_service = FeatureEngineeringService(automl_config=automl_config)
        
        print("AutoML特徴量生成を開始...")
        
        # 拡張特徴量を計算
        enhanced_df = feature_service.calculate_enhanced_features(
            ohlcv_data=df,
            target=target[:len(df)-1]  # ターゲットの長さを調整
        )
        
        print(f"✅ 特徴量生成完了: {len(enhanced_df.columns)}個の特徴量")
        
        # 特徴量名を分析
        ts_features = [col for col in enhanced_df.columns if col.startswith("TS_")]
        af_features = [col for col in enhanced_df.columns if col.startswith("AF_")]
        manual_features = [col for col in enhanced_df.columns if not (col.startswith("TS_") or col.startswith("AF_"))]
        
        print(f"TSFresh特徴量: {len(ts_features)}個")
        print(f"AutoFeat特徴量: {len(af_features)}個")
        print(f"手動特徴量: {len(manual_features)}個")
        
        if ts_features:
            print(f"TSFresh例: {ts_features[:5]}")
        if af_features:
            print(f"AutoFeat例: {af_features[:5]}")
        
        # 統計情報を取得
        stats = feature_service.get_enhancement_stats()
        print(f"\n特徴量生成統計: {stats}")
        
        if ts_features or af_features:
            print("\n✅ AutoML特徴量が正常に生成されました")
            return True
        else:
            print("\n⚠️ AutoML特徴量が生成されませんでした")
            return False
        
    except Exception as e:
        print(f"❌ デバッグ実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("AutoML有効モデル学習テスト開始")
    print("=" * 80)
    
    # 1. AutoML特徴量生成のデバッグ
    debug_result = test_automl_feature_generation_debug()
    
    # 2. AutoML有効でのモデル学習テスト（デバッグが成功した場合のみ）
    if debug_result:
        training_result = test_automl_enabled_training()
    else:
        training_result = False
        print("⚠️ AutoML特徴量生成デバッグが失敗したため、学習テストをスキップします")
    
    print("\n" + "=" * 80)
    print("テスト結果:")
    print(f"1. AutoML特徴量生成デバッグ: {'✅ 成功' if debug_result else '❌ 失敗'}")
    print(f"2. AutoML有効モデル学習: {'✅ 成功' if training_result else '❌ 失敗'}")
    
    if debug_result and training_result:
        print("\n🎉 AutoML機能が完全に動作しています！")
    elif debug_result:
        print("\n⚠️ AutoML特徴量生成は成功しましたが、学習で問題があります。")
    else:
        print("\n⚠️ AutoML特徴量生成に問題があります。")
    
    print("=" * 80)
