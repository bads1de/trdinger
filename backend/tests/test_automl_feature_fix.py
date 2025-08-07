"""
AutoML特徴量分析修正のテストスクリプト

修正後のAutoML特徴量分析機能をテストします。
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
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator

logger = logging.getLogger(__name__)


def test_fixed_feature_importance_save():
    """修正後の特徴量重要度保存をテスト"""
    print("=" * 80)
    print("修正後の特徴量重要度保存テスト")
    print("=" * 80)
    
    try:
        # サンプルデータを作成
        np.random.seed(42)
        n_samples = 500
        
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
        
        # MLTrainingServiceを初期化
        ml_service = MLTrainingService(trainer_type="ensemble")
        
        print("モデル学習を開始...")
        
        # モデルを学習
        result = ml_service.train_model(
            training_data=df,
            save_model=False,
            model_name=None
        )
        
        print("✅ モデル学習完了")
        
        # 修正後の特徴量重要度取得をテスト
        print("\n修正後の特徴量重要度取得テスト:")
        print("-" * 60)
        
        trainer = ml_service.trainer
        feature_importance = trainer.get_feature_importance(top_n=100)
        
        if feature_importance:
            print(f"✅ 特徴量重要度取得成功: {len(feature_importance)}個")
            
            # 上位5個を表示
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for name, importance in sorted_importance:
                print(f"  {name}: {importance:.4f}")
        else:
            print("❌ 特徴量重要度が取得できませんでした")
            return False
        
        # モデル保存をテスト
        print("\n修正後のモデル保存テスト:")
        print("-" * 60)
        
        test_model_name = "test_fixed_feature_importance"
        
        try:
            model_path = trainer.save_model(test_model_name)
            print(f"✅ モデル保存完了: {model_path}")
            
            # 保存されたモデルを確認
            saved_model_data = model_manager.load_model(model_path)
            if saved_model_data and "metadata" in saved_model_data:
                metadata = saved_model_data["metadata"]
                
                if "feature_importance" in metadata:
                    saved_feature_importance = metadata["feature_importance"]
                    print(f"✅ 特徴量重要度がメタデータに保存されています: {len(saved_feature_importance)}個")
                    
                    # 上位5個を表示
                    sorted_importance = sorted(saved_feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    for name, importance in sorted_importance:
                        print(f"  {name}: {importance:.4f}")
                    
                    # テストファイルを削除
                    try:
                        os.remove(model_path)
                        print(f"✅ テストファイルを削除: {model_path}")
                    except Exception as e:
                        print(f"⚠️ テストファイル削除エラー: {e}")
                    
                    return True
                else:
                    print("❌ 特徴量重要度がメタデータに保存されていません")
                    return False
            else:
                print("❌ 保存されたモデルのメタデータを読み込めません")
                return False
                
        except Exception as e:
            print(f"❌ モデル保存エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_automl_feature_analysis_fix():
    """修正後のAutoML特徴量分析をテスト"""
    print("\n" + "=" * 80)
    print("修正後のAutoML特徴量分析テスト")
    print("=" * 80)
    
    try:
        # MLOrchestratorを初期化
        orchestrator = MLOrchestrator()
        
        # 特徴量重要度を取得
        feature_importance = orchestrator.get_feature_importance(100)
        print(f"特徴量重要度取得: {len(feature_importance)}個")
        
        if not feature_importance:
            print("❌ 特徴量重要度が取得できません")
            return False
        
        # AutoMLFeatureAnalyzerで分析
        analyzer = AutoMLFeatureAnalyzer()
        analysis_result = analyzer.analyze_feature_importance(feature_importance, top_n=20)
        
        print("\nAutoML特徴量分析結果:")
        print("-" * 60)
        
        if "error" in analysis_result:
            print(f"❌ 分析エラー: {analysis_result['error']}")
            return False
        
        # 結果の詳細表示
        print(f"総特徴量数: {analysis_result.get('total_features', 0)}")
        
        # タイプ別統計
        type_stats = analysis_result.get("type_statistics", {})
        print("\nタイプ別統計:")
        for type_name, stats in type_stats.items():
            count = stats.get('count', 0)
            ratio = stats.get('importance_ratio', 0)
            print(f"  {type_name}: {count}個 ({ratio:.1f}%)")
        
        # カテゴリ別統計
        category_stats = analysis_result.get("category_statistics", {})
        print("\nカテゴリ別統計:")
        for category_name, stats in category_stats.items():
            count = stats.get('count', 0)
            ratio = stats.get('importance_ratio', 0)
            print(f"  {category_name}: {count}個 ({ratio:.1f}%)")
        
        # AutoML効果
        automl_impact = analysis_result.get("automl_impact", {})
        print("\nAutoML効果:")
        print(f"  AutoML特徴量数: {automl_impact.get('automl_features', 0)}個")
        print(f"  AutoML重要度比率: {automl_impact.get('automl_importance_ratio', 0):.1f}%")
        
        # 上位特徴量
        top_features = analysis_result.get("top_features", [])
        print(f"\n上位特徴量 (上位{min(10, len(top_features))}個):")
        for i, feature in enumerate(top_features[:10]):
            print(f"  {i+1}. {feature['feature_name']} ({feature['feature_type']}) - {feature['importance']:.4f}")
        
        # 分析が正常に完了したかチェック
        if type_stats and category_stats:
            print("\n✅ AutoML特徴量分析が正常に完了しました")
            return True
        else:
            print("\n❌ AutoML特徴量分析が不完全です")
            return False
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_automl_enabled_training():
    """AutoML有効でのモデル学習をテスト"""
    print("\n" + "=" * 80)
    print("AutoML有効モデル学習テスト")
    print("=" * 80)
    
    try:
        # AutoML設定を作成
        automl_config = {
            "tsfresh": {
                "enabled": True,
                "feature_selection": True,
                "max_features": 50
            },
            "autofeat": {
                "enabled": True,
                "max_features": 30,
                "feateng_steps": 2
            }
        }
        
        print("AutoML設定:")
        print(f"  TSFresh: 有効, 最大特徴量数: {automl_config['tsfresh']['max_features']}")
        print(f"  AutoFeat: 有効, 最大特徴量数: {automl_config['autofeat']['max_features']}")
        
        # サンプルデータを作成
        np.random.seed(42)
        n_samples = 300  # AutoMLのため少し小さくする
        
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
            save_model=False,
            model_name=None
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
            if ts_features or af_features:
                analyzer = AutoMLFeatureAnalyzer()
                analysis_result = analyzer.analyze_feature_importance(feature_importance, top_n=20)
                
                print("\nAutoML特徴量分析結果:")
                if "error" not in analysis_result:
                    automl_impact = analysis_result.get("automl_impact", {})
                    print(f"AutoML特徴量比率: {automl_impact.get('automl_feature_ratio', 0):.1f}%")
                    print(f"AutoML重要度比率: {automl_impact.get('automl_importance_ratio', 0):.1f}%")
                    
                    return True
                else:
                    print(f"分析エラー: {analysis_result['error']}")
                    return False
            else:
                print("⚠️ AutoML特徴量が生成されませんでした")
                return False
        else:
            print("❌ 特徴量重要度が取得できませんでした")
            return False
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("AutoML特徴量分析修正テスト開始")
    print("=" * 80)
    
    # 1. 修正後の特徴量重要度保存テスト
    test1_result = test_fixed_feature_importance_save()
    
    # 2. 修正後のAutoML特徴量分析テスト
    test2_result = test_automl_feature_analysis_fix()
    
    # 3. AutoML有効でのモデル学習テスト（時間がかかるため最後）
    # test3_result = test_automl_enabled_training()
    
    print("\n" + "=" * 80)
    print("テスト結果:")
    print(f"1. 特徴量重要度保存: {'✅ 成功' if test1_result else '❌ 失敗'}")
    print(f"2. AutoML特徴量分析: {'✅ 成功' if test2_result else '❌ 失敗'}")
    # print(f"3. AutoML有効学習: {'✅ 成功' if test3_result else '❌ 失敗'}")
    
    if test1_result and test2_result:
        print("\n🎉 基本的な修正が完了しました！")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
    
    print("=" * 80)
