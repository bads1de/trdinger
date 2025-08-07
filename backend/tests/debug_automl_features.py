"""
AutoML特徴量分析のデバッグスクリプト

AutoML特徴量が正しく生成され、分析されているかを調査します。
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

from app.services.ml.model_manager import model_manager
from app.services.ml.feature_engineering.automl_feature_analyzer import AutoMLFeatureAnalyzer
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator

logger = logging.getLogger(__name__)


def debug_current_model_features():
    """現在のモデルの特徴量を詳細調査"""
    print("=" * 80)
    print("現在のモデルの特徴量詳細調査")
    print("=" * 80)
    
    try:
        # 最新のモデルを取得
        models = model_manager.list_models("*")
        if not models:
            print("❌ モデルファイルが見つかりません")
            return None
        
        latest_model = models[0]
        print(f"最新モデル: {latest_model['name']}")
        
        # モデルデータを読み込み
        model_data = model_manager.load_model(latest_model['path'])
        if not model_data or "metadata" not in model_data:
            print("❌ モデルメタデータが見つかりません")
            return None
        
        metadata = model_data["metadata"]
        feature_importance = metadata.get("feature_importance", {})
        
        print(f"\n特徴量重要度データ: {len(feature_importance)}個")
        
        if not feature_importance:
            print("❌ 特徴量重要度データがありません")
            return None
        
        # 特徴量名を分析
        print("\n特徴量名の分析:")
        print("-" * 60)
        
        feature_types = {
            "TS_": 0,  # TSFresh特徴量
            "AF_": 0,  # AutoFeat特徴量
            "manual": 0,  # 手動特徴量
        }
        
        for feature_name in feature_importance.keys():
            if feature_name.startswith("TS_"):
                feature_types["TS_"] += 1
            elif feature_name.startswith("AF_"):
                feature_types["AF_"] += 1
            else:
                feature_types["manual"] += 1
        
        print(f"TSFresh特徴量 (TS_): {feature_types['TS_']}個")
        print(f"AutoFeat特徴量 (AF_): {feature_types['AF_']}個")
        print(f"手動特徴量: {feature_types['manual']}個")
        
        # 各タイプの特徴量例を表示
        print("\n特徴量例:")
        print("-" * 60)
        
        ts_features = [name for name in feature_importance.keys() if name.startswith("TS_")]
        af_features = [name for name in feature_importance.keys() if name.startswith("AF_")]
        manual_features = [name for name in feature_importance.keys() if not (name.startswith("TS_") or name.startswith("AF_"))]
        
        if ts_features:
            print(f"TSFresh特徴量例: {ts_features[:5]}")
        if af_features:
            print(f"AutoFeat特徴量例: {af_features[:5]}")
        if manual_features:
            print(f"手動特徴量例: {manual_features[:5]}")
        
        return feature_importance
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_automl_feature_analyzer():
    """AutoMLFeatureAnalyzerの動作を調査"""
    print("\n" + "=" * 80)
    print("AutoMLFeatureAnalyzer動作調査")
    print("=" * 80)
    
    try:
        # 特徴量重要度データを取得
        feature_importance = debug_current_model_features()
        if not feature_importance:
            print("❌ 特徴量重要度データが取得できません")
            return
        
        # AutoMLFeatureAnalyzerで分析
        analyzer = AutoMLFeatureAnalyzer()
        analysis_result = analyzer.analyze_feature_importance(feature_importance, top_n=20)
        
        print("\nAutoML特徴量分析結果:")
        print("-" * 60)
        
        if "error" in analysis_result:
            print(f"❌ 分析エラー: {analysis_result['error']}")
            return
        
        # 分析結果の詳細表示
        print(f"総特徴量数: {analysis_result.get('total_features', 0)}")
        
        # タイプ別統計
        type_stats = analysis_result.get("type_statistics", {})
        print("\nタイプ別統計:")
        for type_name, stats in type_stats.items():
            print(f"  {type_name}: {stats.get('count', 0)}個 ({stats.get('importance_ratio', 0):.1f}%)")
        
        # カテゴリ別統計
        category_stats = analysis_result.get("category_statistics", {})
        print("\nカテゴリ別統計:")
        for category_name, stats in category_stats.items():
            print(f"  {category_name}: {stats.get('count', 0)}個 ({stats.get('importance_ratio', 0):.1f}%)")
        
        # AutoML効果
        automl_impact = analysis_result.get("automl_impact", {})
        print("\nAutoML効果:")
        print(f"  AutoML特徴量数: {automl_impact.get('automl_features', 0)}個")
        print(f"  AutoML重要度比率: {automl_impact.get('automl_importance_ratio', 0):.1f}%")
        
        # 上位特徴量
        top_features = analysis_result.get("top_features", [])
        print(f"\n上位特徴量 (上位{len(top_features)}個):")
        for i, feature in enumerate(top_features[:10]):
            print(f"  {i+1}. {feature['feature_name']} ({feature['feature_type']}) - {feature['importance']:.4f}")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_ml_orchestrator_analysis():
    """MLOrchestratorでのAutoML特徴量分析を調査"""
    print("\n" + "=" * 80)
    print("MLOrchestrator AutoML特徴量分析調査")
    print("=" * 80)
    
    try:
        # MLOrchestratorを初期化
        orchestrator = MLOrchestrator()
        
        # 特徴量重要度を取得
        feature_importance = orchestrator.get_feature_importance(100)
        print(f"MLOrchestratorから取得した特徴量重要度: {len(feature_importance)}個")
        
        if not feature_importance:
            print("❌ MLOrchestratorから特徴量重要度を取得できません")
            return
        
        # AutoMLFeatureAnalyzerで分析
        analyzer = AutoMLFeatureAnalyzer()
        analysis_result = analyzer.analyze_feature_importance(feature_importance, top_n=20)
        
        print("\nMLOrchestrator経由の分析結果:")
        print("-" * 60)
        
        if "error" in analysis_result:
            print(f"❌ 分析エラー: {analysis_result['error']}")
            return
        
        # 結果の要約表示
        print(f"総特徴量数: {analysis_result.get('total_features', 0)}")
        
        type_stats = analysis_result.get("type_statistics", {})
        category_stats = analysis_result.get("category_statistics", {})
        
        print("\nタイプ別統計:")
        for type_name, stats in type_stats.items():
            print(f"  {type_name}: {stats.get('count', 0)}個")
        
        print("\nカテゴリ別統計:")
        for category_name, stats in category_stats.items():
            print(f"  {category_name}: {stats.get('count', 0)}個")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_automl_feature_generation():
    """AutoML特徴量生成をテスト"""
    print("\n" + "=" * 80)
    print("AutoML特徴量生成テスト")
    print("=" * 80)
    
    try:
        from app.services.ml.ml_training_service import MLTrainingService
        
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
        
        # AutoMLを有効にしてMLTrainingServiceを初期化
        ml_service = MLTrainingService(trainer_type="ensemble", use_automl=True)
        
        print("AutoML有効でモデル学習を開始...")
        
        # モデルを学習（保存はしない）
        result = ml_service.train_model(
            training_data=df,
            save_model=False,
            model_name=None
        )
        
        print("✅ AutoML学習完了")
        
        # 特徴量重要度を確認
        if hasattr(ml_service.trainer, 'get_feature_importance'):
            feature_importance = ml_service.trainer.get_feature_importance(100)
            print(f"生成された特徴量数: {len(feature_importance)}")
            
            # AutoML特徴量の確認
            ts_features = [name for name in feature_importance.keys() if name.startswith("TS_")]
            af_features = [name for name in feature_importance.keys() if name.startswith("AF_")]
            
            print(f"TSFresh特徴量: {len(ts_features)}個")
            print(f"AutoFeat特徴量: {len(af_features)}個")
            
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
                else:
                    print(f"分析エラー: {analysis_result['error']}")
            else:
                print("⚠️ AutoML特徴量が生成されませんでした")
        
        return result
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("AutoML特徴量分析デバッグ開始")
    print("=" * 80)
    
    # 1. 現在のモデルの特徴量調査
    debug_current_model_features()
    
    # 2. AutoMLFeatureAnalyzerの動作調査
    debug_automl_feature_analyzer()
    
    # 3. MLOrchestratorでの分析調査
    debug_ml_orchestrator_analysis()
    
    # 4. AutoML特徴量生成テスト
    test_automl_feature_generation()
    
    print("\n" + "=" * 80)
    print("✅ デバッグ完了")
    print("=" * 80)
