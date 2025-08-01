#!/usr/bin/env python3
"""
MLOrchestratorの特徴量重要度取得をデバッグするスクリプト
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def debug_ml_orchestrator():
    """MLOrchestratorの状態をデバッグ"""
    try:
        print("🔍 MLOrchestrator デバッグ開始")
        
        from backend.app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
        from backend.app.services.ml.model_manager import model_manager
        
        orchestrator = MLOrchestrator()
        
        # 1. MLOrchestratorの状態確認
        print(f"\n1. MLOrchestrator状態:")
        print(f"   is_model_loaded: {orchestrator.is_model_loaded}")
        print(f"   ml_training_service: {orchestrator.ml_training_service}")
        
        if orchestrator.ml_training_service:
            print(f"   trainer: {orchestrator.ml_training_service.trainer}")
            if hasattr(orchestrator.ml_training_service.trainer, 'is_trained'):
                print(f"   trainer.is_trained: {orchestrator.ml_training_service.trainer.is_trained}")
        
        # 2. 利用可能なモデルファイルを確認
        print(f"\n2. 利用可能なモデルファイル:")
        latest_model = model_manager.get_latest_model("*")
        if latest_model:
            print(f"   最新モデル: {latest_model}")
            
            # モデルデータを読み込み
            model_data = model_manager.load_model(latest_model)
            if model_data:
                print(f"   モデルデータ読み込み: 成功")
                if "metadata" in model_data:
                    metadata = model_data["metadata"]
                    print(f"   メタデータ: {list(metadata.keys())}")
                    
                    feature_importance = metadata.get("feature_importance", {})
                    print(f"   特徴量重要度: {len(feature_importance)}個")
                    
                    if feature_importance:
                        print("   特徴量重要度 (Top 3):")
                        sorted_importance = sorted(
                            feature_importance.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:3]
                        for feature, importance in sorted_importance:
                            print(f"     {feature}: {importance:.4f}")
                else:
                    print(f"   メタデータ: なし")
            else:
                print(f"   モデルデータ読み込み: 失敗")
        else:
            print(f"   最新モデル: なし")
        
        # 3. MLOrchestratorの特徴量重要度取得を実行
        print(f"\n3. MLOrchestrator特徴量重要度取得:")
        feature_importance = orchestrator.get_feature_importance(top_n=5)
        
        if feature_importance:
            print(f"   ✅ 取得成功: {len(feature_importance)}個")
            for feature, importance in feature_importance.items():
                print(f"     {feature}: {importance:.4f}")
        else:
            print(f"   ❌ 取得失敗")
        
        # 4. 全てのモデルファイルを確認
        print(f"\n4. 全モデルファイル:")
        all_models = model_manager.list_models()
        for model_info in all_models:
            print(f"   {model_info}")
        
        return {
            "is_model_loaded": orchestrator.is_model_loaded,
            "latest_model": latest_model,
            "feature_importance_count": len(feature_importance),
            "all_models_count": len(all_models)
        }
        
    except Exception as e:
        print(f"❌ デバッグエラー: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def debug_model_manager():
    """ModelManagerの詳細をデバッグ"""
    try:
        print(f"\n🗄️ ModelManager デバッグ開始")
        
        from backend.app.services.ml.model_manager import model_manager
        
        # モデルディレクトリの確認
        model_dir = model_manager.config.MODEL_SAVE_PATH
        print(f"モデルディレクトリ: {model_dir}")
        
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            print(f"ディレクトリ内ファイル数: {len(files)}")
            
            # .pklファイルのみ表示
            pkl_files = [f for f in files if f.endswith('.pkl')]
            print(f"PKLファイル数: {len(pkl_files)}")
            
            for pkl_file in pkl_files[-5:]:  # 最新5件
                file_path = os.path.join(model_dir, pkl_file)
                file_size = os.path.getsize(file_path)
                print(f"  {pkl_file} ({file_size/1024:.1f}KB)")
        else:
            print(f"モデルディレクトリが存在しません")
        
        # 最新モデルの詳細確認
        latest_model = model_manager.get_latest_model("*")
        if latest_model:
            print(f"\n最新モデル詳細: {latest_model}")
            
            try:
                model_data = model_manager.load_model(latest_model)
                if model_data:
                    print(f"  読み込み: 成功")
                    print(f"  キー: {list(model_data.keys())}")
                    
                    if "metadata" in model_data:
                        metadata = model_data["metadata"]
                        print(f"  メタデータキー: {list(metadata.keys())}")
                        
                        # 特徴量重要度の詳細
                        feature_importance = metadata.get("feature_importance", {})
                        if feature_importance:
                            print(f"  特徴量重要度型: {type(feature_importance)}")
                            print(f"  特徴量重要度サンプル: {dict(list(feature_importance.items())[:3])}")
                        else:
                            print(f"  特徴量重要度: なし")
                else:
                    print(f"  読み込み: 失敗")
            except Exception as e:
                print(f"  読み込みエラー: {e}")
        
    except Exception as e:
        print(f"❌ ModelManagerデバッグエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 MLOrchestrator & ModelManager デバッグ開始")
    
    # 1. ModelManagerデバッグ
    debug_model_manager()
    
    # 2. MLOrchestratorデバッグ
    result = debug_ml_orchestrator()
    
    # 結果サマリー
    print(f"\n{'='*50}")
    print("🏁 デバッグ結果サマリー")
    print(f"{'='*50}")
    
    if "error" not in result:
        print(f"✅ モデル読み込み状態: {result['is_model_loaded']}")
        print(f"✅ 最新モデル: {result['latest_model']}")
        print(f"✅ 特徴量重要度: {result['feature_importance_count']}個")
        print(f"✅ 全モデル数: {result['all_models_count']}個")
    else:
        print(f"❌ エラー: {result['error']}")
    
    print(f"\n🎯 デバッグ完了!")
