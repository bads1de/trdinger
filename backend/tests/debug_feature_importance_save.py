"""
特徴量重要度保存のデバッグスクリプト

モデル保存時に特徴量重要度がメタデータに正しく保存されない問題を調査します。
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

logger = logging.getLogger(__name__)


def test_feature_importance_save():
    """特徴量重要度保存のテスト"""
    print("=" * 80)
    print("特徴量重要度保存テスト")
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
            save_model=False,  # まずは保存せずに学習のみ
            model_name=None
        )
        
        print("✅ モデル学習完了")
        
        # 学習後のトレーナーの状態を確認
        trainer = ml_service.trainer
        print(f"トレーナー型: {type(trainer)}")
        print(f"学習済み: {trainer.is_trained}")
        print(f"特徴量カラム数: {len(trainer.feature_columns) if trainer.feature_columns else 0}")
        
        # 特徴量重要度を直接取得してテスト
        print("\n特徴量重要度の直接取得テスト:")
        print("-" * 60)
        
        try:
            feature_importance = trainer.get_feature_importance(top_n=100)
            print(f"✅ 特徴量重要度取得成功: {len(feature_importance)}個")
            
            if feature_importance:
                # 上位5個を表示
                sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                for name, importance in sorted_importance:
                    print(f"  {name}: {importance:.4f}")
            else:
                print("❌ 特徴量重要度が空です")
                
        except Exception as e:
            print(f"❌ 特徴量重要度取得エラー: {e}")
            import traceback
            traceback.print_exc()
        
        # モデル保存をテスト
        print("\nモデル保存テスト:")
        print("-" * 60)
        
        test_model_name = "debug_feature_importance_test"
        
        try:
            # 保存前にメタデータを準備
            test_metadata = {
                "test_flag": True,
                "debug_timestamp": pd.Timestamp.now().isoformat(),
            }
            
            print("モデル保存を実行...")
            model_path = trainer.save_model(test_model_name, test_metadata)
            print(f"✅ モデル保存完了: {model_path}")
            
            # 保存されたモデルを読み込んで確認
            print("\n保存されたモデルの確認:")
            print("-" * 60)
            
            saved_model_data = model_manager.load_model(model_path)
            if saved_model_data and "metadata" in saved_model_data:
                metadata = saved_model_data["metadata"]
                print(f"メタデータのキー: {list(metadata.keys())}")
                
                if "feature_importance" in metadata:
                    feature_importance = metadata["feature_importance"]
                    print(f"✅ 特徴量重要度がメタデータに保存されています: {len(feature_importance)}個")
                    
                    # 上位5個を表示
                    if feature_importance:
                        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        for name, importance in sorted_importance:
                            print(f"  {name}: {importance:.4f}")
                else:
                    print("❌ 特徴量重要度がメタデータに保存されていません")
                    
                # テストフラグの確認
                if metadata.get("test_flag"):
                    print("✅ テストメタデータが正しく保存されています")
                else:
                    print("❌ テストメタデータが保存されていません")
            else:
                print("❌ 保存されたモデルのメタデータを読み込めません")
            
            # テストファイルを削除
            try:
                os.remove(model_path)
                print(f"✅ テストファイルを削除: {model_path}")
            except Exception as e:
                print(f"⚠️ テストファイル削除エラー: {e}")
                
        except Exception as e:
            print(f"❌ モデル保存エラー: {e}")
            import traceback
            traceback.print_exc()
        
        return result
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_existing_model_save():
    """既存モデルの保存処理をデバッグ"""
    print("\n" + "=" * 80)
    print("既存モデルの保存処理デバッグ")
    print("=" * 80)
    
    try:
        # 最新のモデルを取得
        models = model_manager.list_models("*")
        if not models:
            print("❌ 既存モデルが見つかりません")
            return
        
        latest_model = models[0]
        print(f"対象モデル: {latest_model['name']}")
        
        # モデルデータを読み込み
        model_data = model_manager.load_model(latest_model['path'])
        if not model_data:
            print("❌ モデルデータの読み込みに失敗")
            return
        
        model_obj = model_data.get("model")
        feature_columns = model_data.get("feature_columns", [])
        metadata = model_data.get("metadata", {})
        
        print(f"モデル型: {type(model_obj)}")
        print(f"特徴量カラム数: {len(feature_columns)}")
        print(f"既存メタデータキー: {list(metadata.keys())}")
        
        # モデルから直接特徴量重要度を取得
        if hasattr(model_obj, 'get_feature_importance'):
            try:
                print("\nモデルから直接特徴量重要度を取得:")
                feature_importance = model_obj.get_feature_importance()
                print(f"✅ 特徴量重要度取得: {len(feature_importance)}個")
                
                # 新しいメタデータを作成
                new_metadata = metadata.copy()
                new_metadata["feature_importance"] = feature_importance
                new_metadata["debug_updated"] = pd.Timestamp.now().isoformat()
                
                # 新しい名前で保存
                debug_model_name = f"debug_updated_{latest_model['name']}"
                
                print(f"\n特徴量重要度付きで再保存: {debug_model_name}")
                updated_model_path = model_manager.save_model(
                    model=model_obj,
                    model_name=debug_model_name,
                    metadata=new_metadata,
                    scaler=model_data.get("scaler"),
                    feature_columns=feature_columns,
                )
                
                print(f"✅ 更新モデル保存完了: {updated_model_path}")
                
                # 保存されたモデルを確認
                updated_model_data = model_manager.load_model(updated_model_path)
                if updated_model_data and "metadata" in updated_model_data:
                    updated_metadata = updated_model_data["metadata"]
                    if "feature_importance" in updated_metadata:
                        updated_feature_importance = updated_metadata["feature_importance"]
                        print(f"✅ 更新後の特徴量重要度: {len(updated_feature_importance)}個")
                        
                        # 上位5個を表示
                        sorted_importance = sorted(updated_feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        for name, importance in sorted_importance:
                            print(f"  {name}: {importance:.4f}")
                    else:
                        print("❌ 更新後も特徴量重要度が保存されていません")
                else:
                    print("❌ 更新されたモデルのメタデータを読み込めません")
                
                # テストファイルを削除
                try:
                    os.remove(updated_model_path)
                    print(f"✅ テストファイルを削除: {updated_model_path}")
                except Exception as e:
                    print(f"⚠️ テストファイル削除エラー: {e}")
                    
            except Exception as e:
                print(f"❌ 特徴量重要度取得エラー: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ モデルはget_feature_importanceメソッドを持っていません")
        
    except Exception as e:
        print(f"❌ デバッグ実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("特徴量重要度保存デバッグ開始")
    print("=" * 80)
    
    # 1. 新しいモデルでの特徴量重要度保存テスト
    test_feature_importance_save()
    
    # 2. 既存モデルの保存処理デバッグ
    debug_existing_model_save()
    
    print("\n" + "=" * 80)
    print("✅ デバッグ完了")
    print("=" * 80)
