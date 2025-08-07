"""
モデルメタデータの詳細デバッグスクリプト

現在のモデルのメタデータ内容を詳細に調査し、
特徴量重要度が保存されていない原因を特定します。
"""

import os
import sys
import logging
import json
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.ml.model_manager import model_manager

logger = logging.getLogger(__name__)


def debug_model_metadata_detailed():
    """モデルメタデータの詳細調査"""
    print("=" * 80)
    print("モデルメタデータ詳細調査")
    print("=" * 80)
    
    try:
        # 利用可能なモデルファイルを確認
        models = model_manager.list_models("*")
        if not models:
            print("❌ モデルファイルが見つかりません")
            return
        
        print(f"利用可能なモデル数: {len(models)}")
        
        for i, model_info in enumerate(models[:3]):  # 最新3つのモデルを調査
            print(f"\n{'='*60}")
            print(f"モデル {i+1}: {model_info['name']}")
            print(f"{'='*60}")
            
            try:
                # モデルデータを読み込み
                model_data = model_manager.load_model(model_info['path'])
                if not model_data:
                    print("  ❌ モデルデータの読み込みに失敗")
                    continue
                
                print(f"モデルデータのキー: {list(model_data.keys())}")
                
                # メタデータの詳細確認
                if "metadata" in model_data:
                    metadata = model_data["metadata"]
                    print(f"\nメタデータのキー: {list(metadata.keys())}")
                    
                    # 各メタデータキーの詳細表示
                    for key, value in metadata.items():
                        if key == "feature_importance":
                            if isinstance(value, dict):
                                print(f"  {key}: 辞書 ({len(value)}個の特徴量)")
                                if value:
                                    # 上位5個の特徴量を表示
                                    sorted_features = sorted(value.items(), key=lambda x: x[1], reverse=True)[:5]
                                    for feat_name, feat_importance in sorted_features:
                                        print(f"    {feat_name}: {feat_importance:.4f}")
                                else:
                                    print("    (空の辞書)")
                            else:
                                print(f"  {key}: {type(value)} - {value}")
                        elif isinstance(value, dict):
                            print(f"  {key}: 辞書 ({len(value)}個のキー)")
                            if len(value) <= 10:  # 小さい辞書は内容を表示
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, (int, float)):
                                        print(f"    {sub_key}: {sub_value:.4f}")
                                    else:
                                        print(f"    {sub_key}: {type(sub_value)}")
                        elif isinstance(value, list):
                            print(f"  {key}: リスト (長さ: {len(value)})")
                        elif isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {type(value)} - {str(value)[:100]}")
                else:
                    print("  ❌ メタデータが見つかりません")
                
                # モデル本体の確認
                if "model" in model_data:
                    model_obj = model_data["model"]
                    print(f"\nモデル本体の型: {type(model_obj)}")
                    
                    # モデルが特徴量重要度を持っているかチェック
                    if hasattr(model_obj, 'feature_importance'):
                        print("  ✅ モデルはfeature_importanceメソッドを持っています")
                        try:
                            importance = model_obj.feature_importance()
                            print(f"  特徴量重要度の形状: {importance.shape if hasattr(importance, 'shape') else len(importance)}")
                        except Exception as e:
                            print(f"  ❌ feature_importance呼び出しエラー: {e}")
                    elif hasattr(model_obj, 'get_feature_importance'):
                        print("  ✅ モデルはget_feature_importanceメソッドを持っています")
                        try:
                            importance = model_obj.get_feature_importance()
                            print(f"  特徴量重要度: {len(importance)}個")
                        except Exception as e:
                            print(f"  ❌ get_feature_importance呼び出しエラー: {e}")
                    else:
                        print("  ❌ モデルは特徴量重要度メソッドを持っていません")
                
                # 特徴量カラムの確認
                if "feature_columns" in model_data:
                    feature_columns = model_data["feature_columns"]
                    print(f"\n特徴量カラム: {len(feature_columns)}個")
                    if feature_columns:
                        print(f"  例: {feature_columns[:5]}")
                else:
                    print("\n❌ 特徴量カラム情報が見つかりません")
                
            except Exception as e:
                print(f"  ❌ モデル調査エラー: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 調査実行エラー: {e}")
        import traceback
        traceback.print_exc()


def test_feature_importance_extraction():
    """特徴量重要度の抽出テスト"""
    print("\n" + "=" * 80)
    print("特徴量重要度抽出テスト")
    print("=" * 80)
    
    try:
        # 最新のモデルを取得
        models = model_manager.list_models("*")
        if not models:
            print("❌ モデルファイルが見つかりません")
            return
        
        latest_model = models[0]
        print(f"テスト対象モデル: {latest_model['name']}")
        
        # モデルデータを読み込み
        model_data = model_manager.load_model(latest_model['path'])
        if not model_data:
            print("❌ モデルデータの読み込みに失敗")
            return
        
        # モデル本体を取得
        model_obj = model_data.get("model")
        feature_columns = model_data.get("feature_columns", [])
        
        print(f"モデル型: {type(model_obj)}")
        print(f"特徴量カラム数: {len(feature_columns)}")
        
        if not model_obj:
            print("❌ モデル本体が見つかりません")
            return
        
        if not feature_columns:
            print("❌ 特徴量カラムが見つかりません")
            return
        
        # 特徴量重要度の抽出を試行
        feature_importance = {}
        
        # 方法1: feature_importanceメソッド（LightGBM）
        if hasattr(model_obj, 'feature_importance'):
            try:
                print("\n方法1: feature_importanceメソッドを試行")
                importance_scores = model_obj.feature_importance(importance_type="gain")
                print(f"  重要度スコア形状: {importance_scores.shape if hasattr(importance_scores, 'shape') else len(importance_scores)}")
                
                if len(importance_scores) == len(feature_columns):
                    feature_importance = dict(zip(feature_columns, importance_scores))
                    print(f"  ✅ 特徴量重要度を取得: {len(feature_importance)}個")
                    
                    # 上位5個を表示
                    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    for name, importance in sorted_importance:
                        print(f"    {name}: {importance:.4f}")
                else:
                    print(f"  ❌ 重要度スコア数({len(importance_scores)})と特徴量数({len(feature_columns)})が一致しません")
                    
            except Exception as e:
                print(f"  ❌ feature_importanceエラー: {e}")
        
        # 方法2: get_feature_importanceメソッド
        if hasattr(model_obj, 'get_feature_importance'):
            try:
                print("\n方法2: get_feature_importanceメソッドを試行")
                importance_dict = model_obj.get_feature_importance()
                print(f"  ✅ 特徴量重要度を取得: {len(importance_dict)}個")
                
                # 上位5個を表示
                if importance_dict:
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                    for name, importance in sorted_importance:
                        print(f"    {name}: {importance:.4f}")
                
                feature_importance.update(importance_dict)
                
            except Exception as e:
                print(f"  ❌ get_feature_importanceエラー: {e}")
        
        # 方法3: feature_importances_属性（sklearn系）
        if hasattr(model_obj, 'feature_importances_'):
            try:
                print("\n方法3: feature_importances_属性を試行")
                importance_scores = model_obj.feature_importances_
                print(f"  重要度スコア形状: {importance_scores.shape if hasattr(importance_scores, 'shape') else len(importance_scores)}")
                
                if len(importance_scores) == len(feature_columns):
                    sklearn_importance = dict(zip(feature_columns, importance_scores))
                    print(f"  ✅ 特徴量重要度を取得: {len(sklearn_importance)}個")
                    
                    # 上位5個を表示
                    sorted_importance = sorted(sklearn_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    for name, importance in sorted_importance:
                        print(f"    {name}: {importance:.4f}")
                    
                    feature_importance.update(sklearn_importance)
                else:
                    print(f"  ❌ 重要度スコア数({len(importance_scores)})と特徴量数({len(feature_columns)})が一致しません")
                    
            except Exception as e:
                print(f"  ❌ feature_importances_エラー: {e}")
        
        if feature_importance:
            print(f"\n✅ 合計で{len(feature_importance)}個の特徴量重要度を取得しました")
        else:
            print("\n❌ 特徴量重要度を取得できませんでした")
        
        return feature_importance
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    print("モデルメタデータ詳細デバッグ開始")
    print("=" * 80)
    
    # 1. モデルメタデータの詳細調査
    debug_model_metadata_detailed()
    
    # 2. 特徴量重要度の抽出テスト
    test_feature_importance_extraction()
    
    print("\n" + "=" * 80)
    print("✅ デバッグ完了")
    print("=" * 80)
