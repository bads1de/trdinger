"""
モデルメタデータの詳細調査スクリプト

性能指標が0.00%になる問題を調査し、メタデータの内容を確認します。
"""

import os
import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.ml.model_manager import model_manager

logger = logging.getLogger(__name__)


def investigate_model_metadata():
    """モデルメタデータの詳細調査"""
    try:
        print("=" * 80)
        print("モデルメタデータ詳細調査")
        print("=" * 80)
        
        # 1. 利用可能なモデルファイルを確認
        print("\n1. 利用可能なモデルファイル:")
        models = model_manager.list_models("*")
        if not models:
            print("  モデルファイルが見つかりません")
            return
        
        for i, model in enumerate(models):
            print(f"  {i+1}. {model['name']} ({model['size_mb']:.2f}MB)")
        
        # 2. 各モデルのメタデータを詳細調査
        print("\n2. 各モデルのメタデータ詳細調査:")
        
        for model_info in models:
            print(f"\n{'='*60}")
            print(f"モデル: {model_info['name']}")
            print(f"{'='*60}")
            
            try:
                # モデルデータを読み込み
                model_data = model_manager.load_model(model_info['path'])
                if not model_data:
                    print("  モデルデータの読み込みに失敗")
                    continue
                
                print(f"モデルデータのキー: {list(model_data.keys())}")
                
                # メタデータの詳細確認
                metadata = model_data.get("metadata", {})
                print(f"\nメタデータのキー: {list(metadata.keys())}")
                
                # 性能指標関連のキーを探す
                performance_keys = [
                    'accuracy', 'precision', 'recall', 'f1_score',
                    'auc_roc', 'auc_pr', 'balanced_accuracy', 'matthews_corrcoef',
                    'cohen_kappa', 'specificity', 'sensitivity', 'npv', 'ppv',
                    'log_loss', 'brier_score', 'test_metrics', 'train_metrics',
                    'evaluation_metrics', 'performance_metrics'
                ]
                
                print("\n性能指標の確認:")
                found_metrics = {}
                for key in performance_keys:
                    if key in metadata:
                        value = metadata[key]
                        found_metrics[key] = value
                        print(f"  {key}: {value} (型: {type(value)})")
                
                if not found_metrics:
                    print("  性能指標が見つかりません")
                
                # メタデータの全内容を表示（性能指標以外も含む）
                print("\nメタデータの全内容:")
                for key, value in metadata.items():
                    if key not in ['feature_importance']:  # 特徴量重要度は長いので除外
                        if isinstance(value, dict):
                            print(f"  {key}: {type(value)} (辞書)")
                            for sub_key, sub_value in value.items():
                                print(f"    {sub_key}: {sub_value}")
                        elif isinstance(value, list):
                            print(f"  {key}: {type(value)} (リスト, 長さ: {len(value)})")
                        else:
                            print(f"  {key}: {value} (型: {type(value)})")
                
                # モデル本体の確認
                model_obj = model_data.get("model")
                if model_obj:
                    print(f"\nモデル本体の型: {type(model_obj)}")
                    if hasattr(model_obj, '__dict__'):
                        print("モデル本体の属性:")
                        for attr_name in dir(model_obj):
                            if not attr_name.startswith('_'):
                                try:
                                    attr_value = getattr(model_obj, attr_name)
                                    if not callable(attr_value):
                                        print(f"  {attr_name}: {type(attr_value)}")
                                except:
                                    pass
                
            except Exception as e:
                print(f"  エラー: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("調査完了")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"調査実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    investigate_model_metadata()
