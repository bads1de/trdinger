"""
特徴量重要度の詳細調査スクリプト

BodySizeが100%になる問題を調査し、実際の重要度値を確認します。
"""

import os
import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.ml.model_manager import model_manager
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator

logger = logging.getLogger(__name__)


def investigate_feature_importance():
    """特徴量重要度の詳細調査"""
    try:
        print("=" * 80)
        print("特徴量重要度詳細調査")
        print("=" * 80)
        
        # 1. 利用可能なモデルファイルを確認
        print("\n1. 利用可能なモデルファイル:")
        models = model_manager.list_models("*")
        if not models:
            print("  モデルファイルが見つかりません")
            return
        
        for i, model in enumerate(models):
            print(f"  {i+1}. {model['name']} ({model['size_mb']:.2f}MB)")
        
        # 2. 各モデルの特徴量重要度を調査
        print("\n2. 各モデルの特徴量重要度調査:")
        
        for model_info in models:
            print(f"\n--- {model_info['name']} ---")
            
            try:
                # モデルデータを読み込み
                model_data = model_manager.load_model(model_info['path'])
                if not model_data:
                    print("  モデルデータの読み込みに失敗")
                    continue
                
                # メタデータから特徴量重要度を取得
                metadata = model_data.get("metadata", {})
                feature_importance = metadata.get("feature_importance", {})
                
                if not feature_importance:
                    print("  特徴量重要度データなし")
                    continue
                
                print(f"  特徴量数: {len(feature_importance)}")
                
                # 重要度を降順でソート
                sorted_importance = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # 統計情報を計算
                values = list(feature_importance.values())
                total_importance = sum(values)
                max_importance = max(values)
                min_importance = min(values)
                avg_importance = total_importance / len(values)
                
                print(f"  合計重要度: {total_importance:.6f}")
                print(f"  最大重要度: {max_importance:.6f}")
                print(f"  最小重要度: {min_importance:.6f}")
                print(f"  平均重要度: {avg_importance:.6f}")
                
                # 上位10個の特徴量重要度を表示（実際の値）
                print("\n  上位10個の特徴量重要度（実際の値）:")
                for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
                    percentage = (importance / max_importance) * 100
                    print(f"    {i:2d}. {feature}: {importance:.6f} ({percentage:.2f}%)")
                
                # BodySizeの詳細分析
                body_size_features = [item for item in sorted_importance if 'body' in item[0].lower() or 'Body' in item[0]]
                if body_size_features:
                    print("\n  BodySize関連特徴量:")
                    for feature, importance in body_size_features:
                        percentage = (importance / max_importance) * 100
                        absolute_percentage = (importance / total_importance) * 100
                        print(f"    {feature}: {importance:.6f} (相対: {percentage:.2f}%, 絶対: {absolute_percentage:.2f}%)")
                
                # 正規化の確認
                print(f"\n  正規化チェック:")
                print(f"    合計が1.0に近いか: {abs(total_importance - 1.0) < 0.001}")
                print(f"    最大値正規化での最大値: {(max_importance / max_importance) * 100:.2f}%")
                
            except Exception as e:
                print(f"  エラー: {e}")
        
        # 3. MLOrchestratorを使用した取得テスト
        print("\n3. MLOrchestratorを使用した特徴量重要度取得:")
        try:
            ml_orchestrator = MLOrchestrator()
            api_feature_importance = ml_orchestrator.get_feature_importance(top_n=20)
            
            if not api_feature_importance:
                print("  APIから特徴量重要度を取得できませんでした")
            else:
                print(f"  API経由で取得した特徴量重要度の数: {len(api_feature_importance)}")
                
                # 統計情報
                values = list(api_feature_importance.values())
                total = sum(values)
                max_val = max(values)
                
                print(f"  合計: {total:.6f}")
                print(f"  最大値: {max_val:.6f}")
                
                print("\n  API経由の特徴量重要度:")
                for i, (feature, importance) in enumerate(api_feature_importance.items(), 1):
                    percentage = (importance / max_val) * 100
                    absolute_percentage = (importance / total) * 100
                    print(f"    {i:2d}. {feature}: {importance:.6f} (相対: {percentage:.2f}%, 絶対: {absolute_percentage:.2f}%)")
        
        except Exception as e:
            print(f"  MLOrchestrator取得エラー: {e}")
        
        # 4. フロントエンド正規化の再現
        print("\n4. フロントエンド正規化の再現:")
        try:
            # 最新モデルから特徴量重要度を取得
            latest_model_path = model_manager.get_latest_model("*")
            if latest_model_path:
                model_data = model_manager.load_model(latest_model_path)
                metadata = model_data.get("metadata", {})
                feature_importance = metadata.get("feature_importance", {})
                
                if feature_importance:
                    # フロントエンドと同じ正規化を実行
                    values = list(feature_importance.values())
                    max_importance = max(values)
                    
                    print(f"  最大重要度: {max_importance:.6f}")
                    print("  フロントエンド正規化後の値:")
                    
                    sorted_importance = sorted(
                        feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    
                    for i, (feature, importance) in enumerate(sorted_importance, 1):
                        normalized_percent = (importance / max_importance) * 100
                        print(f"    {i:2d}. {feature}: {normalized_percent:.2f}%")
        
        except Exception as e:
            print(f"  フロントエンド正規化再現エラー: {e}")
        
        print("\n" + "=" * 80)
        print("調査完了")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"調査実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    investigate_feature_importance()
