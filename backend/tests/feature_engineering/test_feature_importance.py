#!/usr/bin/env python3
"""
特徴量重要度のテストスクリプト

現在のモデルから特徴量重要度を取得して表示します。
"""

import sys
import os
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.model_manager import model_manager
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_feature_importance():
    """特徴量重要度のテスト"""
    try:
        print("=" * 60)
        print("特徴量重要度テスト")
        print("=" * 60)

        # 1. 利用可能なモデルファイルを確認
        print("\n1. 利用可能なモデルファイル:")
        models = model_manager.list_models("*")
        if not models:
            print("  モデルファイルが見つかりません")
            return

        for i, model in enumerate(models[:5]):  # 最新5個を表示
            print(f"  {i+1}. {model['name']} ({model['size_mb']:.2f}MB)")

        # 2. 最新モデルを取得
        print("\n2. 最新モデルの確認:")
        latest_model_path = model_manager.get_latest_model("*")
        if not latest_model_path:
            print("  最新モデルが見つかりません")
            return

        print(f"  最新モデル: {os.path.basename(latest_model_path)}")

        # 3. モデルデータを読み込み
        print("\n3. モデルデータの読み込み:")
        model_data = model_manager.load_model(latest_model_path)
        if not model_data:
            print("  モデルデータの読み込みに失敗しました")
            return

        print(f"  モデルデータのキー: {list(model_data.keys())}")

        # 4. メタデータの確認
        print("\n4. メタデータの確認:")
        metadata = model_data.get("metadata", {})
        print(f"  メタデータのキー: {list(metadata.keys())}")

        # 5. 特徴量重要度の確認
        print("\n5. 特徴量重要度の確認:")
        feature_importance = metadata.get("feature_importance", {})

        print(f"  feature_importance の型: {type(feature_importance)}")
        print(f"  feature_importance の値: {feature_importance}")

        if not feature_importance:
            print("  特徴量重要度データが見つかりません")
        else:
            print(f"  特徴量重要度の数: {len(feature_importance)}")

            # 上位10個を表示
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]

            print("\n  上位10個の特徴量重要度:")
            for i, (feature, importance) in enumerate(sorted_importance, 1):
                print(f"    {i:2d}. {feature}: {importance:.6f}")

        # 6. MLOrchestratorを使用した取得テスト
        print("\n6. MLOrchestratorを使用した特徴量重要度取得:")
        try:
            ml_orchestrator = MLOrchestrator()
            api_feature_importance = ml_orchestrator.get_feature_importance(top_n=10)

            if not api_feature_importance:
                print("  APIから特徴量重要度を取得できませんでした")
            else:
                print(
                    f"  API経由で取得した特徴量重要度の数: {len(api_feature_importance)}"
                )
                print("\n  API経由の上位特徴量重要度:")
                for i, (feature, importance) in enumerate(
                    api_feature_importance.items(), 1
                ):
                    print(f"    {i:2d}. {feature}: {importance:.6f}")

        except Exception as e:
            print(f"  MLOrchestrator取得エラー: {e}")

        print("\n" + "=" * 60)
        print("テスト完了")
        print("=" * 60)

    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_feature_importance()
