"""
ベイジアン最適化のプロファイル保存シナリオテスト

実際の使用ケースに近い形でベイジアン最適化のプロファイル保存をテストします。
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import asyncio
import logging
from datetime import datetime

from database.connection import SessionLocal
from database.repositories.bayesian_optimization_repository import (
    BayesianOptimizationRepository,
)
from app.api.bayesian_optimization import (
    MLOptimizationRequest,
    optimize_ml_hyperparameters,
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_scenario_1_save_with_profile():
    """シナリオ1: プロファイル保存ありでベイジアン最適化"""
    print("\n=== シナリオ1: プロファイル保存あり ===")

    request = MLOptimizationRequest(
        model_type="LightGBM",
        n_calls=10,
        save_as_profile=True,
        profile_name="lightgbm_optimized_profile",
        profile_description="LightGBMの最適化されたハイパーパラメータプロファイル",
    )

    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)

    try:
        print(f"実行前のレコード数: {len(repo.get_all_results())}")

        # ベイジアン最適化実行
        result = await optimize_ml_hyperparameters(request, db)

        print(f"実行結果: {result['success']}")
        if result["success"]:
            api_result = result["result"]
            print(f"ベストスコア: {api_result['best_score']:.4f}")
            print(f"評価回数: {api_result['total_evaluations']}")
            print(f"最適化時間: {api_result['optimization_time']:.2f}秒")

            if "saved_profile_id" in api_result:
                profile_id = api_result["saved_profile_id"]
                print(f"保存されたプロファイルID: {profile_id}")

                # 保存されたプロファイルを確認
                saved_profile = repo.get_by_id(profile_id)
                if saved_profile:
                    print(f"✅ プロファイル保存確認:")
                    print(f"   名前: {saved_profile.profile_name}")
                    print(f"   モデルタイプ: {saved_profile.model_type}")
                    print(f"   対象モデルタイプ: {saved_profile.target_model_type}")
                    print(f"   ベストパラメータ: {saved_profile.best_params}")
                    return profile_id
                else:
                    print("❌ 保存されたプロファイルが見つかりません")
            else:
                print("❌ saved_profile_idが結果に含まれていません")

        return None

    except Exception as e:
        print(f"❌ シナリオ1エラー: {e}")
        return None
    finally:
        db.close()


async def test_scenario_2_no_save():
    """シナリオ2: プロファイル保存なしでベイジアン最適化"""
    print("\n=== シナリオ2: プロファイル保存なし ===")

    request = MLOptimizationRequest(
        model_type="XGBoost", n_calls=8, save_as_profile=False  # 保存しない
    )

    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)

    try:
        initial_count = len(repo.get_all_results())
        print(f"実行前のレコード数: {initial_count}")

        # ベイジアン最適化実行
        result = await optimize_ml_hyperparameters(request, db)

        print(f"実行結果: {result['success']}")
        if result["success"]:
            api_result = result["result"]
            print(f"ベストスコア: {api_result['best_score']:.4f}")
            print(f"評価回数: {api_result['total_evaluations']}")

            # saved_profile_idが含まれていないことを確認
            if "saved_profile_id" not in api_result:
                print("✅ saved_profile_idが含まれていません（正常）")
            else:
                print("❌ saved_profile_idが含まれています（異常）")

            # レコード数が変わっていないことを確認
            final_count = len(repo.get_all_results())
            print(f"実行後のレコード数: {final_count}")

            if final_count == initial_count:
                print("✅ レコード数が変わっていません（正常）")
            else:
                print("❌ レコード数が変わっています（異常）")

    except Exception as e:
        print(f"❌ シナリオ2エラー: {e}")
    finally:
        db.close()


async def test_scenario_3_profile_name_missing():
    """シナリオ3: save_as_profile=Trueだがprofile_nameが未指定"""
    print("\n=== シナリオ3: プロファイル名未指定 ===")

    request = MLOptimizationRequest(
        model_type="LightGBM",
        n_calls=5,
        save_as_profile=True,
        # profile_nameを指定しない
        profile_description="プロファイル名未指定テスト",
    )

    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)

    try:
        initial_count = len(repo.get_all_results())
        print(f"実行前のレコード数: {initial_count}")

        # ベイジアン最適化実行
        result = await optimize_ml_hyperparameters(request, db)

        print(f"実行結果: {result['success']}")
        if result["success"]:
            api_result = result["result"]
            print(f"ベストスコア: {api_result['best_score']:.4f}")

            # saved_profile_idが含まれていないことを確認
            if "saved_profile_id" not in api_result:
                print("✅ profile_name未指定時は保存されません（正常）")
            else:
                print("❌ profile_name未指定でも保存されました（異常）")

            # レコード数が変わっていないことを確認
            final_count = len(repo.get_all_results())
            print(f"実行後のレコード数: {final_count}")

            if final_count == initial_count:
                print("✅ レコード数が変わっていません（正常）")
            else:
                print("❌ レコード数が変わっています（異常）")

    except Exception as e:
        print(f"❌ シナリオ3エラー: {e}")
    finally:
        db.close()


def test_scenario_4_profile_retrieval(profile_id):
    """シナリオ4: 保存されたプロファイルの取得テスト"""
    print("\n=== シナリオ4: プロファイル取得テスト ===")

    if not profile_id:
        print("プロファイルIDが指定されていません")
        return

    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)

    try:
        # ID指定で取得
        profile = repo.get_by_id(profile_id)
        if profile:
            print(f"✅ ID指定取得成功: {profile.profile_name}")
        else:
            print("❌ ID指定取得失敗")
            return

        # プロファイル名で取得
        profile_by_name = repo.get_by_profile_name(profile.profile_name)
        if profile_by_name:
            print(f"✅ 名前指定取得成功: {profile_by_name.profile_name}")
        else:
            print("❌ 名前指定取得失敗")

        # モデルタイプ別取得
        profiles_by_model = repo.get_profiles_by_model_type(profile.target_model_type)
        if profiles_by_model and any(p.id == profile_id for p in profiles_by_model):
            print(f"✅ モデルタイプ別取得成功: {len(profiles_by_model)}件")
        else:
            print("❌ モデルタイプ別取得失敗")

        # デフォルトプロファイル設定テスト
        success = repo.set_default_profile(profile_id, profile.target_model_type)
        if success:
            print("✅ デフォルトプロファイル設定成功")

            # デフォルトプロファイル取得
            default_profile = repo.get_default_profile(profile.target_model_type)
            if default_profile and default_profile.id == profile_id:
                print("✅ デフォルトプロファイル取得成功")
            else:
                print("❌ デフォルトプロファイル取得失敗")
        else:
            print("❌ デフォルトプロファイル設定失敗")

    except Exception as e:
        print(f"❌ シナリオ4エラー: {e}")
    finally:
        db.close()


def cleanup_test_profiles():
    """テスト用プロファイルのクリーンアップ"""
    print("\n=== テストプロファイルクリーンアップ ===")

    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)

    try:
        # テスト用プロファイルを検索
        test_profiles = []
        all_profiles = repo.get_all_results()

        for profile in all_profiles:
            if any(
                keyword in profile.profile_name.lower()
                for keyword in ["test", "lightgbm_optimized", "scenario"]
            ):
                test_profiles.append(profile)

        print(f"削除対象プロファイル: {len(test_profiles)}件")

        for profile in test_profiles:
            db.delete(profile)
            print(f"削除: {profile.profile_name}")

        db.commit()
        print("✅ クリーンアップ完了")

    except Exception as e:
        print(f"❌ クリーンアップエラー: {e}")
        db.rollback()
    finally:
        db.close()


async def main():
    """メイン実行関数"""
    print("ベイジアン最適化プロファイル保存シナリオテスト開始")
    print("=" * 60)

    # シナリオ1: プロファイル保存あり
    profile_id = await test_scenario_1_save_with_profile()

    # シナリオ2: プロファイル保存なし
    await test_scenario_2_no_save()

    # シナリオ3: プロファイル名未指定
    await test_scenario_3_profile_name_missing()

    # シナリオ4: プロファイル取得テスト
    if profile_id:
        test_scenario_4_profile_retrieval(profile_id)

    # 最終DB状況確認
    print("\n=== 最終DB状況確認 ===")
    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)
    try:
        all_results = repo.get_all_results()
        print(f"総レコード数: {len(all_results)}")
        for result in all_results:
            print(
                f"  ID: {result.id}, Name: {result.profile_name}, Model: {result.model_type}, Default: {result.is_default}"
            )
    finally:
        db.close()

    # クリーンアップ
    cleanup_test_profiles()

    print("\n" + "=" * 60)
    print("シナリオテスト完了")


if __name__ == "__main__":
    asyncio.run(main())
