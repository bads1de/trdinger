"""
シンプルなパラメータ適用テスト
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.ml.lightgbm_trainer import LightGBMTrainer


def create_simple_test_data():
    """シンプルなテストデータを作成"""
    np.random.seed(42)

    # 500行のOHLCVデータを作成
    dates = pd.date_range(start="2023-01-01", periods=500, freq="h")

    # 価格データ（トレンドを持たせる）
    base_price = 45000
    price_trend = np.cumsum(np.random.normal(0, 50, 500))

    data = {
        "timestamp": dates,
        "Open": base_price + price_trend + np.random.normal(0, 100, 500),
        "High": base_price + price_trend + np.random.normal(200, 100, 500),
        "Low": base_price + price_trend + np.random.normal(-200, 100, 500),
        "Close": base_price + price_trend + np.random.normal(0, 100, 500),
        "Volume": np.random.uniform(100, 1000, 500),
    }

    df = pd.DataFrame(data)

    # timestampをインデックスに設定
    df.set_index("timestamp", inplace=True)

    # HighとLowを調整
    for i in range(500):
        df.iloc[i, df.columns.get_loc("High")] = max(
            df.iloc[i, df.columns.get_loc("Open")],
            df.iloc[i, df.columns.get_loc("Close")],
        ) + abs(np.random.normal(0, 50))
        df.iloc[i, df.columns.get_loc("Low")] = min(
            df.iloc[i, df.columns.get_loc("Open")],
            df.iloc[i, df.columns.get_loc("Close")],
        ) - abs(np.random.normal(0, 50))

    return df


def test_parameter_changes():
    """パラメータ変更が結果に影響するかテスト"""
    print("🧪 パラメータ変更テストを開始...")

    # テストデータを作成
    training_data = create_simple_test_data()
    print(f"テストデータ作成完了: {len(training_data)}行")

    # LightGBMTrainerを作成
    trainer = LightGBMTrainer()

    try:
        # テスト1: デフォルトパラメータ
        print("\n📊 テスト1: デフォルトパラメータ")
        result1 = trainer.train_model(
            training_data=training_data,
            save_model=False,
            test_size=0.2,
            random_state=42,
        )
        accuracy1 = result1.get("accuracy", 0)
        print(f"デフォルト結果: 精度={accuracy1}")

        # テスト2: learning_rate変更
        print("\n📊 テスト2: learning_rate=0.01")
        trainer2 = LightGBMTrainer()  # 新しいインスタンス
        result2 = trainer2.train_model(
            training_data=training_data,
            save_model=False,
            test_size=0.2,
            random_state=42,
            learning_rate=0.01,
        )
        accuracy2 = result2.get("accuracy", 0)
        print(f"learning_rate=0.01結果: 精度={accuracy2}")

        # テスト3: num_leaves変更
        print("\n📊 テスト3: num_leaves=10")
        trainer3 = LightGBMTrainer()  # 新しいインスタンス
        result3 = trainer3.train_model(
            training_data=training_data,
            save_model=False,
            test_size=0.2,
            random_state=42,
            num_leaves=10,
        )
        accuracy3 = result3.get("accuracy", 0)
        print(f"num_leaves=10結果: 精度={accuracy3}")

        # 結果比較
        print("\n📈 結果比較:")
        print(f"  デフォルト: {accuracy1}")
        print(f"  learning_rate=0.01: {accuracy2}")
        print(f"  num_leaves=10: {accuracy3}")

        # 精度が異なるかチェック
        accuracies = [accuracy1, accuracy2, accuracy3]
        unique_count = len(set(accuracies))

        if unique_count > 1:
            print("✅ パラメータが正しく適用されています（精度が変化）")
            return True
        else:
            print("❌ パラメータが適用されていません（精度が同じ）")
            print(f"全ての精度: {accuracies}")
            return False

    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔬 シンプルパラメータテスト")
    print("=" * 50)

    success = test_parameter_changes()

    print("\n" + "=" * 50)
    if success:
        print("🎉 テスト成功: パラメータが正しく適用されています")
    else:
        print("⚠️ テスト失敗: パラメータが適用されていない可能性があります")
