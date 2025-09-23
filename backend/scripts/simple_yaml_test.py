#!/usr/bin/env python3
"""
YAML設定ファイル読み込みテスト
"""

import os
from app.services.auto_strategy.core.condition_evolver import YamlIndicatorUtils

def main():
    """メイン実行関数"""
    try:
        # 絶対パスでYAMLファイルのパスを指定
        yaml_path = os.path.join(os.path.dirname(__file__), "..", "app", "services", "auto_strategy", "config", "technical_indicators_config.yaml")
        print(f"YAMLファイルパス: {yaml_path}")
        print(f"ファイルが存在するか: {os.path.exists(yaml_path)}")

        # YAMLファイル読み込み
        yaml_utils = YamlIndicatorUtils(yaml_path)

        # 利用可能な指標一覧を取得
        indicators = yaml_utils.get_available_indicators()
        print(f"利用可能な指標数: {len(indicators)}")
        print(f"指標一覧: {indicators[:10]}...")  # 最初の10個を表示

        # 指標タイプ別の分類を取得
        types = yaml_utils.get_indicator_types()
        print("\n指標タイプ別分類:")
        for type_name, indicator_list in types.items():
            print(f"  {type_name}: {len(indicator_list)}個")

        # RSI指標の詳細情報を取得
        if "RSI" in indicators:
            rsi_info = yaml_utils.get_indicator_info("RSI")
            print(f"\nRSI指標情報: {rsi_info}")

        print("\n✅ YAMLファイル読み込みテスト成功")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()