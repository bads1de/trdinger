#!/usr/bin/env python
"""テストスクリプト: BBANDSとBBのYAML設定ロードテスト"""

import os
import sys

# backend/appのpathを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.services.auto_strategy.utils.common_utils import YamlIndicatorUtils

    print("YAML設定をロード中...")
    yaml_config = YamlIndicatorUtils.load_yaml_config_for_indicators()

    print(f"YAML設定ロード成功。インジケータ数: {len(yaml_config.get('indicators', {}))}")

    # BBとBBANDSが存在するか確認
    indicators = yaml_config.get('indicators', {})

    bb_found = 'BB' in indicators
    bbands_found = 'BBANDS' in indicators

    print(f"BB設定存在: {bb_found}")
    print(f"BBANDS設定存在: {bbands_found}")

    if bb_found:
        bb_config = indicators['BB']
        print("BB設定内容:")
        for key, value in bb_config.items():
            print(f"  {key}: {value}")

    if bbands_found:
        bbands_config = indicators['BBANDS']
        print("BBANDS設定内容:")
        for key, value in bbands_config.items():
            print(f"  {key}: {value}")

    # 両方あるか確認
    if bb_found and bbands_found:
        print("✅ BBとBBANDS両方の設定が正しくロードされました")
    else:
        print("❌ 一部の設定が見つかりません")
        sys.exit(1)

    print("テスト完了")

except Exception as e:
    print(f"❌ エラー発生: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)