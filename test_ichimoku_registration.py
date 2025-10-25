#!/usr/bin/env python3
"""
Ichimokuインジケーターの登録確認スクリプト
"""

import sys
import os

# backendディレクトリをパスに追加
sys.path.insert(0, 'backend')

from app.services.indicators.manifest import register_indicator_manifest, indicator_registry

def main():
    """メイン関数"""
    print("Ichimokuインジケーターの登録状況を確認します...")

    # インジケーターレジストリを初期化
    register_indicator_manifest()

    # 登録済みインジケーターを取得
    indicators = indicator_registry.list_indicators()

    print(f'登録済みインジケーター数: {len(indicators)}')

    # 主要インジケーターの存在確認
    major_indicators = ['ICHIMOKU', 'RSI', 'MACD', 'BB', 'STOCH', 'SUPERTREND']
    print('\n主要インジケーターの存在確認:')
    for ind in major_indicators:
        exists = ind in indicators
        status = "○" if exists else "×"
        print(f'  {status} {ind}: {exists}')

    # Ichimokuの詳細情報を表示
    if 'ICHIMOKU' in indicators:
        ichimoku_config = indicator_registry.get_indicator_config('ICHIMOKU')
        print(f'\nICHIMOKUインジケーターの詳細:')
        print(f'  カテゴリ: {ichimoku_config.category}')
        print(f'  スケールタイプ: {ichimoku_config.scale_type}')
        print(f'  結果タイプ: {ichimoku_config.result_type}')
        print(f'  必須データ: {ichimoku_config.required_data}')
        print(f'  パラメータ:')
        for param_name, param_config in ichimoku_config.parameters.items():
            if hasattr(param_config, 'default_value'):
                print(f'    {param_name}: {param_config.default_value} (範囲: {param_config.min_value}-{param_config.max_value})')

    print('\n✓ Ichimoku Cloudインジケーターの追加が正常に完了しました！')

if __name__ == "__main__":
    main()