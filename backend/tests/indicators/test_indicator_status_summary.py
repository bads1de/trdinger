"""
インジケータステータス要約テスト
"""
import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class TestIndicatorStatusSummary:
    """インジケータステータスの要約テスト"""

    def test_indicator_service_initialization(self, technical_indicator_service):
        """インジケータサービスの初期化テスト"""
        assert technical_indicator_service is not None
        assert hasattr(technical_indicator_service, 'registry')
        assert hasattr(technical_indicator_service, 'calculate_indicator')

    def test_supported_indicators_list(self, technical_indicator_service):
        """サポートされているインジケータ一覧の取得テスト"""
        supported_indicators = technical_indicator_service.get_supported_indicators()

        assert isinstance(supported_indicators, dict)
        assert len(supported_indicators) > 0

        # 主要なインジケータが含まれていることを確認
        key_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'BB', 'STOCH', 'ADX', 'CCI', 'ATR', 'MFI', 'STC']
        for indicator in key_indicators:
            assert indicator in supported_indicators, f"{indicator}がサポートリストに含まれていません"

        print(f"サポートされているインジケータ数: {len(supported_indicators)}")
        print("主要インジケータ一覧:", list(supported_indicators.keys())[:20])  # 最初の20個を表示

    def test_indicator_config_loading(self):
        """インジケータ設定の読み込みテスト"""
        from app.services.indicators.config import indicator_registry

        assert indicator_registry is not None

        # レジストリにインジケータが登録されていることを確認
        indicator_configs = list(indicator_registry._configs.keys())
        assert len(indicator_configs) > 0

        print(f"登録されているインジケータ設定数: {len(indicator_configs)}")
        print("登録インジケータ一覧:", indicator_configs[:20])  # 最初の20個を表示

    def test_basic_indicator_calculations(self, sample_ohlcv_data, technical_indicator_service):
        """基本的なインジケータ計算テスト"""
        df = sample_ohlcv_data.copy()

        # テストする主要インジケータ
        test_indicators = [
            ("RSI", {"length": 14}),
            ("SMA", {"length": 20}),
            ("EMA", {"length": 20}),
            ("MACD", {"fast": 12, "slow": 26, "signal": 9}),
            ("BB", {"period": 20, "std": 2.0}),
            ("STOCH", {"fastk_period": 5, "d_length": 3, "slowd_period": 3}),
            ("ADX", {"length": 14}),
            ("CCI", {"period": 14}),
            ("ATR", {"period": 14}),
            ("MFI", {"length": 14}),
            ("STC", {"length": 10}),
        ]

        successful_calculations = []
        failed_calculations = []

        for indicator_name, params in test_indicators:
            try:
                result = technical_indicator_service.calculate_indicator(df, indicator_name, params)

                if result is not None:
                    successful_calculations.append(indicator_name)
                    print(f"✅ {indicator_name}: 計算成功")
                else:
                    failed_calculations.append(indicator_name)
                    print(f"❌ {indicator_name}: 結果がNone")

            except Exception as e:
                failed_calculations.append(indicator_name)
                print(f"❌ {indicator_name}: 計算エラー - {str(e)}")

        # 結果のサマリー
        print("\n=== テスト結果サマリー ===")
        print(f"✅ 成功: {len(successful_calculations)}個")
        print(f"❌ 失敗: {len(failed_calculations)}個")
        print(f"総計: {len(test_indicators)}個")

        if successful_calculations:
            print(f"成功したインジケータ: {successful_calculations}")

        if failed_calculations:
            print(f"失敗したインジケータ: {failed_calculations}")

        # 少なくとも主要なインジケータは成功すべき
        assert len(successful_calculations) > 0, "主要インジケータのどれも計算できませんでした"

        # STCは特に重要なので確認
        assert "STC" in successful_calculations, "STCインジケータの計算に失敗しました"

    def test_indicator_registry_consistency(self):
        """インジケータレジストリの一貫性テスト"""
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        # レジストリとサービスの整合性を確認
        service = TechnicalIndicatorService()
        service_indicators = set(service.get_supported_indicators().keys())
        registry_indicators = set(indicator_registry._configs.keys())

        # 両方に含まれるインジケータ
        common_indicators = service_indicators & registry_indicators
        print(f"レジストリとサービスで共通のインジケータ数: {len(common_indicators)}")

        # サービスのみのインジケータ（pandas-ta直接使用）
        service_only = service_indicators - registry_indicators
        print(f"サービスのみのインジケータ数: {len(service_only)}")

        # レジストリのみのインジケータ
        registry_only = registry_indicators - service_indicators
        print(f"レジストリのみのインジケータ数: {len(registry_only)}")

        assert len(common_indicators) > 0, "共通のインジケータがありません"