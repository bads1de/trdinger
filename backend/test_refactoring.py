#!/usr/bin/env python3
"""
リファクタリング後の包括的テスト
"""

import sys
import os
from datetime import datetime
from unittest.mock import Mock

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """インポートテスト"""
    try:
        from app.services.backtest.backtest_service import BacktestService
        from app.services.backtest.backtest_data_service import BacktestDataService
        from app.services.backtest.data.data_retrieval_service import DataRetrievalService
        from app.services.backtest.data.data_conversion_service import DataConversionService
        from app.services.backtest.data.data_integration_service import DataIntegrationService
        print("✅ インポートテスト: 成功")
        return True
    except Exception as e:
        print(f"❌ インポートテスト: 失敗 - {e}")
        return False

def test_service_initialization():
    """サービス初期化テスト"""
    try:
        from app.services.backtest.backtest_service import BacktestService
        from app.services.backtest.backtest_data_service import BacktestDataService
        from app.services.backtest.data.data_retrieval_service import DataRetrievalService
        from app.services.backtest.data.data_conversion_service import DataConversionService
        from app.services.backtest.data.data_integration_service import DataIntegrationService
        
        # サービス初期化
        backtest_service = BacktestService()
        data_service = BacktestDataService()
        retrieval_service = DataRetrievalService()
        conversion_service = DataConversionService()
        integration_service = DataIntegrationService(retrieval_service)
        
        print("✅ サービス初期化テスト: 成功")
        return True
    except Exception as e:
        print(f"❌ サービス初期化テスト: 失敗 - {e}")
        return False

def test_data_conversion():
    """データ変換テスト"""
    try:
        from app.services.backtest.data.data_conversion_service import DataConversionService
        
        conversion_service = DataConversionService()
        
        # テスト用のOHLCVデータを作成
        test_ohlcv = [
            type('OHLCVData', (), {
                'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': 102.0, 'volume': 1000.0,
                'timestamp': datetime(2024, 1, 1)
            })(),
            type('OHLCVData', (), {
                'open': 102.0, 'high': 107.0, 'low': 98.0, 'close': 104.0, 'volume': 1100.0,
                'timestamp': datetime(2024, 1, 2)
            })()
        ]
        
        # DataFrame変換テスト
        df = conversion_service.convert_ohlcv_to_dataframe(test_ohlcv)
        
        # 結果検証
        assert len(df) == 2
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert df.iloc[0]['Open'] == 100.0
        assert df.iloc[1]['Close'] == 104.0
        
        print("✅ データ変換テスト: 成功")
        return True
    except Exception as e:
        print(f"❌ データ変換テスト: 失敗 - {e}")
        return False

def test_exception_classes():
    """例外クラステスト"""
    try:
        from app.services.backtest.data.data_retrieval_service import DataRetrievalError
        from app.services.backtest.data.data_conversion_service import DataConversionError
        from app.services.backtest.data.data_integration_service import DataIntegrationError
        
        # 例外クラスのテスト
        try:
            raise DataRetrievalError('テストエラー')
        except DataRetrievalError as e:
            assert str(e) == 'テストエラー'
        
        try:
            raise DataConversionError('テストエラー')
        except DataConversionError as e:
            assert str(e) == 'テストエラー'
        
        try:
            raise DataIntegrationError('テストエラー')
        except DataIntegrationError as e:
            assert str(e) == 'テストエラー'
        
        print("✅ 例外クラステスト: 成功")
        return True
    except Exception as e:
        print(f"❌ 例外クラステスト: 失敗 - {e}")
        return False

def test_facade_pattern():
    """Facadeパターンテスト"""
    try:
        from app.services.backtest.backtest_data_service import BacktestDataService
        import pandas as pd
        
        # BacktestDataServiceのFacadeパターンテスト
        data_service = BacktestDataService()
        
        # 空のDataFrameでのサマリーテスト
        empty_df = pd.DataFrame()
        summary = data_service.get_data_summary(empty_df)
        
        assert 'error' in summary
        assert summary['error'] == 'データがありません'
        
        print("✅ Facadeパターンテスト: 成功")
        return True
    except Exception as e:
        print(f"❌ Facadeパターンテスト: 失敗 - {e}")
        return False

def test_dependency_injection():
    """依存性注入テスト"""
    try:
        from app.services.backtest.backtest_service import BacktestService
        from app.services.backtest.backtest_data_service import BacktestDataService
        
        # モックサービスを作成
        mock_data_service = Mock()
        
        # 依存性注入テスト
        backtest_service = BacktestService(data_service=mock_data_service)
        assert backtest_service.data_service == mock_data_service
        
        print("✅ 依存性注入テスト: 成功")
        return True
    except Exception as e:
        print(f"❌ 依存性注入テスト: 失敗 - {e}")
        return False

def test_backward_compatibility():
    """後方互換性テスト"""
    try:
        from app.services.backtest.backtest_data_service import BacktestDataService
        
        # 後方互換性テスト
        data_service = BacktestDataService()
        
        # 古いインターフェースが維持されていることを確認
        assert hasattr(data_service, 'ohlcv_repo')
        assert hasattr(data_service, 'oi_repo')
        assert hasattr(data_service, 'fr_repo')
        assert hasattr(data_service, 'fear_greed_repo')
        
        # 新しいサービスが初期化されていることを確認
        assert hasattr(data_service, '_retrieval_service')
        assert hasattr(data_service, '_conversion_service')
        assert hasattr(data_service, '_integration_service')
        
        print("✅ 後方互換性テスト: 成功")
        return True
    except Exception as e:
        print(f"❌ 後方互換性テスト: 失敗 - {e}")
        return False

def main():
    """メインテスト実行"""
    print("🧪 リファクタリング後の包括的テスト開始")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_service_initialization,
        test_data_conversion,
        test_exception_classes,
        test_facade_pattern,
        test_dependency_injection,
        test_backward_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 テスト結果: {passed}件成功, {failed}件失敗")
    
    if failed == 0:
        print("🎉 すべてのテストが成功しました！")
        return True
    else:
        print("⚠️  一部のテストが失敗しました。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
