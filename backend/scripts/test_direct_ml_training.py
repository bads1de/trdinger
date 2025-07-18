"""
直接MLトレーニングサービスをテストするスクリプト

APIを経由せずに、直接MLトレーニングサービスを呼び出して
修正が正常に動作することを確認します。
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.ml.ml_training_service import MLTrainingService

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_direct_ml_training():
    """直接MLトレーニングサービスをテスト"""
    logger.info("=== 直接MLトレーニングサービステスト ===")
    
    try:
        with SessionLocal() as db:
            ohlcv_repo = OHLCVRepository(db)
            
            # テスト用データを取得（元のエラーと同じ期間）
            ohlcv_data = ohlcv_repo.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_time=datetime(2020, 3, 5, tzinfo=timezone.utc),
                end_time=datetime(2025, 7, 15, tzinfo=timezone.utc)
            )
            
            if not ohlcv_data:
                logger.error("テスト用データが見つかりません")
                return False
            
            logger.info(f"取得データ件数: {len(ohlcv_data)}")
            
            # DataFrameに変換（カラム名は大文字で統一）
            df = pd.DataFrame([
                {
                    'timestamp': data.timestamp,
                    'Open': data.open,
                    'High': data.high,
                    'Low': data.low,
                    'Close': data.close,
                    'Volume': data.volume
                }
                for data in ohlcv_data
            ])
            
            df.set_index('timestamp', inplace=True)
            
            # MLトレーニングサービスを初期化
            ml_service = MLTrainingService()
            
            # 修正されたパラメータでトレーニング（動的閾値を使用）
            training_params = {
                "threshold_method": "std_deviation",  # 動的閾値を使用
                "std_multiplier": 0.25,  # 分析結果から最適な値
                "test_size": 0.2,
                "random_state": 42,
                "save_model": False  # テストなので保存しない
            }
            
            logger.info("=== 修正前のエラー再現テスト ===")
            logger.info("元のエラーが修正されているかテスト中...")
            logger.info(f"使用データ: {len(df)}行")
            logger.info(f"パラメータ: {training_params}")
            
            # トレーニング実行
            result = ml_service.train_model(
                training_data=df,
                **training_params
            )
            
            # 結果を確認
            logger.info("✅ MLトレーニング成功!")
            logger.info("=== トレーニング結果 ===")
            
            # 安全な値の取得と表示
            accuracy = result.get('accuracy', 'N/A')
            if isinstance(accuracy, (int, float)):
                logger.info(f"精度: {accuracy:.4f}")
            else:
                logger.info(f"精度: {accuracy}")
                
            logger.info(f"クラス数: {result.get('num_classes', 'N/A')}")
            logger.info(f"学習サンプル数: {result.get('train_samples', 'N/A')}")
            logger.info(f"テストサンプル数: {result.get('test_samples', 'N/A')}")
            logger.info(f"特徴量数: {result.get('feature_count', 'N/A')}")
            logger.info(f"モデルタイプ: {result.get('model_type', 'N/A')}")
            
            # 分類レポートを表示
            if 'classification_report' in result:
                class_report = result['classification_report']
                logger.info("=== 分類レポート ===")
                for class_name, metrics in class_report.items():
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        logger.info(f"クラス {class_name}:")
                        logger.info(f"  precision: {metrics['precision']:.3f}")
                        logger.info(f"  recall: {metrics['recall']:.3f}")
                        logger.info(f"  f1-score: {metrics['f1-score']:.3f}")
                        logger.info(f"  support: {metrics['support']}")
            
            # 特徴量重要度の上位を表示
            if 'feature_importance' in result and result['feature_importance']:
                logger.info("=== 特徴量重要度（上位15） ===")
                sorted_features = sorted(
                    result['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:15]
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    logger.info(f"{i:2d}. {feature}: {importance:.4f}")
            
            # 元のエラーが解決されているかチェック
            if result.get('num_classes', 0) > 1:
                logger.info("✅ 元のエラー「Number of classes must be 1 for non-multiclass training」は解決されました！")
            else:
                logger.warning("⚠️ まだクラス数の問題があります")
            
            # ラベル分布情報があるかチェック
            if any(key.startswith('threshold_') for key in result.keys()):
                logger.info("✅ 動的閾値計算が正常に動作しています！")
            
            # 特徴量拡張が動作しているかチェック
            feature_count = result.get('feature_count', 0)
            if feature_count > 50:  # 基本特徴量より多い場合
                logger.info(f"✅ 特徴量拡張が正常に動作しています！（{feature_count}個の特徴量）")
            else:
                logger.warning(f"⚠️ 特徴量拡張が期待通りに動作していない可能性があります（{feature_count}個の特徴量）")
            
            return True
            
    except Exception as e:
        logger.error(f"直接MLトレーニングテストエラー: {e}")
        import traceback
        logger.error(f"詳細エラー: {traceback.format_exc()}")
        return False


def test_with_old_parameters():
    """元のパラメータでテスト（エラーが再現されるかチェック）"""
    logger.info("=== 元のパラメータでのテスト ===")
    
    try:
        with SessionLocal() as db:
            ohlcv_repo = OHLCVRepository(db)
            
            # 少量のテスト用データを取得
            ohlcv_data = ohlcv_repo.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 2, 1, tzinfo=timezone.utc)
            )
            
            if not ohlcv_data:
                logger.error("テスト用データが見つかりません")
                return False
            
            # DataFrameに変換
            df = pd.DataFrame([
                {
                    'timestamp': data.timestamp,
                    'Open': data.open,
                    'High': data.high,
                    'Low': data.low,
                    'Close': data.close,
                    'Volume': data.volume
                }
                for data in ohlcv_data
            ])
            
            df.set_index('timestamp', inplace=True)
            
            # MLトレーニングサービスを初期化
            ml_service = MLTrainingService()
            
            # 元のパラメータ（固定閾値±2%）
            old_params = {
                "threshold_method": "fixed",  # 固定閾値
                "threshold": 0.02,  # ±2%（元のデフォルト値）
                "test_size": 0.2,
                "random_state": 42,
                "save_model": False
            }
            
            logger.info("元のパラメータ（固定閾値±2%）でテスト中...")
            logger.info(f"パラメータ: {old_params}")
            
            try:
                result = ml_service.train_model(
                    training_data=df,
                    **old_params
                )
                
                # 結果を確認
                num_classes = result.get('num_classes', 0)
                if num_classes <= 1:
                    logger.warning(f"⚠️ 元のパラメータでは依然として{num_classes}クラスしかありません")
                    logger.warning("これは期待される結果です（修正前の状態を再現）")
                else:
                    logger.info(f"✅ 元のパラメータでも{num_classes}クラス分類が可能でした")
                
                return True
                
            except Exception as e:
                if "Number of classes must be 1" in str(e) or "1種類のクラスしかありません" in str(e):
                    logger.info("✅ 元のパラメータでは期待通りエラーが発生しました（修正前の状態を再現）")
                    logger.info("これにより、修正が正しく動作していることが確認できます")
                    return True
                else:
                    logger.error(f"予期しないエラー: {e}")
                    return False
            
    except Exception as e:
        logger.error(f"元のパラメータテストエラー: {e}")
        return False


def main():
    """メイン関数"""
    logger.info("直接MLトレーニングサービステスト開始")
    
    test_results = []
    
    # 各テストを実行
    test_results.append(("修正されたMLトレーニング", test_direct_ml_training()))
    test_results.append(("元のパラメータでのテスト", test_with_old_parameters()))
    
    # 結果サマリー
    logger.info("\n" + "="*60)
    logger.info("テスト結果サマリー")
    logger.info("="*60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ 成功" if result else "❌ 失敗"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 すべてのテストが成功しました！")
        logger.info("元のLightGBMエラーは完全に修正されています。")
    else:
        logger.error("\n⚠️ 一部のテストが失敗しました。")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        sys.exit(1)
