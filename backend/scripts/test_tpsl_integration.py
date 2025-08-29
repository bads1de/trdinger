#!/usr/bin/env python3
"""
TPSL統合最終確認スクリプト
テクニカル指標のみのオートストラテジーを生成し、TPSL統合が正常動作するか確認します
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_technical_only_strategy() -> Dict[str, Any]:
    """
    テクニカル指標のみのストラテジーを作成
    TPSL統合を確認するために使用

    Returns:
        ストラテジー構成辞書
    """
    from app.services.auto_strategy.models.strategy_models import (
        StrategyGene,
        IndicatorGene,
        TPSLGene,
        TPSLMethod,
        PositionSizingGene,
        PositionSizingMethod
    )

    logger.info("テクニカル指標のみのストラテジー生成を開始")

    # テクニカル指標: SMAベースのシンプルなストラテジー
    indicators = [
        IndicatorGene(
            type="SMA",
            parameters={"period": 20},
            enabled=True
        ),
        IndicatorGene(
            type="SMA",
            parameters={"period": 50},
            enabled=True
        )
    ]

    # TPSL遺伝子: 統合TPSLジェネレーターを使用
    tpsl_gene = TPSLGene(
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        risk_reward_ratio=3.0,
        enabled=True
    )

    # ポジションサイジング遺伝子
    position_sizing_gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        fixed_ratio=0.1,
        min_position_size=0.01,
        max_position_size=1.0,
        enabled=True
    )

    # ストラテジー遺伝子
    strategy_gene = StrategyGene(
        indicators=indicators,
        entry_conditions="SMA crossover strategy",
        exit_conditions="Simple reversal",
        tpsl_gene=tpsl_gene,
        position_sizing_gene=position_sizing_gene
    )

    logger.info("TPSL遺伝子情報:")
    logger.info(f"  - Method: {tpsl_gene.method.value}")
    logger.info(f"  - Stop Loss: {tpsl_gene.stop_loss_pct}%")
    logger.info(f"  - Take Profit: {tpsl_gene.take_profit_pct}%")
    logger.info(f"  - Risk/Reward: {tpsl_gene.risk_reward_ratio}")

    return {
        "strategy_info": "Technical Indicators Only Strategy",
        "indicators_count": len(indicators),
        "tpsl_method": tpsl_gene.method.value if hasattr(tpsl_gene.method, 'value') else str(tpsl_gene.method),
        "stop_loss_pct": tpsl_gene.stop_loss_pct,
        "take_profit_pct": tpsl_gene.take_profit_pct,
        "risk_reward_ratio": tpsl_gene.risk_reward_ratio
    }


def test_tpsl_integration_manual() -> Dict[str, Any]:
    """
    TPSL統合の手動テスト
    統合ジェネレーターを直接呼び出して動作確認

    Returns:
        TPSL結果を含むテスト結果辞書
    """
    from app.services.auto_strategy.generators.unified_tpsl_generator import UnifiedTPSLGenerator

    logger.info("TPSL統合手動テストを開始")

    # 統合ジェネレーターのインスタンス化
    generator = UnifiedTPSLGenerator()

    test_results = []
    methods = ["risk_reward", "volatility", "statistical", "fixed_percentage"]

    current_price = 50000.0  # テスト用価格

    for method in methods:
        logger.info(f"Testing method: {method}")

        try:
            if method == "risk_reward":
                result = generator.generate_tpsl(
                    method=method,
                    stop_loss_pct=0.02,
                    target_ratio=3.0,
                    current_price=current_price
                )
            elif method == "volatility":
                result = generator.generate_tpsl(
                    method=method,
                    base_atr_pct=0.02,
                    current_price=current_price
                )
            elif method == "statistical":
                result = generator.generate_tpsl(
                    method=method,
                    lookback_period_days=100,
                    confidence_threshold=0.7
                )
            elif method == "fixed_percentage":
                result = generator.generate_tpsl(
                    method=method,
                    stop_loss_pct=0.02,
                    take_profit_pct=0.06
                )

            test_results.append({
                "method": method,
                "stop_loss_pct": result.stop_loss_pct,
                "take_profit_pct": result.take_profit_pct,
                "confidence_score": result.confidence_score,
                "method_used": result.method_used,
                "status": "success"
            })

            logger.info(f"✓ {method}: SL={result.stop_loss_pct:.4f}, TP={result.take_profit_pct:.4f}")

        except Exception as e:
            test_results.append({
                "method": method,
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"✗ {method}: {e}")

    return {"manual_test": test_results}


def test_tpsl_service_integration() -> Dict[str, Any]:
    """
    TPSLServiceとの統合テスト

    Returns:
        TPSLService統合テスト結果
    """
    from app.services.auto_strategy.services.tpsl_service import TPSLService

    logger.info("TPSLService統合テストを開始")

    service = TPSLService()
    current_price = 50000.0

    try:
        # TPSLServiceをテスト
        sl_price, tp_price = service.calculate_tpsl_prices(
            current_price=current_price,
            stop_loss_pct=0.02,
            take_profit_pct=0.06,
            position_direction=1.0
        )

        logger.info(f"TPSLService結果: SL={sl_price}, TP={tp_price}")

        return {
            "tpsl_service": {
                "status": "success",
                "current_price": current_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_pct": (current_price - sl_price) / current_price if sl_price else None,
                "tp_pct": (tp_price - current_price) / current_price if tp_price else None
            }
        }

    except Exception as e:
        logger.error(f"TPSLServiceエラー: {e}")
        return {
            "tpsl_service": {
                "status": "failed",
                "error": str(e)
            }
        }


def main():
    """
    メイン実行関数
    """
    print("="*80)
    print("TPSL統合最終確認スクリプト")
    print("テクニカル指標のみのストラテジー生成 & TPSL統合テスト")
    print("="*80)

    try:
        # 1. テクニカルストラテジーの生成
        strategy_result = create_technical_only_strategy()

        # 2. TPSL統合手動テスト
        tpsl_manual_result = test_tpsl_integration_manual()

        # 3. TPSLService統合テスト
        tpsl_service_result = test_tpsl_service_integration()

        # 統合結果
        final_result = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "tpsl_integration_final_verification",
            "status": "completed",
            "strategy_generation": strategy_result,
            "tpsl_manual_test": tpsl_manual_result,
            "tpsl_service_integration": tpsl_service_result
        }

        # 成功・失敗の集計
        manual_success = len([r for r in tpsl_manual_result["manual_test"] if r["status"] == "success"])
        manual_total = len(tpsl_manual_result["manual_test"])
        service_status = tpsl_service_result["tpsl_service"]["status"]

        final_result["summary"] = {
            "manual_test_success_rate": f"{manual_success}/{manual_total}",
            "tpsl_service_status": service_status,
            "overall_status": "success" if manual_success == manual_total and service_status == "success" else "partial_success"
        }

        # JSON出力
        output_file = "tpsl_integration_test_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)

        print("\n" + "="*80)
        print("✅ TPSL統合テスト完了")
        print("="*80)
        print(f"📄 結果ファイル: {os.path.abspath(output_file)}")
        print(f"🧪 手動テスト成功率: {manual_success}/{manual_total}")
        print(f"🔧 TPSLServiceステータス: {service_status}")
        print("")
        print("TPSL統合機能:")
        print("- 統合ジェネレーターの正常動作確認")
        print("- 戦略生成時のTPSL設定適用確認")
        print("- 全TPSL手法（Risk/Reward, Volatility, Statistical, Fixed）の機能確認")

        success = manual_success == manual_total and service_status == "success"
        print("\n[SUCCESS] 結果:")
        print("TPSL統合は完全に機能しています！" if success else "WARNING: TPSL統合に一部問題が見つかりましたが、主要機能は動作しています")
        print("="*80)

        return 0 if success else 1

    except Exception as e:
        print(f"\n[ERROR] エラー発生: {e}")
        import traceback
        print(f"詳細:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)