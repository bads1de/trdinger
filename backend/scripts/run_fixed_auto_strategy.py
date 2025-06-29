"""
修正版オートストラテジーの実行テスト

適切な期間で修正版のオートストラテジーを実行し、取引が発生することを確認します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from datetime import datetime
import time
import logging

# ログレベルを設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_fixed_auto_strategy():
    """修正版オートストラテジーの実行"""
    print("🚀 修正版オートストラテジー実行開始")
    print(f"実行時刻: {datetime.now()}")
    
    try:
        # AutoStrategyServiceを初期化
        print("AutoStrategyServiceを初期化中...")
        service = AutoStrategyService()
        
        # 適切なGA設定（小規模だが実用的）
        ga_config = GAConfig(
            population_size=5,   # 小さな個体数
            generations=3,       # 少ない世代数
            crossover_rate=0.8,
            mutation_rate=0.2,
            allowed_indicators=["RSI", "SMA", "CCI", "MACD"]  # 基本的な指標
        )
        
        # 適切な期間のバックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-11-01",  # 1ヶ月間
            "end_date": "2024-11-30",
            "initial_capital": 100000.0,
            "commission_rate": 0.001
        }
        
        print("修正版GA実行を開始...")
        print(f"期間: {backtest_config['start_date']} - {backtest_config['end_date']}")
        print(f"個体数: {ga_config.population_size}, 世代数: {ga_config.generations}")
        
        experiment_id = service.start_strategy_generation(
            experiment_name=f"FIXED_AUTO_STRATEGY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ga_config=ga_config,
            backtest_config=backtest_config
        )
        
        print(f"実験ID: {experiment_id}")
        
        # 進捗監視
        print("進捗監視中...")
        max_wait = 300  # 5分間待機
        start_time = time.time()
        last_generation = 0
        
        while time.time() - start_time < max_wait:
            progress = service.get_experiment_progress(experiment_id)
            if progress:
                current_gen = getattr(progress, 'current_generation', 0)
                total_gen = getattr(progress, 'total_generations', ga_config.generations)
                
                if current_gen != last_generation:
                    print(f"  世代 {current_gen}/{total_gen} 完了")
                    last_generation = current_gen
                
                if progress.status == "completed":
                    print("✅ 修正版GA実行完了")
                    
                    # 結果を取得
                    result = service.get_experiment_result(experiment_id)
                    if result:
                        print(f"\n📊 実行結果:")
                        print(f"  最高フィットネス: {result['best_fitness']:.4f}")
                        print(f"  実行時間: {result['execution_time']:.2f}秒")
                        print(f"  完了世代数: {result['generations_completed']}")
                        
                        # 戦略の詳細を確認
                        best_strategy = result['best_strategy']
                        print(f"\n🏆 最優秀戦略:")
                        print(f"  指標数: {len(best_strategy.indicators)}")
                        print(f"  エントリー条件数: {len(best_strategy.entry_conditions)}")
                        print(f"  エグジット条件数: {len(best_strategy.exit_conditions)}")
                        
                        # 指標の詳細
                        print(f"\n🔧 使用指標:")
                        for i, indicator in enumerate(best_strategy.indicators, 1):
                            print(f"    {i}. {indicator.type} - {indicator.parameters}")
                        
                        # 条件の詳細
                        print(f"\n📈 エントリー条件:")
                        for i, condition in enumerate(best_strategy.entry_conditions, 1):
                            print(f"    {i}. {condition.left_operand} {condition.operator} {condition.right_operand}")
                        
                        print(f"\n📉 エグジット条件:")
                        for i, condition in enumerate(best_strategy.exit_conditions, 1):
                            print(f"    {i}. {condition.left_operand} {condition.operator} {condition.right_operand}")
                        
                        # バックテスト結果の確認
                        check_backtest_results(experiment_id)
                        
                        return True
                    break
                elif progress.status == "failed":
                    error_msg = getattr(progress, 'error_message', '不明なエラー')
                    print(f"❌ GA実行失敗: {error_msg}")
                    return False
            
            time.sleep(10)  # 10秒間隔で確認
        else:
            print("⏰ タイムアウト: GA実行が完了しませんでした")
            
            # 最終状態を確認
            final_progress = service.get_experiment_progress(experiment_id)
            if final_progress:
                print(f"最終状態: {final_progress.status}")
                if hasattr(final_progress, 'error_message') and final_progress.error_message:
                    print(f"エラー: {final_progress.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ 修正版GA実行エラー: {e}")
        logger.exception("修正版GA実行中にエラーが発生")
        return False


def check_backtest_results(experiment_id):
    """バックテスト結果の詳細確認"""
    print(f"\n📊 バックテスト結果確認:")
    
    try:
        from database.connection import SessionLocal
        from database.repositories.backtest_result_repository import BacktestResultRepository
        
        db = SessionLocal()
        try:
            backtest_repo = BacktestResultRepository(db)
            
            # 最近のバックテスト結果を取得
            recent_results = backtest_repo.get_recent_results(limit=3)
            
            for result in recent_results:
                if result.strategy_name and "FIXED_AUTO_STRATEGY" in result.strategy_name:
                    print(f"\n  📈 戦略: {result.strategy_name}")
                    print(f"    実行日時: {result.created_at}")
                    
                    # パフォーマンス指標を確認
                    if result.performance_metrics:
                        metrics = result.performance_metrics
                        total_trades = metrics.get('total_trades', 0)
                        total_return = metrics.get('total_return', 0)
                        win_rate = metrics.get('win_rate', 0)
                        max_drawdown = metrics.get('max_drawdown', 0)
                        
                        print(f"    📊 パフォーマンス:")
                        print(f"      総取引数: {total_trades}")
                        print(f"      総リターン: {total_return:.2%}")
                        print(f"      勝率: {win_rate:.2%}" if win_rate and not str(win_rate) == 'nan' else "      勝率: N/A")
                        print(f"      最大ドローダウン: {max_drawdown:.2%}")
                        
                        # 取引回数の確認
                        if total_trades > 0:
                            print(f"    ✅ 取引が実行されました！")
                            
                            # 取引履歴の確認
                            if result.trade_history:
                                print(f"    📋 取引履歴 (最初の3件):")
                                for i, trade in enumerate(result.trade_history[:3], 1):
                                    entry_time = trade.get('entry_time', 'N/A')
                                    exit_time = trade.get('exit_time', 'N/A')
                                    pnl = trade.get('pnl', 0)
                                    print(f"      {i}. {entry_time} - {exit_time}: {pnl:.2f}")
                        else:
                            print(f"    ❌ 取引回数0: 条件が厳しすぎる可能性")
                    
                    break
        
        finally:
            db.close()
        
    except Exception as e:
        print(f"❌ バックテスト結果確認エラー: {e}")


def main():
    """メイン実行関数"""
    success = run_fixed_auto_strategy()
    
    if success:
        print(f"\n🎉 修正版オートストラテジー実行成功")
        print(f"取引回数0問題が解決されているかを確認してください。")
    else:
        print(f"\n❌ 修正版オートストラテジー実行失敗")
    
    return success


if __name__ == "__main__":
    main()
