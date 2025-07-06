"""
提供されたJSONパラメータでの修正検証テスト

実際に問題が報告されたJSONパラメータを使用して、
修正が正しく動作することを確認します。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_provided_json_parameters():
    """
    提供されたJSONパラメータでの修正検証
    """
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene

        # 提供されたJSONパラメータを再現
        strategy_gene_data = {
            "id": "",
            "indicators": [
                {
                    "type": "ADX",
                    "parameters": {
                        "period": 25
                    },
                    "enabled": True
                }
            ],
            "entry_conditions": [
                {
                    "left_operand": "ADX",
                    "operator": "<",
                    "right_operand": 30
                }
            ],
            "long_entry_conditions": [],
            "short_entry_conditions": [],
            "exit_conditions": [
                {
                    "left_operand": "ADX",
                    "operator": ">",
                    "right_operand": 70
                }
            ],
            "risk_management": {
                "position_size": 0.1
            },
            "tpsl_gene": {
                "method": "adaptive",
                "stop_loss_pct": 0.0226,
                "take_profit_pct": 0.084,
                "risk_reward_ratio": 2.713,
                "base_stop_loss": 0.0446,
                "atr_multiplier_sl": 2.579,
                "atr_multiplier_tp": 5.074,
                "atr_period": 14,
                "lookback_period": 100,
                "confidence_threshold": 0.7,
                "method_weights": {
                    "fixed": 0.25,
                    "risk_reward": 0.35,
                    "volatility": 0.25,
                    "statistical": 0.15
                },
                "enabled": True,
                "priority": 1
            },
            "metadata": {
                "generated_by": "GeneEncoder_decode",
                "source": "fallback_individual",
                "indicators_count": 1,
                "decoded_from_length": 24,
                "tpsl_gene_included": True
            }
        }

        print("📋 提供されたJSONパラメータでの検証を開始...")

        # 戦略遺伝子を復元
        gene = StrategyGene.from_dict(strategy_gene_data)

        print(f"✅ 戦略遺伝子復元成功: ID={gene.id or 'auto-generated'}")
        print(f"✅ 指標数: {len(gene.indicators)}")
        print(f"✅ エントリー条件数: {len(gene.entry_conditions)}")
        print(f"✅ エグジット条件数: {len(gene.exit_conditions)}")

        # 修正前の問題を確認
        print(f"📊 long_entry_conditions: {len(gene.long_entry_conditions)}個")
        print(f"📊 short_entry_conditions: {len(gene.short_entry_conditions)}個")

        # 修正後の動作を確認
        long_conditions = gene.get_effective_long_conditions()
        short_conditions = gene.get_effective_short_conditions()

        print(f"🔧 修正後のロング条件数: {len(long_conditions)}")
        print(f"🔧 修正後のショート条件数: {len(short_conditions)}")

        # 条件の詳細を表示
        if long_conditions:
            for i, cond in enumerate(long_conditions):
                print(f"  ロング条件{i+1}: {cond.left_operand} {cond.operator} {cond.right_operand}")

        if short_conditions:
            for i, cond in enumerate(short_conditions):
                print(f"  ショート条件{i+1}: {cond.left_operand} {cond.operator} {cond.right_operand}")

        # 期待される結果の検証
        assert len(long_conditions) > 0, "❌ ロング条件が取得できませんでした"
        assert len(short_conditions) > 0, "❌ ショート条件が取得できませんでした"
        
        # 条件の内容が正しいことを確認
        assert long_conditions[0].left_operand == "ADX", "❌ ロング条件の内容が正しくありません"
        assert long_conditions[0].operator == "<", "❌ ロング条件の演算子が正しくありません"
        assert long_conditions[0].right_operand == 30, "❌ ロング条件の右オペランドが正しくありません"

        assert short_conditions[0].left_operand == "ADX", "❌ ショート条件の内容が正しくありません"
        assert short_conditions[0].operator == "<", "❌ ショート条件の演算子が正しくありません"
        assert short_conditions[0].right_operand == 30, "❌ ショート条件の右オペランドが正しくありません"

        print("✅ 提供されたJSONパラメータでの修正検証成功！")
        print("✅ entry_conditionsが適切にlong/short条件に変換されました")
        
        return True

    except Exception as e:
        print(f"❌ 提供されたJSONパラメータでの検証エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_factory_with_provided_json():
    """
    StrategyFactoryでの戦略クラス作成テスト
    """
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory

        # 提供されたJSONパラメータを再現
        strategy_gene_data = {
            "id": "test_provided_json",
            "indicators": [
                {
                    "type": "ADX",
                    "parameters": {"period": 25},
                    "enabled": True
                }
            ],
            "entry_conditions": [
                {
                    "left_operand": "ADX",
                    "operator": "<",
                    "right_operand": 30
                }
            ],
            "long_entry_conditions": [],
            "short_entry_conditions": [],
            "exit_conditions": [
                {
                    "left_operand": "ADX",
                    "operator": ">",
                    "right_operand": 70
                }
            ],
            "risk_management": {"position_size": 0.1},
        }

        print("📋 StrategyFactoryでの戦略クラス作成テスト...")

        # 戦略遺伝子を復元
        gene = StrategyGene.from_dict(strategy_gene_data)

        # StrategyFactoryで戦略クラスを作成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene)

        print(f"✅ 戦略クラス作成成功: {strategy_class.__name__}")

        # 戦略クラスに必要なメソッドが存在することを確認
        required_methods = [
            '_check_long_entry_conditions',
            '_check_short_entry_conditions',
            '_check_exit_conditions'
        ]

        for method_name in required_methods:
            assert hasattr(strategy_class, method_name), f"❌ {method_name}メソッドが存在しません"
            print(f"✅ {method_name}メソッド存在確認")

        print("✅ StrategyFactoryでの戦略クラス作成テスト成功！")
        return True

    except Exception as e:
        print(f"❌ StrategyFactoryでの戦略クラス作成テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_config_preparation():
    """
    バックテスト設定の準備テスト
    """
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene

        # 提供されたJSONパラメータを再現
        strategy_gene_data = {
            "id": "test_backtest_config",
            "indicators": [
                {
                    "type": "ADX",
                    "parameters": {"period": 25},
                    "enabled": True
                }
            ],
            "entry_conditions": [
                {
                    "left_operand": "ADX",
                    "operator": "<",
                    "right_operand": 30
                }
            ],
            "long_entry_conditions": [],
            "short_entry_conditions": [],
            "exit_conditions": [
                {
                    "left_operand": "ADX",
                    "operator": ">",
                    "right_operand": 70
                }
            ],
            "risk_management": {"position_size": 0.1},
        }

        print("📋 バックテスト設定の準備テスト...")

        # 戦略遺伝子を復元
        gene = StrategyGene.from_dict(strategy_gene_data)

        # バックテスト設定を準備（AutoStrategyServiceの_prepare_detailed_backtest_configと同様）
        backtest_config = {
            "strategy_name": f"AUTO_STRATEGY_TEST_{gene.id[:8]}",
            "strategy_config": {
                "strategy_type": "GENERATED_AUTO",
                "parameters": {"strategy_gene": gene.to_dict()},
            }
        }

        print(f"✅ バックテスト設定準備成功")
        print(f"✅ 戦略名: {backtest_config['strategy_name']}")
        print(f"✅ 戦略タイプ: {backtest_config['strategy_config']['strategy_type']}")

        # 戦略遺伝子の辞書変換が正しく動作することを確認
        gene_dict = backtest_config['strategy_config']['parameters']['strategy_gene']
        assert 'indicators' in gene_dict, "❌ 指標情報が含まれていません"
        assert 'entry_conditions' in gene_dict, "❌ エントリー条件が含まれていません"
        assert 'exit_conditions' in gene_dict, "❌ エグジット条件が含まれていません"

        print("✅ バックテスト設定の準備テスト成功！")
        return True

    except Exception as e:
        print(f"❌ バックテスト設定の準備テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """全てのテストを実行"""
    print("🧪 提供されたJSONパラメータでの修正検証テストを開始します...\n")

    tests = [
        ("提供されたJSONパラメータでの検証", test_provided_json_parameters),
        ("StrategyFactoryでの戦略クラス作成", test_strategy_factory_with_provided_json),
        ("バックテスト設定の準備", test_backtest_config_preparation),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"📋 {test_name}を実行中...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'成功' if result else '失敗'}\n")
        except Exception as e:
            print(f"❌ {test_name}: エラー - {e}\n")
            results.append((test_name, False))

    # 結果サマリー
    print("📊 テスト結果サマリー:")
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for test_name, result in results:
        print(f"  {'✅' if result else '❌'} {test_name}")
    
    print(f"\n🎯 成功: {success_count}/{total_count}")
    return success_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
