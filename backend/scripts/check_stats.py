import requests

# 統計情報を取得
stats_url = "http://127.0.0.1:8000/api/strategies/showcase/stats"
stats_response = requests.get(stats_url)
stats = stats_response.json()

print("=== 投資戦略ショーケース統計 ===")
print(f"総戦略数: {stats['statistics']['total_strategies']}")
print(f"平均リターン: {stats['statistics']['avg_return']:.2f}%")
print(f"平均シャープレシオ: {stats['statistics']['avg_sharpe_ratio']:.3f}")
print(f"平均最大ドローダウン: {stats['statistics']['avg_max_drawdown']:.2f}%")

print(f"\nカテゴリ分布:")
for category, count in stats["statistics"]["category_distribution"].items():
    print(f"  {category}: {count}個")

print(f"\nリスク分布:")
for risk, count in stats["statistics"]["risk_distribution"].items():
    print(f"  {risk}: {count}個")

# 戦略一覧の一部を表示
list_url = "http://127.0.0.1:8000/api/strategies/showcase?limit=10&sort_by=expected_return&sort_order=desc"
list_response = requests.get(list_url)
strategies = list_response.json()

print(f"\n=== トップ10戦略（リターン順） ===")
for i, strategy in enumerate(strategies["strategies"], 1):
    print(f"{i:2d}. {strategy['name']}")
    print(f"    カテゴリ: {strategy['category']}, リスク: {strategy['risk_level']}")
    print(
        f"    リターン: {strategy['expected_return']}%, シャープ: {strategy['sharpe_ratio']}, DD: {strategy['max_drawdown']}%"
    )
    print(f"    指標: {', '.join(strategy['indicators'])}")
    print()
