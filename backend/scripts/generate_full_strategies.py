import requests
import time


def generate_full_strategies():
    url = "http://127.0.0.1:8000/api/strategies/showcase/generate"
    data = {"count": 30}

    try:
        print("30個の戦略生成を開始...")
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        # 少し待ってから戦略一覧を確認
        print("\n5秒後に戦略一覧を確認...")
        time.sleep(5)

        # 戦略一覧を取得
        list_url = "http://127.0.0.1:8000/api/strategies/showcase"
        list_response = requests.get(list_url)
        strategies = list_response.json()

        print(f"\n生成された戦略数: {strategies['total_count']}")

        # 各戦略の概要を表示
        for strategy in strategies["strategies"]:
            print(
                f"- {strategy['name']} ({strategy['category']}, {strategy['risk_level']})"
            )
            print(
                f"  リターン: {strategy['expected_return']}%, シャープ: {strategy['sharpe_ratio']}, DD: {strategy['max_drawdown']}%"
            )

        # 統計情報を取得
        stats_url = "http://127.0.0.1:8000/api/strategies/showcase/stats"
        stats_response = requests.get(stats_url)
        stats = stats_response.json()

        print(f"\n=== 統計情報 ===")
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

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    generate_full_strategies()
