import json

# JSONファイルを読み込んで実際のデータを確認
with open('backtest_result_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== 実際のJSONデータ確認 ===')
print(f'戦略名: {data["strategy_name"]}')
print(f'初期資金: {data["initial_capital"]}')
print(f'最終資産: {data["final_equity"]}')
print(f'総リターン: {data["total_return"]}')
print(f'総取引数: {data["total_trades"]}')
print(f'勝率: {data["win_rate"]}')
print(f'Sharpe比率: {data["sharpe_ratio"]}')
print(f'最大ドローダウン: {data["max_drawdown"]}')
print(f'取引履歴数: {len(data["trade_history"]) if "trade_history" in data else 0}')
print(f'資産曲線ポイント数: {len(data["equity_curve"]) if "equity_curve" in data else 0}')

print('\n=== 主要メトリクス一覧 ===')
key_metrics = ['total_return', 'total_trades', 'win_rate', 'sharpe_ratio', 'max_drawdown', 'final_equity', 'equity_peak']
for key in key_metrics:
    value = data.get(key, 'N/A')
    status = 'OK' if value != 0 and value != 'N/A' and value != 0.0 else 'ZERO'
    print(f'{key}: {value} [{status}]')

print('\n=== 取引履歴サンプル ===')
if 'trade_history' in data and len(data['trade_history']) > 0:
    trade = data['trade_history'][0]
    print(f'エントリー時間: {trade.get("entry_time", "N/A")}')
    print(f'エグジット時間: {trade.get("exit_time", "N/A")}')
    print(f'エントリー価格: {trade.get("entry_price", "N/A")}')
    print(f'エグジット価格: {trade.get("exit_price", "N/A")}')
    print(f'PnL: {trade.get("pnl", "N/A")}')
    print(f'リターン%: {trade.get("return_pct", "N/A")}')
else:
    print('取引履歴なし')

print('\n=== 資産曲線サンプル ===')
if 'equity_curve' in data and len(data['equity_curve']) > 0:
    first_point = data['equity_curve'][0]
    last_point = data['equity_curve'][-1]
    print(f'開始時: {first_point.get("timestamp", "N/A")} - 資産: {first_point.get("equity", "N/A")}')
    print(f'終了時: {last_point.get("timestamp", "N/A")} - 資産: {last_point.get("equity", "N/A")}')
else:
    print('資産曲線なし')

print('\n=== 修正確認 ===')
all_non_zero = all(data.get(key, 0) != 0 and data.get(key, 0) != 0.0 for key in ['total_return', 'total_trades', 'win_rate', 'sharpe_ratio'])
print(f'主要メトリクスがすべて非ゼロ: {"SUCCESS" if all_non_zero else "FAILED"}')
print(f'統計抽出: {"SUCCESS" if len([k for k in key_metrics if k in data]) > 0 else "FAILED"}')
print(f'取引履歴抽出: {"SUCCESS" if len(data.get("trade_history", [])) > 0 else "FAILED"}')
print(f'資産曲線抽出: {"SUCCESS" if len(data.get("equity_curve", [])) > 0 else "FAILED"}')