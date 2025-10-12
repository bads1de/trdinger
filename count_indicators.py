import re

with open('C:/Users/buti3/trading/backend/app/services/indicators/manifest.py', 'r', encoding='utf-8') as f:
    content = f.read()

# インジケーター名を抽出
matches = re.findall(r"'([A-Z_]+)': \{", content)
unique_indicators = list(set(matches))
print(f"見つかったユニークなインジケーター数: {len(unique_indicators)}")
print("インジケーター一覧:")
for i, indicator in enumerate(sorted(unique_indicators), 1):
    print(f"{i:2d}. {indicator}")
