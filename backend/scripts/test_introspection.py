import pandas_ta as ta
import re
import inspect

def get_indicators_from_category():
    all_indicators = []
    for cat, items in ta.Category.items():
        all_indicators.extend(items)
    return sorted(list(set(all_indicators)))

def detect_return_type(name):
    func = getattr(ta, name, None)
    if not func: return "Unknown"
    doc = func.__doc__
    if not doc: return "Unknown"
    
    # Returns:セクションを探す
    match = re.search(r"Returns:\s+pd\.(DataFrame|Series)", doc)
    if match:
        return match.group(1)
    return "Unknown"

indicators = ["rsi", "macd", "bbands", "sma", "stoch"]
for ind in indicators:
    print(f"{ind}: {detect_return_type(ind)}")

print("\nTotal indicators from Category:", len(get_indicators_from_category()))
