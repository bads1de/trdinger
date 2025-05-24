"""
戦略実行エンジン
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

from .indicators import TechnicalIndicators


@dataclass
class Trade:
    """取引記録"""
    timestamp: datetime
    type: str  # 'buy' or 'sell'
    price: float
    quantity: float
    commission: float
    pnl: float = 0.0


@dataclass
class Position:
    """ポジション情報"""
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0


class StrategyExecutor:
    """戦略実行クラス"""
    
    def __init__(self, initial_capital: float = 100000, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.reset()
    
    def reset(self):
        """状態をリセット"""
        self.capital = self.initial_capital
        self.position = Position()
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.indicators_cache: Dict[str, Any] = {}
    
    def calculate_indicators(self, data: pd.DataFrame, indicators_config: List[Dict]) -> Dict[str, Any]:
        """指標を計算してキャッシュ"""
        self.indicators_cache = {}
        
        for indicator_config in indicators_config:
            name = indicator_config['name']
            params = indicator_config['params']
            
            # 指標を計算
            result = TechnicalIndicators.calculate_indicator(data, name, params)
            
            # キャッシュに保存（複数の値を返す指標の場合は辞書として保存）
            if isinstance(result, dict):
                for key, value in result.items():
                    cache_key = f"{name}_{key}"
                    self.indicators_cache[cache_key] = pd.Series(value, index=data.index)
            else:
                self.indicators_cache[name] = pd.Series(result, index=data.index)
        
        return self.indicators_cache
    
    def evaluate_condition(self, condition: str, current_index: int, data: pd.DataFrame) -> bool:
        """
        売買条件を評価
        
        Args:
            condition: 条件文字列 (例: "SMA(close, 20) > SMA(close, 50)")
            current_index: 現在のデータインデックス
            data: 価格データ
            
        Returns:
            条件が満たされているかどうか
        """
        try:
            # 現在の価格データを取得
            current_data = data.iloc[current_index]
            
            # 条件文字列を評価可能な形に変換
            eval_condition = self._parse_condition(condition, current_index, current_data)
            
            # 条件を評価
            return eval(eval_condition)
        
        except Exception as e:
            print(f"条件評価エラー: {condition}, エラー: {e}")
            return False
    
    def _parse_condition(self, condition: str, current_index: int, current_data: pd.Series) -> str:
        """条件文字列を評価可能な形に変換"""
        # 価格データの置換 (close, open, high, low)
        condition = re.sub(r'\bclose\b', str(current_data['close']), condition)
        condition = re.sub(r'\bopen\b', str(current_data['open']), condition)
        condition = re.sub(r'\bhigh\b', str(current_data['high']), condition)
        condition = re.sub(r'\blow\b', str(current_data['low']), condition)
        
        # 指標の置換 (例: SMA(close, 20) -> cached_sma_value)
        # SMA, EMA, RSI などの単純な指標
        for indicator_name in ['SMA', 'EMA', 'RSI', 'ATR']:
            pattern = rf'{indicator_name}\([^,]+,\s*(\d+)\)'
            matches = re.findall(pattern, condition)
            for period in matches:
                cache_key = f"{indicator_name}"
                if cache_key in self.indicators_cache:
                    value = self.indicators_cache[cache_key].iloc[current_index]
                    if not pd.isna(value):
                        condition = re.sub(
                            rf'{indicator_name}\([^,]+,\s*{period}\)',
                            str(value),
                            condition,
                            count=1
                        )
        
        # MACD の置換
        macd_pattern = r'MACD\([^)]+\)'
        if re.search(macd_pattern, condition):
            if 'MACD_macd' in self.indicators_cache:
                macd_value = self.indicators_cache['MACD_macd'].iloc[current_index]
                if not pd.isna(macd_value):
                    condition = re.sub(macd_pattern, str(macd_value), condition)
        
        return condition
    
    def execute_trade(self, trade_type: str, price: float, timestamp: datetime, 
                     quantity: float = None) -> Trade:
        """取引を実行"""
        if quantity is None:
            # 全資金で取引（簡易実装）
            if trade_type == 'buy':
                quantity = (self.capital * 0.95) / price  # 手数料を考慮して95%
            else:  # sell
                quantity = self.position.quantity
        
        commission = price * quantity * self.commission_rate
        
        if trade_type == 'buy' and self.position.quantity == 0:
            # 新規買いポジション
            total_cost = price * quantity + commission
            if total_cost <= self.capital:
                self.capital -= total_cost
                self.position.quantity = quantity
                self.position.avg_price = price
                
                trade = Trade(
                    timestamp=timestamp,
                    type=trade_type,
                    price=price,
                    quantity=quantity,
                    commission=commission
                )
                self.trades.append(trade)
                return trade
        
        elif trade_type == 'sell' and self.position.quantity > 0:
            # ポジション決済
            proceeds = price * quantity - commission
            pnl = (price - self.position.avg_price) * quantity - commission
            
            self.capital += proceeds
            self.position.quantity -= quantity
            
            if self.position.quantity <= 0:
                self.position.quantity = 0
                self.position.avg_price = 0
            
            trade = Trade(
                timestamp=timestamp,
                type=trade_type,
                price=price,
                quantity=quantity,
                commission=commission,
                pnl=pnl
            )
            self.trades.append(trade)
            return trade
        
        return None
    
    def update_equity(self, current_price: float, timestamp: datetime):
        """現在の資産価値を更新"""
        if self.position.quantity > 0:
            position_value = self.position.quantity * current_price
            total_equity = self.capital + position_value
            unrealized_pnl = (current_price - self.position.avg_price) * self.position.quantity
        else:
            total_equity = self.capital
            unrealized_pnl = 0
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'capital': self.capital,
            'position_value': position_value if self.position.quantity > 0 else 0,
            'unrealized_pnl': unrealized_pnl
        })
    
    def run_backtest(self, data: pd.DataFrame, strategy_config: Dict) -> Dict:
        """バックテストを実行"""
        self.reset()
        
        # 指標を計算
        indicators = strategy_config.get('indicators', [])
        self.calculate_indicators(data, indicators)
        
        entry_rules = strategy_config.get('entry_rules', [])
        exit_rules = strategy_config.get('exit_rules', [])
        
        # データを順次処理
        for i in range(len(data)):
            current_data = data.iloc[i]
            timestamp = current_data.name
            current_price = current_data['close']
            
            # エントリー条件をチェック
            if self.position.quantity == 0:  # ポジションなし
                entry_signal = all(
                    self.evaluate_condition(rule['condition'], i, data)
                    for rule in entry_rules
                )
                
                if entry_signal:
                    self.execute_trade('buy', current_price, timestamp)
            
            # エグジット条件をチェック
            elif self.position.quantity > 0:  # ポジションあり
                exit_signal = any(
                    self.evaluate_condition(rule['condition'], i, data)
                    for rule in exit_rules
                )
                
                if exit_signal:
                    self.execute_trade('sell', current_price, timestamp)
            
            # 資産価値を更新
            self.update_equity(current_price, timestamp)
        
        # 最終的にポジションが残っている場合は決済
        if self.position.quantity > 0:
            final_price = data.iloc[-1]['close']
            final_timestamp = data.index[-1]
            self.execute_trade('sell', final_price, final_timestamp)
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict:
        """パフォーマンス指標を計算"""
        if not self.equity_curve:
            return {}
        
        equity_series = pd.Series([point['equity'] for point in self.equity_curve])
        returns = equity_series.pct_change().dropna()
        
        # 基本指標
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # シャープレシオ
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # 年率換算
        else:
            sharpe_ratio = 0
        
        # 最大ドローダウン
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # 取引統計
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        total_trades = len([t for t in self.trades if t.type == 'sell'])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity_series.iloc[-1],
            'equity_curve': self.equity_curve,
            'trades': [
                {
                    'timestamp': t.timestamp.isoformat(),
                    'type': t.type,
                    'price': t.price,
                    'quantity': t.quantity,
                    'commission': t.commission,
                    'pnl': t.pnl
                }
                for t in self.trades
            ]
        }
