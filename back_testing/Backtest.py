#Back testing modulu

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Backtest:
    def __init__(self, initial_balance=100000, transaction_fee=1, margin_rate=0.1, stop_loss=0.05, 
                 trailing_stop_pct=0.02, point_value=1):
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.margin_rate = margin_rate  # 保證金比例 (例如，0.1 代表 10% 保證金)
        self.stop_loss = stop_loss  # 停損閾值 (例如，0.1 代表 10% 損失)
        self.trailing_stop_pct = trailing_stop_pct  # 移動停利百分比，用於調整停利
        self.trailing_stop = None  # 移動停利的水平
        self.balance = initial_balance
        self.position = 0  # 持有的股票數量 (正數表示多頭，負數表示空頭)
        self.trade_log = []
        self.portfolio_values = [initial_balance]
        self.entry_price = None  # 進場價格
        self.entry_time = None  # 進場時間
        self.profit_loss_log = []  # 記錄損益的日誌
        self.total_transaction_fees = 0  # 總交易手續費
        self.point_value = point_value  # 每一點的價值 (可以自定義)

    # Other methods remain the same...


    def execute_trade(self, action, price, time):
        """
        執行交易 (買入/賣出)，考慮保證金並計算每點的損益。
        
        參數:
        action (str): 'buy' (買入多頭), 'sell' (平倉多頭), 'short' (進空頭), 'cover' (平倉空頭)
        price (float): 執行交易的價格
        time (pd.Timestamp): 執行交易的時間
        """
        profit_loss = 0
        if action == 'buy' and self.position == 0:
            # 使用保證金計算購買的股票數量
            margin_balance = self.balance * self.margin_rate
            position_size = margin_balance / (price + self.transaction_fee)
            required_margin = position_size * price * self.margin_rate
            
            # 檢查保證金是否充足
            if self.balance >= required_margin:
                self.position += position_size
                self.balance -= position_size * (price + self.transaction_fee)
                self.total_transaction_fees += self.transaction_fee
                self.entry_price = price
                self.entry_time = time
                self.trade_log.append({'action': 'buy', 'price': price, 'position_size': position_size, 'time': time})
        elif action == 'sell' and self.position > 0:
            # 平倉所有持有的多頭股票
            profit_loss = ((price - self.entry_price) * self.point_value) * self.position - (2 * self.transaction_fee)
            self.balance += self.position * (price - self.transaction_fee)
            self.total_transaction_fees += self.transaction_fee
            self.trade_log.append({'action': 'sell', 'price': price, 'position_size': self.position, 'time': time, 'profit_loss': profit_loss})
            self.position = 0
            self.entry_price = None
            self.entry_time = None
            self.trailing_stop = None
        elif action == 'short' and self.position == 0:
            # 使用保證金進空頭
            margin_balance = self.balance * self.margin_rate
            position_size = margin_balance / (price + self.transaction_fee)
            required_margin = position_size * price * self.margin_rate
            
            # 檢查保證金是否充足
            if self.balance >= required_margin:
                self.position -= position_size
                self.balance += position_size * (price - self.transaction_fee)
                self.total_transaction_fees += self.transaction_fee
                self.entry_price = price
                self.entry_time = time
                self.trade_log.append({'action': 'short', 'price': price, 'position_size': position_size, 'time': time})
        elif action == 'cover' and self.position < 0:
            # 平倉所有持有的空頭股票
            profit_loss = ((self.entry_price - price) * self.point_value) * abs(self.position) - (2 * self.transaction_fee)
            self.balance -= abs(self.position) * (price + self.transaction_fee)
            self.total_transaction_fees += self.transaction_fee
            self.trade_log.append({'action': 'cover', 'price': price, 'position_size': abs(self.position), 'time': time, 'profit_loss': profit_loss})
            self.position = 0
            self.entry_price = None
            self.entry_time = None
            self.trailing_stop = None
        # 只有在平倉後才記錄資產價值
        if action in ['sell', 'cover']:
            self.portfolio_values.append(self.balance)
            self.profit_loss_log.append({'time': time, 'profit_loss': profit_loss})

    def run_backtest(self, predicted_returns, actual_prices,
                     buy_threshold = 0.0008,
                     short_threshold = -0.005):
        """
        根據預測回報率進行回測。
        
        參數:
        predicted_returns (pd.Series): 模型的預測回報率
        actual_prices (pd.Series): 參考的實際價格
        """
        for i in range(len(predicted_returns)):
            current_price = actual_prices[i]
            current_time = actual_prices.index[i]
            if self.position > 0:
                # 更新多頭倉位的移動停利
                if self.trailing_stop is None or current_price > self.trailing_stop:
                    self.trailing_stop = current_price * (1 - self.trailing_stop_pct)
                # 檢查多頭倉位的停損
                if (self.entry_price - current_price) / self.entry_price >= self.stop_loss:
                    self.execute_trade('sell', current_price, current_time)
                    continue
                # 檢查多頭倉位的移動停利
                if current_price <= self.trailing_stop:
                    self.execute_trade('sell', current_price, current_time)
                    continue
            elif self.position < 0:
                # 更新空頭倉位的移動停利
                if self.trailing_stop is None or current_price < self.trailing_stop:
                    self.trailing_stop = current_price * (1 + self.trailing_stop_pct)
                # 檢查空頭倉位的停損
                if (current_price - self.entry_price) / self.entry_price >= self.stop_loss:
                    self.execute_trade('cover', current_price, current_time)
                    continue
                # 檢查空頭倉位的移動停利
                if current_price >= self.trailing_stop:
                    self.execute_trade('cover', current_price, current_time)
                    continue
            # 根據預測回報率進行買入/賣出操作
            if predicted_returns[i] > buy_threshold:
                if self.position == 0:
                    self.execute_trade('buy', current_price, current_time)
            elif predicted_returns[i] < short_threshold:
                if self.position == 0:
                    self.execute_trade('short', current_price, current_time)

        # 如果在結束時還有持倉，則平倉
        if self.position > 0:
            self.execute_trade('sell', actual_prices.iloc[-1], actual_prices.index[-1])
        elif self.position < 0:
            self.execute_trade('cover', actual_prices.iloc[-1], actual_prices.index[-1])

        # 計算最終資產價值
        final_value = self.balance
        return_rate = (final_value - self.initial_balance) / self.initial_balance
        return return_rate, self.trade_log, self.portfolio_values, self.profit_loss_log

    def summary_table(self):
        """
        輸出總交易次數、總損益、總手續費、勝率、總平倉數量和賺賠比的摘要表。
        """
        total_trades = len(self.trade_log)
        total_profit_loss = sum(trade.get('profit_loss', 0) for trade in self.trade_log if 'profit_loss' in trade) - self.total_transaction_fees
        total_fees = self.total_transaction_fees
        
        winning_trades = sum(1 for trade in self.trade_log if 'profit_loss' in trade and trade['profit_loss'] > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_closed_positions = sum(1 for trade in self.trade_log if trade['action'] in ['sell', 'cover'])
        
        total_profit_points = sum(trade['profit_loss'] for trade in self.trade_log if 'profit_loss' in trade and trade['profit_loss'] > 0)
        total_loss_points = abs(sum(trade['profit_loss'] for trade in self.trade_log if 'profit_loss' in trade and trade['profit_loss'] < 0))
        profit_loss_ratio = (total_profit_points / total_loss_points) if total_loss_points > 0 else float('inf')
        
        summary_df = pd.DataFrame({
            'Metric': ['Total Trades', 'Total Profit/Loss (After Fees)', 'Total Transaction Fees', 'Win Rate (%)', 'Total Closed Positions', 'Profit/Loss Ratio'],
            'Value': [total_trades, total_profit_loss, total_fees, win_rate, total_closed_positions, profit_loss_ratio]
        })
        print(summary_df)

    def plot_profit_loss(self,time_series, predict, actual):
        """
        繪製未來預測與交易標記，以及隨著時間變化的實現損益和累積損益，並在下面的子圖右軸顯示累積損益。
        """
        profit_loss_df = pd.DataFrame(self.profit_loss_log)
        start_time = profit_loss_df['time'].min()
        end_time = profit_loss_df['time'].max()
        all_times = pd.date_range(start=start_time, end=end_time, freq='T')
        profit_loss_df = profit_loss_df.set_index('time').reindex(all_times, fill_value=0).reset_index()
        profit_loss_df.columns = ['time', 'profit_loss']
        
        # 計算累積損益
        profit_loss_df['cumulative_profit_loss'] = profit_loss_df['profit_loss'].cumsum()
        
        # 創建包含所有時間段的時間序列，沒有交易的時間點設置為 0
        profit_loss_series = profit_loss_df.set_index('time')['profit_loss']
        
        # 創建子圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        # 在 ax1 上繪製未來預測與交易標記
        ax1.plot(time_series, predict, color='#6a5acd', alpha=0.8, label="Ensemble Model Predictions")  # 使用紫色
        ax1.plot(time_series, actual, color='#1e90ff', alpha=0.8, label="True Values")  # 使用藍色

        # 繪製交易標記 (買入、賣出、做空、平倉)，只有一次性添加標籤
        buy_marker = None
        sell_marker = None
        short_marker = None
        cover_marker = None
        
        for trade in self.trade_log:
            if trade['action'] == 'buy':
                buy_marker = ax1.scatter(trade['time'], trade['price'], color='#32cd32', marker='^', s=100)  # 買入信號 (綠色)
            elif trade['action'] == 'sell':
                sell_marker = ax1.scatter(trade['time'], trade['price'], color='#ff6347', marker='v', s=100)  # 賣出信號 (紅色)
            elif trade['action'] == 'short':
                short_marker = ax1.scatter(trade['time'], trade['price'], color='#ff8c00', marker='s', s=100)  # 做空信號 (橙色)
            elif trade['action'] == 'cover':
                cover_marker = ax1.scatter(trade['time'], trade['price'], color='#ffd700', marker='D', s=100)  # 平倉信號 (黃色)
        
        # 在圖例中添加一次性的標籤
        ax1.legend(
            handles=[buy_marker, sell_marker, short_marker, cover_marker],
            labels=["Buy Signal", "Sell Signal", "Short Signal", "Cover Signal"],
            loc='upper left', bbox_to_anchor=(1, 1), fontsize=12
        )

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Prices')
        ax1.set_title('Future Predictions vs Actual Future Values with Trade Markers', fontsize=16)
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        ax1.tick_params(axis='x', rotation=45, labelcolor='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # 在 ax2 上繪製隨著時間變化的實現損益柱狀圖
        ax2.bar(profit_loss_series.index, profit_loss_series.values, color=['#32cd32' if v > 0 else '#ff6347' for v in profit_loss_series.values], alpha=0.7, width=0.005)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Profit / Loss')
        ax2.set_title('Realized Profit/Loss Over Time', fontsize=16)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)
        ax2.tick_params(axis='x', rotation=45, labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # 創建右軸顯示累積損益
        ax3 = ax2.twinx()  # 創建與 ax2 共享 x 軸的右軸
        ax3.plot(profit_loss_df['time'], profit_loss_df['cumulative_profit_loss'], color='#8a2be2', alpha=0.9)  # 使用紫色
        ax3.set_ylabel('Cumulative Profit/Loss', color='black')
        ax3.tick_params(axis='y', labelcolor='black')

        # 移除圖例：首先檢查是否存在圖例
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()
        if ax3.get_legend() is not None:
            ax3.get_legend().remove()
        
        # 自動調整布局
        plt.tight_layout()
        plt.show()