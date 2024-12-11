#Back testing modulu

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from datetime import time
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


    import pandas as pd
    from datetime import time

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

    def run_backtest(self, predicted_returns, actual_prices, buy_threshold=0.0008, short_threshold=-0.005, no_trade_before_9am=True):
        """
        根據預測回報率進行回測。
        
        參數:
        predicted_returns (pd.Series): 模型的預測回報率
        actual_prices (pd.Series): 參考的實際價格
        """
        for i in range(len(predicted_returns)):
            # 跳過早上九點前的交易
            if no_trade_before_9am and actual_prices.index[i].time() < time(9, 0):
                continue
            
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
        total_profit_loss = sum(trade.get('profit_loss', 0) for trade in self.profit_loss_log if 'profit_loss' in trade) - self.total_transaction_fees
        total_fees = self.total_transaction_fees
        
        winning_trades = sum(1 for trade in self.profit_loss_log if 'profit_loss' in trade and trade['profit_loss'] > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_closed_positions = sum(1 for trade in self.profit_loss_log if trade['action'] in ['sell', 'cover'])
        
        total_profit_points = sum(trade['profit_loss'] for trade in self.profit_loss_log if 'profit_loss' in trade and trade['profit_loss'] > 0)
        total_loss_points = abs(sum(trade['profit_loss'] for trade in self.profit_loss_log if 'profit_loss' in trade and trade['profit_loss'] < 0))
        profit_loss_ratio = (total_profit_points / total_loss_points) if total_loss_points > 0 else float('inf')
        
        summary_df = pd.DataFrame({
            'Metric': ['Total Trades', 'Total Profit/Loss (After Fees)', 'Total Transaction Fees', 'Win Rate (%)', 'Total Closed Positions', 'Profit/Loss Ratio'],
            'Value': [total_trades, total_profit_loss, total_fees, win_rate, total_closed_positions, profit_loss_ratio]
        })
        print(summary_df)

    def plot_profit_loss(self, time, return_rate, actual):
        """
        繪製未來預測與交易標記，以及隨著時間變化的實現損益和累積損益。
        """
        # 構建損益 DataFrame
        profit_loss_df = pd.DataFrame(self.profit_loss_log)

        # 確保存在 'time' 和 'profit_loss' 欄位
        if 'time' not in profit_loss_df.columns or 'profit_loss' not in profit_loss_df.columns:
            raise ValueError("profit_loss_log 中缺少 'time' 或 'profit_loss' 欄位")
        
        # 設定時間範圍和填充缺失時間點
        start_time = profit_loss_df['time'].min()
        end_time = profit_loss_df['time'].max()
        all_times = pd.date_range(start=start_time, end=end_time, freq='T')
        profit_loss_df = (
            profit_loss_df.set_index('time')
            .reindex(all_times, fill_value=0)
            .reset_index()
            .rename(columns={'index': 'time'})
        )

        # 計算累積損益
        profit_loss_df['cumulative_profit_loss'] = profit_loss_df['profit_loss'].cumsum()
            
        # 創建子圖
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

        # 在 ax1 上繪製未來預測與交易標記
        ax1.plot(time, actual, color='#1e90ff', alpha=0.8, label="True Values")  # 使用藍色

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
        markers = [m for m in [buy_marker, sell_marker, short_marker, cover_marker] if m is not None]
        labels = ["Buy Signal", "Sell Signal", "Short Signal", "Cover Signal"][:len(markers)]
        ax1.legend(handles=markers, labels=labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Prices')
        ax1.set_title('Future Predictions vs Actual Future Values with Trade Markers', fontsize=16)
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        ax1.tick_params(axis='x', rotation=45, labelcolor='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # 在 ax2 上繪製隨著時間變化的實現損益柱狀圖
        ax2.bar(profit_loss_series.index, profit_loss_series.values, color=['#32cd32' if v > 0 else '#ff6347' for v in profit_loss_series.values], alpha=0.7, width=0.01)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Profit / Loss')
        ax2.set_title('Realized Profit/Loss Over Time', fontsize=16)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)
        ax2.tick_params(axis='x', rotation=45, labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # 創建右軸顯示累積損益
        ax5 = ax2.twinx()  # 創建與 ax2 共享 x 軸的右軸
        ax5.plot(profit_loss_df['time'], profit_loss_df['cumulative_profit_loss'], color='#8a2be2', alpha=0.9)  # 使用紫色
        ax5.set_ylabel('Cumulative Profit/Loss', color='black')
        ax5.tick_params(axis='y', labelcolor='black')

        # 在 ax3 上繪製 predicted_returns
        ax3.plot(time, return_rate, color='#ff4500', alpha=0.8, label='Predicted Returns')  # 使用橙紅色
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Predicted Returns')
        ax3.set_title('Predicted Returns Over Time', fontsize=16)
        ax3.grid(axis='y', linestyle='--', alpha=0.6)
        ax3.tick_params(axis='x', rotation=45, labelcolor='black')
        ax3.tick_params(axis='y', labelcolor='black')
        
        # 自動調整布局
        plt.tight_layout()
        plt.show()
        plt.tight_layout()
        plt.show()
        
        
#Back testing modulus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import datetime

from datetime import time
class ForwardTest:
    def __init__(self, initial_balance=100000, transaction_fee=1, margin_rate=0.1, stop_loss=0.001, 
                 trailing_stop_pct=0.000001, point_value=1, data_folder="trading_data", symbol="unknown",
                 start_time = "09:00", end_time = "13:25"):
        # 初始化參數
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.margin_rate = margin_rate
        self.stop_loss = stop_loss
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_stop = None
        self.balance = initial_balance
        self.position = 0
        self.trade_log = []
        self.portfolio_values = [initial_balance]
        self.entry_price ,self.entry_time = None, None
        self.profit_loss_log = []
        self.total_transaction_fees = 0
        self.point_value = point_value
        self.lookback_prices = []
        self.strategy_status = []
        self.data_folder = data_folder
        self.symbol = symbol
        self.trailing_stop_log, self.stop_loss_log = [], []
        self.trade_time = {
            "start_time": start_time,  # 默認開始時間
            "end_time": end_time    # 默認結束時間
        }


        # 嘗試從資料夾加載數據
        previous_data = self.load_trading_files()
        if previous_data:
            self._initialize_from_saved_data(previous_data)

    def _initialize_from_saved_data(self, data):
        """
        根據保存的數據初始化交易機器人狀態。
        """
        self.balance = data.get("balance", self.initial_balance)
        self.position = data.get("position", 0)
        self.entry_price = data.get("entry_price", None)
        self.trailing_stop = data.get("trailing_stop", None)
        self.stop_loss = data.get("stop_loss", self.stop_loss)
        self.entry_time = pd.to_datetime(data.get("entry_time"), errors='coerce')
        self.total_transaction_fees = data.get("total_transaction_fees", 0)

        # 轉換 trade_log 和 profit_loss_log 的時間字段為 Timestamp
        def deserialize_time_field(log_list):
            for entry in log_list:
                if 'time' in entry:
                    entry['time'] = pd.to_datetime(entry['time'], errors='coerce')

        self.trade_log = data.get("trade_log", [])
        deserialize_time_field(self.trade_log)

        self.profit_loss_log = data.get("profit_loss_log", [])
        deserialize_time_field(self.profit_loss_log)

        self.stop_loss_log = data.get("stop_loss_log", [])
        deserialize_time_field(self.stop_loss_log)

        print(f"Initialized from saved data: Balance={self.balance}, Position={self.position}, Total Fees={self.total_transaction_fees}")


    def save_trading_files(self):
        """
        保存交易相关文件到指定文件夹，时间格式化为指定样式。
        """
        # 确保文件夹存在
        latest_time = "unknown_date"
        if self.trade_log:
            latest_time = max(
                pd.to_datetime(entry['time']) for entry in self.trade_log if 'time' in entry
            ).strftime("%Y%m%d")
        subfolder = f"{self.symbol}_{latest_time}"
        target_folder = os.path.join(self.data_folder, subfolder)
        os.makedirs(target_folder, exist_ok=True)

        # 格式化时间字段的副本
        def format_time_field(log_list):
            formatted_log = copy.deepcopy(log_list)
            for entry in formatted_log:
                if 'time' in entry and isinstance(entry['time'], pd.Timestamp):
                    entry['time'] = entry['time'].strftime("%Y/%m/%d %I:%M:%S %p")
            return formatted_log

        # 保存交易日志
        if self.trade_log:
            formatted_trade_log = format_time_field(self.trade_log)
            trade_log_df = pd.DataFrame(formatted_trade_log)
            trade_log_df.to_json(os.path.join(target_folder, "trade_log.json"), orient="records", indent=4)
            trade_log_df.to_excel(os.path.join(target_folder, "trade_log.xlsx"), index=False)

        # 保存损益日志
        if self.profit_loss_log:
            formatted_profit_loss_log = format_time_field(self.profit_loss_log)
            profit_loss_log_df = pd.DataFrame(formatted_profit_loss_log)
            profit_loss_log_df.to_json(os.path.join(target_folder, "profit_loss_log.json"), orient="records", indent=4)
            profit_loss_log_df.to_excel(os.path.join(target_folder, "profit_loss_log.xlsx"), index=False)

        # 保存停损日志
        if self.stop_loss_log:
            formatted_stop_loss_log = format_time_field(self.stop_loss_log)
            stop_loss_log_df = pd.DataFrame(formatted_stop_loss_log)
            stop_loss_log_df.to_json(os.path.join(target_folder, "stop_loss_log.json"), orient="records", indent=4)
            stop_loss_log_df.to_excel(os.path.join(target_folder, "stop_loss_log.xlsx"), index=False)

        # 保存当前状态
        status = {
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "trailing_stop": self.trailing_stop,
            "stop_loss": self.stop_loss,
            "entry_time": self.entry_time.strftime("%Y/%m/%d %I:%M:%S %p") if isinstance(self.entry_time, pd.Timestamp) else self.entry_time,
            "total_transaction_fees": self.total_transaction_fees
        }
        with open(os.path.join(target_folder, "status.json"), 'w') as f:
            json.dump(status, f, indent=4)

        print(f"Trading data saved to folder: {target_folder}")




    def load_trading_files(self):
        """
        從指定資料夾加載交易相關文件，並格式化時間字段。
        """
        data = {}

        # 確認資料夾是否存在
        if not os.path.exists(self.data_folder):
            print(f"Data folder {self.data_folder} does not exist.")
            return data

        # 找到最新的子資料夾
        subfolders = [f.path for f in os.scandir(self.data_folder) if f.is_dir()]
        if not subfolders:
            print(f"No subfolders found in {self.data_folder}.")
            return data

        latest_folder = max(subfolders, key=os.path.getmtime)

        def parse_time_field(log_list, field_name='time'):
            """解析時間字段並轉換為 Timestamp"""
            for entry in log_list:
                if field_name in entry and isinstance(entry[field_name], str):
                    entry[field_name] = pd.to_datetime(entry[field_name], errors='coerce')
            return log_list

        # 讀取交易明細
        trade_log_path = os.path.join(latest_folder, "trade_log.json")
        if os.path.exists(trade_log_path) and os.path.getsize(trade_log_path) > 0:
            with open(trade_log_path, 'r') as f:
                self.trade_log = parse_time_field(json.load(f))

        # 讀取損益紀錄
        profit_loss_log_path = os.path.join(latest_folder, "profit_loss_log.json")
        if os.path.exists(profit_loss_log_path) and os.path.getsize(profit_loss_log_path) > 0:
            with open(profit_loss_log_path, 'r') as f:
                self.profit_loss_log = parse_time_field(json.load(f))

        # 讀取停損紀錄
        stop_loss_log_path = os.path.join(latest_folder, "stop_loss_log.json")
        if os.path.exists(stop_loss_log_path) and os.path.getsize(stop_loss_log_path) > 0:
            with open(stop_loss_log_path, 'r') as f:
                self.stop_loss_log = parse_time_field(json.load(f), field_name="Time")

        # 讀取當前狀態
        status_path = os.path.join(latest_folder, "status.json")
        if os.path.exists(status_path) and os.path.getsize(status_path) > 0:
            with open(status_path, 'r') as f:
                status_data = json.load(f)
                if 'entry_time' in status_data and isinstance(status_data['entry_time'], str):
                    status_data['entry_time'] = pd.to_datetime(status_data['entry_time'], errors='coerce')
                self.total_transaction_fees = status_data.get("total_transaction_fees", 0)
                self.balance = status_data.get("balance", 0)
                self.position = status_data.get("position", 0)
                self.entry_price = status_data.get("entry_price", None)
                self.trailing_stop = status_data.get("trailing_stop", None)
                self.stop_loss = status_data.get("stop_loss", None)
                self.entry_time = status_data.get("entry_time", None)

        print(f"Trading data loaded from folder: {latest_folder}")



    def _prepare_serializable(self, obj):
        """
        將資料轉換為 JSON 可序列化格式，使用 pandas 和 numpy 的內建方法處理。
        """
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')  # DataFrame 转为列表字典
        if isinstance(obj, pd.Series):
            return obj.tolist()  # Series 转为列表
        if isinstance(obj, list):
            return [self._prepare_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self._prepare_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return obj.isoformat()  # 时间类型转 ISO 格式字符串
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # numpy 类型转为 Python 原生类型
        return obj  # 返回基础类型（int, float, str, None）

    def terminate_trading(self, current_price, current_time):
        """
        終止交易並保存數據。
        """
        # 平倉所有持倉
        if self.position > 0:
            print(f"Terminating trading: Selling all long positions at {current_time}, Price: {current_price}")
            self.execute_trade('sell', current_price, current_time, contracts=self.position)
        elif self.position < 0:
            print(f"Terminating trading: Covering all short positions at {current_time}, Price: {current_price}")
            self.execute_trade('cover', current_price, current_time, contracts=abs(self.position))

        # 保存數據到資料夾
        self.save_trading_files()
        print("Trading terminated and data saved.")

    def execute_trade(self, action, price, time, contracts=1):
        profit_loss = 0
        if action == 'buy' and self.position == 0:
            # Buy contracts (long position)
            required_margin = contracts * price * self.margin_rate

            if self.balance >= required_margin:
                self.position += contracts
                self.balance -= required_margin + (contracts * self.transaction_fee)
                self.total_transaction_fees += contracts * self.transaction_fee
                self.entry_price = price
                self.entry_time = time
                self.trade_log.append({'action': 'buy', 'price': price, 'contracts': contracts, 'time': time})
                
        elif action == 'sell' and self.position > 0:
            # Sell contracts (close long position)
            profit_loss = (price - self.entry_price) * self.position * self.point_value - (2 * self.transaction_fee * self.position)
            self.balance += (self.position * price * self.margin_rate) + profit_loss
            self.total_transaction_fees += self.transaction_fee * self.position
            self.trade_log.append({'action': 'sell', 'price': price, 'contracts': self.position, 'time': time, 'profit_loss': profit_loss})
            # Record profit/loss to log
            self.profit_loss_log.append({'time': time, 'profit_loss': profit_loss, 'strategy': self.strategy_status[-1]['Strategy state']})
            self.position = 0
            self.entry_price = None
            self.entry_time = None
            self.trailing_stop = None
            
        elif action == 'short' and self.position == 0:
            # Enter short contracts
            required_margin = contracts * price * self.margin_rate

            if self.balance >= required_margin:
                self.position -= contracts
                self.balance += required_margin - (contracts * self.transaction_fee)
                self.total_transaction_fees += contracts * self.transaction_fee
                self.entry_price = price
                self.entry_time = time
                self.trade_log.append({'action': 'short', 'price': price, 'contracts': contracts, 'time': time})
                
        elif action == 'cover' and self.position < 0:
            # Cover contracts (close short position)
            profit_loss = (self.entry_price - price) * abs(self.position) * self.point_value - (2 * self.transaction_fee * abs(self.position))
            self.balance += (abs(self.position) * price * self.margin_rate) + profit_loss
            self.total_transaction_fees += self.transaction_fee * abs(self.position)
            self.trade_log.append({'action': 'cover', 'price': price, 'contracts': abs(self.position), 'time': time, 'profit_loss': profit_loss})
            # Record profit/loss to log
            self.profit_loss_log.append({'time': time, 'profit_loss': profit_loss, 'strategy': self.strategy_status[-1]['Strategy state']})
            self.position = 0
            self.entry_price = None
            self.entry_time = None
            self.trailing_stop = None
        if action in ['sell', 'cover']:
            self.portfolio_values.append(self.balance)

    def detect_market_state(self, predicted_return, actual_price, threshold=0.02, osc_window=5):
        if len(self.lookback_prices) < osc_window:
            return 'default'

        recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
        recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']

        if recent_low <= actual_price <= recent_high:
            return 'oscillation'
        elif actual_price > recent_high or actual_price < recent_low:
            return 'trend'
        return 'None'

    def apply_strategy(self, strategy_func, *args, **kwargs):
        strategy_func(*args, **kwargs)

    def oscillation_strategy(self, predicted_return, actual_price, current_time, recent_high, recent_low):
        # 计算区间上下边界
        lower_bound = recent_low + (recent_high - recent_low) * 0.2
        upper_bound = recent_high - (recent_high - recent_low) * 0.2

        # 买入策略
        if predicted_return > 0 and actual_price < lower_bound:
            if self.position == 0:
                self.execute_trade('buy', actual_price, current_time, contracts=1)

        # 做空策略
        elif predicted_return < 0 and actual_price > upper_bound:
            if self.position == 0:
                self.execute_trade('short', actual_price, current_time, contracts=1)


    def trend_strategy(self, predicted_return, actual_price, current_time, buy_threshold, short_threshold, recent_high, recent_low):
        # 计算回撤位
        retrace_382 = recent_low + (recent_high - recent_low) * 0.382
        retrace_618 = recent_low + (recent_high - recent_low) * 0.618

        # 买入策略
        if predicted_return > buy_threshold and actual_price < retrace_382:
            if self.position == 0:
                self.execute_trade('buy', actual_price, current_time, contracts=1)

        # 做空策略
        elif predicted_return < short_threshold and actual_price > retrace_618:
            if self.position == 0:
                self.execute_trade('short', actual_price, current_time, contracts=1)

    """
    突破策略:追高空低
    """               
    def breakout_strategy(self, predicted_return, actual_price, current_time, recent_high, recent_low):
        
        if  recent_high < actual_price:
            if self.position == 0:
                self.execute_trade('buy', actual_price, current_time, contracts=1)
                
        elif recent_low > actual_price:
            if self.position == 0:
                self.execute_trade('short', actual_price, current_time, contracts=1)
                
    def default_strategy(self, predicted_return, actual_price, current_time,
                      buy_threshold, short_threshold):
        
        #當下的五分K最高與最低價格
        high, low = self.lookback_prices[-1]['high'], self.lookback_prices[-1]['low']
        
        retrace_low = low + (high - low) * 0.2
        retrace_high= low + (high - low) * 0.8
        # 买入策略
        if predicted_return > buy_threshold and actual_price < retrace_low:
            if self.position == 0:
                self.execute_trade('buy', actual_price, current_time, contracts=1)

        # 做空策略
        elif predicted_return < short_threshold and actual_price > retrace_high:
            if self.position == 0:
                self.execute_trade('short', actual_price, current_time, contracts=1)
            
        

    #接收預測與即時資料
    def run_backtest(self, predicted_return, current_time, real_time_data, 
                     buy_threshold=0.0002, short_threshold=-0.0001,
                     osc_window=2, contracts=1):
        
        actual_price = real_time_data['close']
        high = real_time_data['high']
        low = real_time_data['low']
        open_price = real_time_data['open']
        volume = real_time_data['volume']

        # 將字符串格式時間轉為 datetime.time
        start_time = pd.to_datetime(self.trade_time["start_time"], format="%H:%M").time()
        end_time = pd.to_datetime(self.trade_time["end_time"], format="%H:%M").time()
    
        # 限制交易時間：9:00 AM 前和 1:30 PM 後
        if current_time.time() < start_time or current_time.time() > end_time:
            # 如果有持倉，強制平倉
            if self.position > 0:  # 平多單
                print(f"Force close long position at {current_time}, Price: {actual_price}")
                self.execute_trade('sell', actual_price, current_time, contracts=self.position)
            elif self.position < 0:  # 平空單
                print(f"Force close short position at {current_time}, Price: {actual_price}")
                self.execute_trade('cover', actual_price, current_time, contracts=abs(self.position))
            
            # 阻止任何交易行為，直接返回目前狀態
            return self.get_current_state()
        
        # Update historical prices for oscillation/trend detection
        self.lookback_prices.append({'high': high, 'low': low, 'close': actual_price, 'volumne' : volume})
        if len(self.lookback_prices) > osc_window + 1:
            self.lookback_prices.pop(0)

        # Detect market state
        market_state = self.detect_market_state(predicted_return, actual_price, osc_window=osc_window)

        self.strategy_status.append({'Time': current_time, 'Strategy state': market_state})
        
        # Print current market state
        print(f"Current Strategy: {market_state}")
     
        """ 計算移動停利價格"""
        # Dynamically update trailing stop based on position
        if self.position and  len(self.lookback_prices) >= 2:  # Long position
            if self.trailing_stop is None or actual_price > self.lookback_prices[-2]['high']:
                self.trailing_stop = actual_price * (1 - self.trailing_stop_pct)

        elif self.position < 0 and len(self.lookback_prices) >= 2:  # Short position
            if self.trailing_stop is None or actual_price < self.lookback_prices[-2]['low']:
                self.trailing_stop = actual_price * (1 + self.trailing_stop_pct)

        """確認是否會出發移動停利"""
        # Check for trailing stop conditions
        if self.position > 0 and self.trailing_stop is not  None:  # Long trailing stop
            if self.entry_price <= actual_price <= self.trailing_stop:
                """即時價格需小於停利價格，且即時也得大於入場價格(確保目前是賺錢)"""
                print(f"Trigger Trailing Stop (Buy -> Sell) at {current_time}, Price: {actual_price}")
                self.trailing_stop_log.append({'Time':current_time,'price': actual_price,'trailing_stop_ptice':self.trailing_stop})
                self.execute_trade('sell', actual_price, current_time, contracts=self.position)

        elif self.position < 0 and self.trailing_stop is not  None:  # Short trailing stop
            """ 即時價格需大於停利價格，且即時價格也得小於放空價格(確保目前是賺錢)"""
            if self.entry_price >= actual_price >= self.trailing_stop: 
                print(f"Trigger Trailing Stop (Short -> Cover) at {current_time}, Price: {actual_price}")
                self.trailing_stop_log.append({'Time':current_time,'price': actual_price,'trailing_stop_price':self.trailing_stop})
                self.execute_trade('cover', actual_price, current_time, contracts=abs(self.position))

        """ 停損觸發機制"""
        # Check for stop loss conditions
        if self.position > 0:  # Long stop loss
            if (self.entry_price - actual_price) / self.entry_price >= self.stop_loss:
                print(f"Trigger Stop Loss (Buy -> Sell) at {current_time}, Price: {actual_price}")
                self.stop_loss_log.append({'Time':current_time,'price': actual_price,'entry_price':self.entry_price})
                self.execute_trade('sell', actual_price, current_time, contracts=self.position)

        elif self.position < 0:  # Short stop loss
            if (actual_price - self.entry_price) / self.entry_price >= self.stop_loss:
                print(f"Trigger Stop Loss (Short -> Cover) at {current_time}, Price: {actual_price}")
                self.stop_loss_log.append({'Time':current_time,'price': actual_price,'entry_price':self.entry_price})
                self.execute_trade('cover', actual_price, current_time, contracts=abs(self.position))

        """ 震盪策略停利機制"""
        if market_state == 'oscillation':
            # Check for trailing stop conditions
            if self.position > 0 and self.entry_price < actual_price:  # Long trailing stop
                if  abs(self.entry_price - actual_price) / self.entry_price > 0.0001 :
                    print(f"Trigger Trailing Stop for oscilliation (Buy -> Sell) at {current_time}, Price: {actual_price}")
                    self.trailing_stop_log.append({'Time':current_time,'price': actual_price,'trailing_stop_price':self.trailing_stop})
                    self.execute_trade('sell', actual_price, current_time, contracts=self.position)


            elif self.position < 0  and self.entry_price > actual_price:  # Short trailing stop
                if  abs(self.entry_price - actual_price) / self.entry_price > 0.0001 :
                    print(f"Trigger Trailing Stop for oscilliation (Short -> Cover) at {current_time}, Price: {actual_price}")
                    self.trailing_stop_log.append({'Time':current_time,'price': actual_price,'trailing_stop_price':self.trailing_stop})
                    self.execute_trade('cover', actual_price, current_time, contracts=abs(self.position))           
            
 

        # Apply strategy based on detected market state
        if market_state == 'oscillation':
            
            # Calculate oscillation strategy values from the window of high/low prices
            recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
            recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']  
                      
            self.apply_strategy(self.oscillation_strategy, predicted_return, actual_price, current_time, recent_high, recent_low)
            
        elif market_state == 'trend':
            
            # Calculate oscillation strategy values from the window of high/low prices
            recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
            recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']  
                      
            self.apply_strategy(self.trend_strategy, predicted_return, actual_price, current_time, buy_threshold, short_threshold
                                , recent_high, recent_low)
        
        elif market_state == 'default':
            
            self.apply_strategy(self.default_strategy, predicted_return, actual_price, current_time, buy_threshold, short_threshold)

        return self.get_current_state()


    def get_current_state(self):
        final_value = self.balance
        return {
            'return_rate': (final_value - self.initial_balance) / self.initial_balance,
            'current_balance': self.balance,
            'position': self.position,
            'trade_log': self.trade_log,
            'profit_loss_log': self.profit_loss_log
        }


    def plot_profit_loss(self, time, return_rate, actual, x_limit=None, forward_test=None):
        """
        繪製未來預測與交易標記，以及隨著時間變化的實現損益和綜合損益，並在下面的子圖右軸顯示綜合損益。
        """
        # 動態設定 X 軸範圍
        if x_limit is None:
            # 如果時間序列是空的，初始化為當前時間
            if time.empty or pd.isna(time.max()):
                upper_limit = pd.Timestamp.now() + pd.Timedelta(minutes=15)
                lower_limit = pd.Timestamp.now() - pd.Timedelta(hours=12)  # 假設顯示最近 12 小時
            else:
                # 使用時間序列的最大值作為上限，並添加緩衝
                upper_limit = time.max() + pd.Timedelta(minutes=15)
                lower_limit = time.min() - pd.Timedelta(minutes=15)  # 也可根據需求設置下限
            x_limit = (lower_limit, upper_limit)

        # 打印範圍以供檢查
        print(f"Dynamic X-axis limits set to: {x_limit}")

        if not self.profit_loss_log:
            print("Warning: No profit/loss data available.")
            profit_loss_df = pd.DataFrame({'time': time, 'profit_loss': [0] * len(time)})
        else:
            profit_loss_df = pd.DataFrame(self.profit_loss_log)
            if 'time' in profit_loss_df.columns and 'profit_loss' in profit_loss_df.columns:
                profit_loss_df['time'] = pd.to_datetime(profit_loss_df['time'])

                # 检查并处理重复的时间值
                if profit_loss_df['time'].duplicated().any():
                    print("Warning: Duplicate time values found in profit_loss_df. Removing duplicates...")
                    profit_loss_df = profit_loss_df.drop_duplicates(subset='time')

                # 设置索引并重新索引
                profit_loss_df = profit_loss_df.set_index('time').reindex(pd.Index(time).unique(), fill_value=0).reset_index()
                profit_loss_df.rename(columns={profit_loss_df.columns[0]: 'time'}, inplace=True)
                profit_loss_df = profit_loss_df[['time', 'profit_loss']]
            else:
                raise ValueError("Profit/Loss log must contain 'time' and 'profit_loss' columns.")

        # 计算累积损益
        profit_loss_df['cumulative_profit_loss'] = profit_loss_df['profit_loss'].cumsum().fillna(0)
        print("Cumulative Profit/Loss Data:")
        print(profit_loss_df[['time', 'cumulative_profit_loss']].head())

        # Create a time series for profit/loss, with missing times set to 0
        profit_loss_series = profit_loss_df.set_index('time')['profit_loss']



        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

        # Plot the candlestick chart on ax1
        # Convert actual prices to a DataFrame with open, high, low, close values for candlestick plotting
        ohlc_data = pd.DataFrame({
            'time': time,
            'open': actual['open'],
            'high': actual['high'],
            'low': actual['low'],
            'close': actual['close']
        })
        ohlc_data['time'] = pd.to_datetime(ohlc_data['time'])
        ohlc_data = ohlc_data.set_index('time')
        
        # Plot the candlestick chart
        import mplfinance as mpf
        mpf.plot(ohlc_data, type='candle', ax=ax1, style='charles', show_nontrading=True)

        # Plot trade markers (buy, sell, short, cover) with single label addition
        buy_marker = None
        sell_marker = None
        short_marker = None
        cover_marker = None

        for trade in self.trade_log:
            if trade['action'] == 'buy':
                buy_marker = ax1.scatter(trade['time'], trade['price'], color='#32cd32', marker='^', s=100, zorder=5)  # Buy signal (green)
            elif trade['action'] == 'sell':
                sell_marker = ax1.scatter(trade['time'], trade['price'], color='#ff6347', marker='v', s=100, zorder=5)  # Sell signal (red)
            elif trade['action'] == 'short':
                short_marker = ax1.scatter(trade['time'], trade['price'], color='#ff8c00', marker='s', s=100, zorder=5)  # Short signal (orange)
            elif trade['action'] == 'cover':
                cover_marker = ax1.scatter(trade['time'], trade['price'], color='#ffd700', marker='D', s=100, zorder=5)  # Cover signal (yellow)

        # Add legend for trade markers if any trades exist
        markers = [m for m in [buy_marker, sell_marker, short_marker, cover_marker] if m is not None]
        labels = ["Buy Signal", "Sell Signal", "Short Signal", "Cover Signal"][:len(markers)]
        if markers:
            ax1.legend(handles=markers, labels=labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Prices')
        ax1.set_title('Future Predictions vs Actual Future Values with Trade Markers', fontsize=16)
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        ax1.tick_params(axis='x', rotation=45, labelcolor='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xlim(x_limit)  # Set X-axis limit

        # Plot realized profit/loss over time on ax2
        ax2.bar(profit_loss_series.index, profit_loss_series.values,
                color=['#32cd32' if v > 0 else '#ff6347' for v in profit_loss_series.values], alpha=0.7, width=0.01)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Profit / Loss')
        ax2.set_title('Realized Profit/Loss Over Time', fontsize=16)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)
        ax2.tick_params(axis='x', rotation=45, labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_xlim(x_limit)  # Set X-axis limit

        # Create right y-axis for cumulative profit/loss on ax2
        ax5 = ax2.twinx()
        ax5.plot(profit_loss_df['time'], profit_loss_df['cumulative_profit_loss'], color='#8a2be2', alpha=0.9, zorder=3)  # Purple for cumulative profit/loss
        ax5.set_ylabel('Cumulative Profit/Loss', color='black')
        ax5.tick_params(axis='y', labelcolor='black')
        ax5.grid(axis='y', linestyle='--', alpha=0.6)

        # Plot predicted returns on ax3
        ax3.plot(time, return_rate, color='#ff4500', alpha=0.8, label='Predicted Returns')  # Use orange-red for predicted returns
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Predicted Returns')
        ax3.set_title('Predicted Returns Over Time', fontsize=16)
        ax3.grid(axis='y', linestyle='--', alpha=0.6)
        ax3.tick_params(axis='x', rotation=45, labelcolor='black')
        ax3.tick_params(axis='y', labelcolor='black')
        ax3.set_xlim(x_limit)  # Set X-axis limit

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

        # Real-time plotting update for forward testing
        if forward_test is not None:
            fig.canvas.draw()
            plt.pause(0.001)

    def monitor_stop_loss(self, live_price_data, osc_window, predicted_return, buy_threshold, short_threshold):
        """
        即時監控停損條件，並根據價格自動平倉。
        Paramter:
            live_price_data: 即時價格 (Dataframe)
            osc_window: 盤整窗口 (int)
            predicted_return: 五分鐘回報率 (int)
            
        """
        
        live_price = live_price_data['price'].iloc[-1]
        current_time = live_price_data.index[-1]
        
        # 將字符串格式時間轉為 datetime.time
        start_time = pd.to_datetime(self.trade_time["start_time"], format="%H:%M").time()
        end_time = pd.to_datetime(self.trade_time["end_time"], format="%H:%M").time()
        
        # 限制交易時間：9:00 AM 前和 1:30 PM 後
        if current_time.time() < start_time or current_time.time() > end_time:
            # 如果有持倉，強制平倉
            if self.position > 0:  # 平多單
                print(f"Force close long position at {current_time}, Price: {live_price}")
                self.execute_trade('sell', live_price, current_time, contracts=self.position)
            elif self.position < 0:  # 平空單
                print(f"Force close short position at {current_time}, Price: {live_price}")
                self.execute_trade('cover', live_price, current_time, contracts=abs(self.position))
            
            # 阻止任何交易行為，直接返回目前狀態
            return self.get_current_state()        
        
        if self.position > 0:  # 多頭持倉
            if (self.entry_price - live_price) / self.entry_price >= self.stop_loss:
                print(f"Stop loss triggered for long position at price {live_price}")
                self.stop_loss_log.append({'Time':current_time,'price': live_price,'entry_price':self.entry_price})
                self.execute_trade('sell', live_price, current_time, contracts=self.position)
                
            # Check for trailing stop conditions
            if self.trailing_stop is not  None:
                if self.entry_price <= live_price <= self.trailing_stop:
                    print(f"Trigger Trailing Stop (Buy -> Sell) at {current_time}, Price: {live_price}")
                    self.trailing_stop_log.append({'Time':current_time,'price': live_price,'trailing_stop_price':self.trailing_stop})
                    self.execute_trade('sell', live_price, current_time, contracts=self.position)
                        
                    
        elif self.position < 0:  # 空頭持倉
            if (live_price - self.entry_price) / self.entry_price >= self.stop_loss:
                print(f"Stop loss triggered for short position at price {live_price}")
                self.stop_loss_log.append({'Time':current_time,'price': live_price,'entry_price':self.entry_price})
                self.execute_trade('cover', live_price, current_time, contracts=abs(self.position))

            if self.trailing_stop is not  None:
                if  self.entry_price >= live_price >= self.trailing_stop:
                    print(f"Trigger Trailing Stop (Short -> Cover) at {current_time}, Price: {live_price}")
                    self.trailing_stop_log.append({'Time':current_time,'price': live_price,'trailing_stop_price':self.trailing_stop})
                    self.execute_trade('cover', live_price, current_time, contracts=abs(self.position))

        """ 震盪策略停利機制"""
        if self.strategy_status[-1]['Strategy state'] == 'oscillation':
            
            # Calculate oscillation strategy values from the window of high/low prices
            recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
            recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']
           
            #偵測是否突然突破以便隨時轉換狀態
            self.live_adjust_strategy(live_price, current_time, recent_high, recent_low, predicted_return)
                 
            # Check for trailing stop conditions
            if self.position > 0 and self.entry_price < live_price:  # Long trailing stop
                              
                if  abs(self.entry_price - live_price) / self.entry_price > 0.001 :
                    print(f"Trigger Trailing Stop for oscilliation (Buy -> Sell) at {current_time}, Price: {live_price}")
                    self.trailing_stop_log.append({'Time':current_time,'price': live_price,'trailing_stop_price':self.trailing_stop})
                    self.execute_trade('sell', live_price, current_time, contracts=self.position)


            elif self.position < 0  and self.entry_price > live_price:  # Short trailing stop
                                
                if  abs(self.entry_price - live_price) / self.entry_price > 0.001 :
                    
                    print(f"Trigger Trailing Stop for oscilliation (Short -> Cover) at {current_time}, Price: {live_price}")
                    self.trailing_stop_log.append({'Time':current_time,'price': live_price,'trailing_stop_price':self.trailing_stop})
                    self.execute_trade('cover', live_price, current_time, contracts=abs(self.position))           
                       
        if self.position == 0 and self.strategy_status[-1]['Strategy state'] == 'oscillation':
            # Calculate oscillation strategy values from the window of high/low prices
            recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
            recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']  
                      
            self.apply_strategy(self.oscillation_strategy, predicted_return, live_price, current_time, recent_high, recent_low) 

        # Apply strategy based on detected market state
        if self.strategy_status[-1]['Strategy state'] == 'oscillation':
            
            # Calculate oscillation strategy values from the window of high/low prices
            recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
            recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']  
                      
            self.apply_strategy(self.oscillation_strategy, predicted_return, live_price, current_time, recent_high, recent_low)
            
        elif self.strategy_status[-1]['Strategy state'] == 'trend':
            
            # Calculate oscillation strategy values from the window of high/low prices
            recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
            recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']  
                      
            self.apply_strategy(self.trend_strategy, predicted_return, live_price, current_time, buy_threshold, short_threshold, 
                                recent_high, recent_low)
        
        elif self.strategy_status[-1]['Strategy state'] == 'default':
            
            self.apply_strategy(self.default_strategy, predicted_return, live_price, current_time, buy_threshold, short_threshold)
         
        return self.get_current_state()
    
    #即時調製策略
    def live_adjust_strategy(self, live_price , current_time, recent_high, recent_low, predicted_return):
        
        """如果盤整趨勢被打破就切換成趨勢策略"""
        if  self.strategy_status[-1]['Strategy state'] == 'oscillation' and self.position ==0 :
            
            if self.profit_loss_log[-1]['profit_loss']:
                
                self.strategy_status.append({'Time': current_time, 'Strategy state': 'trend'})
                
                """搶突破"""
                self.apply_strategy(self.breakout_strategy, predicted_return, live_price, current_time, 
                                recent_high, recent_low)                
                
            elif recent_high < live_price or recent_low > live_price :
                self.strategy_status.append({'Time': current_time, 'Strategy state': 'trend'})
                
                """搶突破"""
                self.apply_strategy(self.breakout_strategy, predicted_return, live_price, current_time, 
                                recent_high, recent_low)
                
        elif recent_high < live_price or recent_low > live_price : 
            
            self.strategy_status.append({'Time': current_time, 'Strategy state': 'trend'})            

            
    def summary_table(self):
        """
        輸出總交易次數、總損益、總手續費、勝率、總平倉數量和賺賠比的摘要表。
        """
        total_trades = len(self.trade_log)
        total_profit_loss = sum(trade.get('profit_loss', 0) for trade in self.profit_loss_log if 'profit_loss' in trade) - self.total_transaction_fees
        total_fees = self.total_transaction_fees
        
        winning_trades = sum(1 for trade in self.profit_loss_log if 'profit_loss' in trade and trade['profit_loss'] > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_closed_positions = sum(1 for trade in self.trade_log if trade['action'] in ['sell', 'cover'])
        
        total_profit_points = sum(trade['profit_loss'] for trade in self.profit_loss_log if 'profit_loss' in trade and trade['profit_loss'] > 0)
        total_loss_points = abs(sum(trade['profit_loss'] for trade in self.profit_loss_log if 'profit_loss' in trade and trade['profit_loss'] < 0))
        profit_loss_ratio = (total_profit_points / total_loss_points) if total_loss_points > 0 else float('inf')
        
        summary_df = pd.DataFrame({
            'Metric': ['Total Trades', 'Total Profit/Loss (After Fees)', 'Total Transaction Fees', 'Win Rate (%)', 'Total Closed Positions', 'Profit/Loss Ratio'],
            'Value': [total_trades, total_profit_loss, total_fees, win_rate, total_closed_positions, profit_loss_ratio]
        })
        print(summary_df)

class RealtimeSpeedForceIndicators:
    def __init__(self, speed_window=5, force_window=5, range_window=5):
        """
        Initialize the strategy with specific rolling window sizes.
        :param speed_window: Window size for speed calculation.
        :param force_window: Window size for force calculation.
        :param range_window: Window size for smoothed range_diff calculation.
        """
        self.speed_window = speed_window
        self.force_window = force_window
        self.range_window = range_window
        self.data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'speed', 'force', 'smoothed_range_diff'])

    def add_new_data(self, new_data):
        """
        Add new data to the strategy and calculate indicators.
        :param new_data: A DataFrame containing the latest data row(s).
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Ensure new_data contains the required columns
        if not all(col in new_data.columns for col in required_columns):
            raise ValueError(f"Input data must contain columns: {required_columns}")

        # Select only required columns from new_data
        filtered_data = new_data[required_columns].copy()

        # Append the new filtered data
        self.data = pd.concat([self.data, filtered_data], ignore_index=True)

        # Calculate speed (close price difference)
        self.data['speed'] = self.data['close'].diff()

        # Calculate force (second derivative of volume)
        self.data['volume_diff'] = self.data['volume'].diff()
        self.data['force'] = self.data['volume_diff'].diff()

        # Calculate range_diff (high-low difference)
        self.data['range_diff'] = self.data['high'] - self.data['low']

        # Smooth range_diff using a rolling window
        self.data['smoothed_range_diff'] = (
            self.data['range_diff']
            .rolling(window=self.range_window, min_periods=1)
            .mean()
        )

        # Replace all NaN values with 0
        self.data.fillna(0, inplace=True)

    def get_latest_indicators(self):
        """
        Get the latest calculated indicators.
        :return: A dictionary containing the latest speed, force, and smoothed range_diff.
        
        speed > 0 and force > 0   and smoothed_range_diff < volatility_limit => Long
        
        Buy if the current price is closer to the retracement levels (e.g., 38.2% Fibonacci retracement).
        
        speed < 0 and force < 0   and smoothed_range_diff < volatility_limit => Short
        
        Short if the current price is closer to the retracement levels (e.g., 61.8% Fibonacci retracement)
        
        """
        if not self.data.empty:
            latest_row = self.data.iloc[-1]
            return {
                'speed': latest_row.get('speed', 0),
                'force': latest_row.get('force', 0),
                'smoothed_range_diff': latest_row.get('smoothed_range_diff', 0),
            }
        return {'speed': 0, 'force': 0, 'smoothed_range_diff': 0}

    
    