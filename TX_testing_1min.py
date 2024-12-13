#%%
import sys
import pandas as pd
from package.alpha_eric import AlphaFactory
from package.TAIndicator import TAIndicatorSettings
from package.scraping_and_indicators import StockDataScraper
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from filterpy.kalman import KalmanFilter
from ta import add_all_ta_features

data = pd.read_excel(r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API\TX00\TX00_5_20241101_20241209.xlsx")
data = data.set_index('date')


alpha = AlphaFactory(data)
indicator = TAIndicatorSettings()
indicator2 = StockDataScraper()

data_alpha = alpha.add_all_alphas()

filtered_settings, timeperiod_only_indicators = indicator.process_settings()  # 处理所有步骤并获取结果
# data_done1 = indicator2.add_indicator_with_timeperiods(data,timeperiod_only_indicators, timeperiods=[5, 10, 20, 50, 100, 200])
indicator_list = list(filtered_settings.keys())
data_done0 = indicator2.add_specified_indicators(data_alpha, indicator_list, filtered_settings)
data_done1 = add_all_ta_features(data_done0,open='open',high='high',low='low',close='close',volume='volume')

data_done2 = indicator2.add_indicator_with_timeperiods(data_done1,timeperiod_only_indicators, timeperiods=[5, 10, 20, 60, 120, 240])
print(data_done2.info)


#增加target variable
data_done2['predict_15min_return'] = (data_done2['close'].shift(-3)-data_done2['close'])/data_done2['close']

# Add year, month, day, weekday features
data_done2['day'] = data_done2.index.day
data_done2['minute'] = data_done2.index.minute
data_done2['hour'] = data_done2.index.hour



label ='predict_15min_return'

end_day, end_month = 15, 11
start_day, start_month = 15, 10
# 訓練數據和未來數據的切片
data_training = data_done2['2024-11-19':'2024-12-09']
data_future = data_done2['2024-12-09':'2024-12-09']

train_date = ['2024-11-10','2024-12-08']
test_date = ['2024-12-09','2024-12-09']
# 處理train set
# 處理train set

y = data_training[label]
X = data_training.drop(columns=[label])
X = X.replace([np.inf, -np.inf], np.nan)  # 将 inf 替换为 NaN，以便可以使用 dropna() 删除它们
X = X.dropna(axis=1, how='any')  # 删除包含 NaN 的列


# Apply Kalman filter to the target variable (avg_return_after_20_days)
def apply_kalman_filter(data):
    kf = KalmanFilter(dim_x=1, dim_z=1)  # 1D filter for a univariate time series

    # Initialize the state estimate and covariance matrix
    kf.x = np.array([[data.iloc[0]]])  # Initial state (using the first value)
    kf.P = np.array([[1000]])  # Initial uncertainty (you can adjust this)
    kf.F = np.array([[1]])  # State transition matrix (identity in this case)
    kf.H = np.array([[1]])  # Measurement function (direct observation)
    kf.R = np.array([[2]])  # Measurement noise covariance (adjustable)
    kf.Q = np.array([[1]])  # Process noise covariance (adjustable)

    smoothed_data = []
    
    # Loop through the data and apply the Kalman filter
    for i in range(len(data)):
        kf.predict()  # Predict the next state
        
        kf.update(data.iloc[i])  # Update the state with the observed value
        smoothed_data.append(kf.x[0, 0])  # Store the smoothed value

    return pd.Series(smoothed_data, index=data.index)

y_smoothed_kalman = apply_kalman_filter(y)

# Visualize the result
plt.figure(figsize=(10, 6))
plt.plot(y.tail(60), label="Original avg_return_after_20_days", color='blue')
plt.plot(y_smoothed_kalman.tail(60), label="Smoothed by Kalman Filter", color='orange')
plt.legend()
plt.title("Kalman Filter Smoothing of Target Variable: profit_or_loss_after_20_days_pp")
plt.show()

# y_smoothed_kalman = (((y_smoothed_kalman.shift(-3) - y_smoothed_kalman)/y_smoothed_kalman)).iloc[:-20]

# 使用切片來獲取 data.index 中最後 20 個時間戳，提供未來使用
future_time = pd.date_range(start='2024-11-20 08:50:00', end='2024-12-20 13:45:00', freq='5T') 

# 使用切片來保留不包括最後 20 行的索引
y_future = data_future[label]
y_actual = data_future['close']
X_future = data_future.drop(columns=[label])
X_future = X_future.replace([np.inf, -np.inf], np.nan) 
X_future = X_future.dropna(axis=1, how='any')  # 删除包含 NaN 的列
####################### 篩選資料 #######################################
 
# Calculate R^2 between each feature and target variable, and eliminate features with low R^2
r2_threshold = 0.1
selected_features = []

# for column in X.columns:
#     # 使用相同的索引来对齐 X 和 y，确保样本数量一致
#     common_index = X[column].dropna().index.intersection(y.dropna().index)
#     X_column_aligned = X[column].loc[common_index]
#     y_aligned = y.loc[common_index]

#     # 计算 R²
#     if len(y_aligned) > 0:  # 确保对齐后数据不为空
#         r2 = r2_score(y_aligned, X_column_aligned)
#         if abs(r2) >= r2_threshold:
#             selected_features.append(column)

# Select only the features with R² above the threshold

# X = X[selected_features]

# Feature Selection - Remove Low Variance Features
# Remove features with variance less than 0.05
selector = VarianceThreshold(threshold=0.05)
X_selected = selector.fit_transform(X)

# Record remaining features after removing low variance features
remaining_low_variance_features = X.columns[selector.get_support()].tolist()
X_future = X_future[remaining_low_variance_features]

# Convert X_selected back to DataFrame with correct column names
X_selected_df = X[remaining_low_variance_features]

# Remove Highly Correlated Features (correlation > 0.9)
corr_matrix = X_selected_df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
X_reduced = X_selected_df.drop(columns=to_drop)

# Record remaining features after removing highly correlated features
remaining_highly_correlated_features = X_reduced.columns.tolist()

# Update X_future with the final selected features
X_future = X_future[remaining_highly_correlated_features]


# Display remaining features with original column names
print(remaining_highly_correlated_features)

if y_smoothed_kalman.isna().any():  # 检查是否存在任何 NaN
    # 找出包含 NaN 的索引
    rows_with_nan_index = y_smoothed_kalman[y_smoothed_kalman.isna()].index

    # 删除包含 NaN 的行
    y_smoothed_kalman.dropna(inplace=True)

    # 基于包含 NaN 的行的索引，删除 X_reduced 中相应的行
    X_reduced.drop(index=rows_with_nan_index, inplace=True)


# # 將 list 轉換為 DataFrame
df = pd.DataFrame(remaining_highly_correlated_features, columns=['Highly Correlated Features'])

# # 儲存為 Excel
df.to_excel('highly_correlated_features.xlsx', index=False)
# Split Training and Testing Data
# Use rows -420 to -91 for the training set, and the last 90 rows as the test set


X_train = X_reduced[train_date[0]:train_date[1]]
y_train = y_smoothed_kalman[train_date[0]:train_date[1]]
X_test = X_reduced[test_date[0]:test_date[1]]
y_test = y_smoothed_kalman[test_date[0]:test_date[1]]
# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
import joblib
from sklearn.linear_model import LinearRegression

torch.manual_seed(42)
np.random.seed(42)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Residual LSTM model
# Define the Residual LSTM model
class ResidualLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(ResidualLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()  # Add ReLU activation

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size).to(x.device)

        out1, _ = self.lstm1(x.unsqueeze(1) if x.ndim == 2 else x, (h_0, c_0))
        
        h_0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(x.device)
        c_0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(x.device)
        out2, _ = self.lstm2(out1, (h_0_2, c_0_2))

        # Residual connection
        residual = x if x.ndim == 2 else x[:, -1, :]
        if residual.size(1) > out2.size(2):
            residual = residual[:, :out2.size(2)]  # Trim residual dimensions to match LSTM output dimensions
        elif residual.size(1) < out2.size(2):
            padding = torch.zeros((batch_size, out2.size(2) - residual.size(1))).to(x.device)
            residual = torch.cat((residual, padding), dim=1)  # Pad residual dimensions to match LSTM output dimensions
        out = out2[:, -1, :] + residual  # Add input as residual
        
        out = self.dropout(out)
        out = self.fc(out)
        out = self.relu(out)  # Apply ReLU activation
        return out

# Data preprocessing
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_future_scaled = scaler_X.transform(X_future)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Convert the data to tabular format (2D)
X_train_tab = X_train_scaled
X_test_tab = X_test_scaled
X_future_tab = X_future_scaled

# Train the Residual LSTM model
hidden_size = 512
num_layers = 3
learning_rate = 0.005

residual_lstm_model = ResidualLSTMModel(X_train_tab.shape[1], hidden_size, 1, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(residual_lstm_model.parameters(), lr=learning_rate)

X_train_tensor = torch.tensor(X_train_tab, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

epochs = 600
for epoch in range(epochs):
    residual_lstm_model.train()
    optimizer.zero_grad()
    outputs = residual_lstm_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Residual LSTM Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Make predictions using the Residual LSTM model
residual_lstm_model.eval()

X_test_tensor = torch.tensor(X_test_tab, dtype=torch.float32).to(device)
X_future_tensor = torch.tensor(X_future_tab, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_lstm_scaled = residual_lstm_model(X_test_tensor).cpu().numpy()
    y_future_lstm_scaled = residual_lstm_model(X_future_tensor).cpu().numpy()

# Rescale LSTM predictions
y_pred_lstm_rescaled = scaler_y.inverse_transform(y_pred_lstm_scaled)
y_future_lstm_rescaled = scaler_y.inverse_transform(y_future_lstm_scaled)

# Evaluate the Residual LSTM model's performance
rmse_lstm = mean_squared_error(y_test, y_pred_lstm_rescaled, squared=False)
r2_lstm = r2_score(y_test, y_pred_lstm_rescaled)
print(f'Residual LSTM RMSE: {rmse_lstm:.4f}, R^2: {r2_lstm:.4f}')


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(2).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(2).to(device)



# Residual TCN Model (保持不变)
class ResidualTCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size=3, num_layers=3, dropout_rate=0.2):
        super(ResidualTCNModel, self).__init__()
        self.tcn_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        for i in range(num_layers):
            self.tcn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_size if i == 0 else hidden_size,
                        out_channels=hidden_size,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) * (2**i) // 2,
                        dilation=2**i
                    ),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(inplace=False)
                )
            )
            if i == 0 or input_size != hidden_size:
                self.residual_layers.append(
                    nn.Conv1d(
                        in_channels=input_size if i == 0 else hidden_size,
                        out_channels=hidden_size,
                        kernel_size=1
                    )
                )
            else:
                self.residual_layers.append(None)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        for i, tcn_layer in enumerate(self.tcn_layers):
            residual = x
            x = tcn_layer(x)
            if self.residual_layers[i] is not None:
                residual = self.residual_layers[i](residual)
            x = x + residual

        x = x[:, :, -1]
        x = self.fc(x)
        return x


# Training TCN
hidden_size_tcn = 128
kernel_size = 3
num_layers_tcn = 2
dropout_rate_tcn = 0.2
epochs_tcn = 200
learning_rate_tcn = 0.001

residual_tcn_model = ResidualTCNModel(1, hidden_size_tcn, 1, kernel_size, num_layers_tcn, dropout_rate_tcn).to(device)
optimizer_tcn = optim.Adam(residual_tcn_model.parameters(), lr=learning_rate_tcn)

for epoch in range(epochs_tcn):
    residual_tcn_model.train()
    optimizer_tcn.zero_grad()
    outputs = residual_tcn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer_tcn.step()
    if (epoch + 1) % 10 == 0:
        print(f"Residual TCN Epoch [{epoch + 1}/{epochs_tcn}], Loss: {loss.item():.4f}")

# TCN predictions
residual_tcn_model.eval()
with torch.no_grad():
    y_pred_tcn_scaled = residual_tcn_model(X_test_tensor).cpu().numpy()

y_pred_tcn_rescaled = scaler_y.inverse_transform(y_pred_tcn_scaled)

# Evaluate the Residual LSTM model's performance
rmse_tcn = mean_squared_error(y_test, y_pred_tcn_rescaled, squared=False)
r2_tcn= r2_score(y_test, y_pred_tcn_rescaled)

# Ensemble Model
# Prepare predictions from LSTM and LightGBM as features
ensemble_features_train = np.hstack((y_pred_lstm_rescaled, y_pred_tcn_rescaled))
ensemble_model = LinearRegression()
ensemble_model.fit(ensemble_features_train, y_test)

y_pred_ensemble_rescaled = ensemble_model.predict(ensemble_features_train)

# Evaluate the Ensemble model's performance
rmse_ensemble = mean_squared_error(y_test, y_pred_ensemble_rescaled, squared=False)
r2_ensemble = r2_score(y_test, y_pred_ensemble_rescaled)

# Print all model results
print(f'Root Mean Squared Error (RMSE) for LSTM: {rmse_lstm}')
print(f'R^2 Score for LSTM: {r2_lstm}')
print(f'Root Mean Squared Error (RMSE) for TCN: {rmse_tcn}')
print(f'R^2 Score for TCN: {r2_tcn}')
print(f'Root Mean Squared Error (RMSE) for Ensemble: {rmse_ensemble}')
print(f'R^2 Score for Ensemble: {r2_ensemble}')

# Plotting
plt.figure(figsize=(14, 8))
plt.plot(y_test.index, y_test, label="Actual Values", color="blue")
plt.plot(y_test.index, y_pred_lstm_rescaled, label="LSTM Predictions", color="orange")
plt.plot(y_test.index, y_pred_tcn_rescaled, label="TCN Predictions", color="green")
plt.plot(y_test.index, y_pred_ensemble_rescaled, label="Ensemble Predictions", color="red")
plt.xlabel("Index")
plt.ylabel("Prices")
plt.title("LSTM, TCN, and Ensemble Model Predictions")
plt.legend()
plt.show()
# %%

import dill
# Save models and scalers using dill (ignoring objects that cannot be pickled)
import dill

# Save the entire ensemble model
with open('TX_1min_model_5.pkl', 'wb') as f: 
    data_to_save = {
        'lstm_model': residual_lstm_model,
        'lstm_optimizer_state_dict': optimizer.state_dict(),
        'tcn_model': residual_tcn_model,
        'ensemble_model': ensemble_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    dill.dump(data_to_save, f)

# 检查保存的数据的键
print(f"保存的数据的键: {data_to_save.keys()}")

# Load the model


# # 提取各個物件
# lstm_model = saved_objects['lstm_model']
# lstm_optimizer_state_dict = saved_objects['lstm_optimizer_state_dict']
# lgbm_model = saved_objects['lgbm_model']
# ensemble_model = saved_objects['ensemble_model']
# scaler_X = saved_objects['scaler_X']
# scaler_y = saved_objects['scaler_y']

# # 如果需要恢復 LSTM 優化器的狀態：
# # lstm_optimizer.load_state_dict(lstm_optimizer_state_dict)




#%%
from datetime import time
class Backtest:
    def __init__(self, initial_balance=100000, transaction_fee=1, margin_rate=0.1, stop_loss=0.05, 
                 trailing_stop_pct=0.02, point_value=1):
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
        self.entry_price = None
        self.entry_time = None
        self.profit_loss_log = []
        self.total_transaction_fees = 0
        self.point_value = point_value
        self.trailing_stop_log,  self.stop_loss_log, = [], []

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
            self.profit_loss_log.append({'time': time, 'profit_loss': profit_loss})
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
            self.profit_loss_log.append({'time': time, 'profit_loss': profit_loss})
            self.position = 0
            self.entry_price = None
            self.entry_time = None
            self.trailing_stop = None
        # 只有在平倉後才記錄資產價值
        if action in ['sell', 'cover']:
            self.portfolio_values.append(self.balance)

    def run_backtest(self, predicted_returns, actual_prices, buy_threshold=0.0008, short_threshold=-0.005,
                     no_trade_before_9am=True, oscillation_range=0.01, trend_lookback=6, breakout_lookback=6):
        """
        根据預測回報率和多種策略進行回測。
        
        参數:
        predicted_returns (pd.Series): 模型預測回報率
        actual_prices (pd.Series): 参考实际價格
        """
        for i in range(len(predicted_returns)):
            # 跳過早上9點前的交易
            if no_trade_before_9am and actual_prices.index[i].time() < time(9, 0):
                continue

            current_price = actual_prices.iloc[i]
            current_time = actual_prices.index[i]

            # 候有付惠: 检查持仓状态（多头或空头）
            if self.position > 0:  # 多头持仓
                if self.trailing_stop is None or current_price > self.trailing_stop:
                    self.trailing_stop = current_price * (1 - self.trailing_stop_pct)
                if (self.entry_price - current_price) / self.entry_price >= self.stop_loss:
                    self.execute_trade('sell', current_price, current_time)
                    continue
                if current_price <= self.trailing_stop:
                    self.execute_trade('sell', current_price, current_time)
                    continue
            elif self.position < 0:  # 空头持仓
                if self.trailing_stop is None or current_price < self.trailing_stop:
                    self.trailing_stop = current_price * (1 + self.trailing_stop_pct)
                if (current_price - self.entry_price) / self.entry_price >= self.stop_loss:
                    self.execute_trade('cover', current_price, current_time)
                    continue
                if current_price >= self.trailing_stop:
                    self.execute_trade('cover', current_price, current_time)
                    continue

            # 自動檢測市場狀態並應用適合的策略
            market_state = self.detect_market_state(i, actual_prices, trend_lookback, breakout_lookback)
            print(f"Market State at {current_time}: {market_state}")

            if market_state == 'oscillation':
                self.apply_oscillation_strategy(i, predicted_returns, actual_prices, current_price, current_time, oscillation_range)
            elif market_state == 'trend':
                self.apply_trend_strategy(i, predicted_returns, actual_prices, current_price, current_time, buy_threshold, short_threshold, trend_lookback)
            elif market_state == 'breakout':
                self.apply_breakout_strategy(i, predicted_returns, actual_prices, current_price, current_time, buy_threshold, short_threshold, breakout_lookback)

        # 如果在結束時還有持倉，則平倉
        if self.position > 0:
            self.execute_trade('sell', actual_prices.iloc[-1], actual_prices.index[-1])
        elif self.position < 0:
            self.execute_trade('cover', actual_prices.iloc[-1], actual_prices.index[-1])

        # 計算最終資產價值
        final_value = self.balance
        return_rate = (final_value - self.initial_balance) / self.initial_balance
        return return_rate, self.trade_log, self.portfolio_values, self.profit_loss_log

    def detect_market_state(self, i, actual_prices, trend_lookback, breakout_lookback):
        """
        自動檢測市場狀態：震盪、趨勢或突破。
        """
        recent_high = actual_prices.iloc[max(0, i - trend_lookback):i].max()
        recent_low = actual_prices.iloc[max(0, i - trend_lookback):i].min()
        avg_price = actual_prices.iloc[max(0, i - trend_lookback):i].mean()

        if recent_low < actual_prices.iloc[i] < recent_high:
            return 'oscillation'
        elif abs(actual_prices.iloc[i] - avg_price) / avg_price > 0.02:
            return 'trend'
        elif actual_prices.iloc[i] > recent_high or actual_prices.iloc[i] < recent_low:
            return 'breakout'
        return 'unknown'

    def apply_oscillation_strategy(self, i, predicted_returns, actual_prices, current_price, current_time, oscillation_range):
        """
        應用震盪策略。
        """
        recent_high = actual_prices.iloc[max(0, i - 10):i].max()
        recent_low = actual_prices.iloc[max(0, i - 10):i].min()

        if predicted_returns.iloc[i] > 0 and current_price < (recent_low + oscillation_range):
            self.execute_trade('buy', current_price, current_time)
        elif predicted_returns.iloc[i] < 0 and current_price > (recent_high - oscillation_range):
            self.execute_trade('short', current_price, current_time)

    def apply_trend_strategy(self, i, predicted_returns, actual_prices, current_price, current_time, buy_threshold, short_threshold, trend_lookback):
        """
        應用趨勢策略。
        """
        avg_price = actual_prices.iloc[max(0, i - trend_lookback):i].mean()

        if predicted_returns.iloc[i] > buy_threshold and current_price > avg_price:
            if self.position == 0:
                self.execute_trade('buy', current_price, current_time)
        elif predicted_returns.iloc[i] < short_threshold and current_price < avg_price:
            if self.position == 0:
                self.execute_trade('short', current_price, current_time)

    def apply_breakout_strategy(self, i, predicted_returns, actual_prices, current_price, current_time, buy_threshold, short_threshold, breakout_lookback):
        """
        應用突破策略。
        """
        breakout_high = actual_prices.iloc[max(0, i - breakout_lookback):i].max()
        breakout_low = actual_prices.iloc[max(0, i - breakout_lookback):i].min()

        if current_price > breakout_high and predicted_returns.iloc[i] > buy_threshold:
            if self.position == 0:
                self.execute_trade('buy', current_price, current_time)
        elif current_price < breakout_low and predicted_returns.iloc[i] < short_threshold:
            if self.position == 0:
                self.execute_trade('short', current_price, current_time)

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




    def plot_profit_loss(self, time, return_rate, actual, x_limit=None, default_start_time='09:00:00', default_end_time='13:30:00', forward_test=None):
        """
        繪製未來預測與交易標記，以及隨著時間變化的實現損益和綜合損益，並在下面的子圖右軸顯示綜合損益。
        """
        # Default X-axis limit from user-provided or default start and end times
        if x_limit is None:
            if pd.isna(time.min()):
                date_part = pd.Timestamp.now().date()
            else:
                date_part = time.min().date()
            x_limit = (
                pd.Timestamp.combine(date_part, pd.to_datetime(default_start_time).time()) - pd.Timedelta(minutes=15),
                pd.Timestamp.combine(date_part, pd.to_datetime(default_end_time).time()) + pd.Timedelta(minutes=15)
            )

        # Handle empty profit_loss_log
        if not self.profit_loss_log:
            print("Warning: No profit/loss data available.")
            # Create an empty DataFrame with the correct time range
            profit_loss_df = pd.DataFrame({'time': pd.date_range(start=x_limit[0], end=x_limit[1], freq='T'), 'profit_loss': 0})
        else:
            profit_loss_df = pd.DataFrame(self.profit_loss_log)
            start_time = profit_loss_df['time'].min()
            end_time = profit_loss_df['time'].max()
            all_times = pd.date_range(start=start_time, end=end_time, freq='T')
            profit_loss_df = profit_loss_df.set_index('time').reindex(all_times).fillna(0).reset_index()
            profit_loss_df.columns = ['time', 'profit_loss']

        # Calculate cumulative profit/loss, ensuring it starts from zero
        profit_loss_df['cumulative_profit_loss'] = profit_loss_df['profit_loss'].cumsum().fillna(0)

        # Debugging: Print the contents of profit_loss_df for cumulative profit/loss
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








# residual = (y_future_ensemble_rescaled.flatten()[0]-y_actual[0])/2
residual =0
# 計算預測回報率

predicted_returns = pd.Series(y_pred_lstm_rescaled.flatten())

# 使用保證金、停損、移動停利以及多/空倉進行回測
backtest = Backtest(initial_balance=200000, transaction_fee=30, margin_rate=0.1, stop_loss=0.001, trailing_stop_pct=0.0001, point_value=50)

return_rate, trade_log, portfolio_values, profit_loss_log = backtest.run_backtest(predicted_returns, 
                                                                                  data_training['close'][test_date[0]:test_date[1]][:-3],
                                                                                  buy_threshold=0.00001,
                                                                                  short_threshold=-0.00005,
                                                                                  no_trade_before_9am=True)

# 輸出回測結果
print(f'Final Return Rate: {return_rate * 100:.2f}%')
print('Trade Log:')
for trade in trade_log:
    print(trade)


# 打印损益日志
print("\nProfit/Loss Log:")
for record in backtest.profit_loss_log:
    print(record)

# 繪製未來預測與買入/賣出標記以及損益圖
backtest.plot_profit_loss(time=X_test.index,
                          return_rate=predicted_returns.values,
                          actual=data_training[['open','high','low','close','volume']][test_date[0]:test_date[1]][:-3])

# 輸出摘要表
backtest.summary_table()


from back_testing import Backtest


# %%
# %%
from datetime import time
import os
import json

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
        self.entry_price = None
        self.entry_time = None
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
        self.entry_price = data.get("entry_price", None)  # 初始化 entry_price
        self.trailing_stop = data.get("trailing_stop", None)  # 初始化 trailing_stop
        self.stop_loss = data.get("stop_loss", self.stop_loss)  # 初始化 stop_loss
        self.entry_time = data.get("entry_time", None)  # 初始化 entry_time
        self.trade_log = data.get("trade_log", [])
        self.profit_loss_log = data.get("profit_loss_log", [])
        self.stop_loss_log = data.get("stop_loss_log", [])  # 初始化 stop_loss_log
        print(f"Initialized from saved data: Balance={self.balance}, Position={self.position}")

    def save_trading_files(self):
        """
        儲存交易相關檔案到指定資料夾，並根據標的和日期命名。
        """
        # 確保資料夾存在
        latest_time = "unknown_date"
        if self.trade_log:
            latest_time = max(
                pd.to_datetime(entry['time']) for entry in self.trade_log if 'time' in entry
            ).strftime("%Y%m%d")
        subfolder = f"{self.symbol}_{latest_time}"
        target_folder = os.path.join(self.data_folder, subfolder)
        os.makedirs(target_folder, exist_ok=True)

        # 儲存交易明細
        if self.trade_log:
            trade_log_df = pd.DataFrame(self.trade_log)
            trade_log_df.to_json(os.path.join(target_folder, "trade_log.json"), orient="records", indent=4)
            trade_log_df.to_excel(os.path.join(target_folder, "trade_log.xlsx"), index=False)

        # 儲存損益紀錄
        if self.profit_loss_log:
            profit_loss_log_df = pd.DataFrame(self.profit_loss_log)
            profit_loss_log_df.to_json(os.path.join(target_folder, "profit_loss_log.json"), orient="records", indent=4)
            profit_loss_log_df.to_excel(os.path.join(target_folder, "profit_loss_log.xlsx"), index=False)

        # 儲存停損紀錄
        if self.stop_loss_log:
            stop_loss_log_df = pd.DataFrame(self.stop_loss_log)
            stop_loss_log_df.to_json(os.path.join(target_folder, "stop_loss_log.json"), orient="records", indent=4)
            stop_loss_log_df.to_excel(os.path.join(target_folder, "stop_loss_log.xlsx"), index=False)

        # 儲存當前狀態，包括更多參數
        status = {
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,  # 新增 entry_price
            "trailing_stop": self.trailing_stop,  # 新增 trailing_stop
            "stop_loss": self.stop_loss,  # 新增 stop_loss
            "entry_time": self.entry_time  # 新增交易時間參數
        }
        with open(os.path.join(target_folder, "status.json"), 'w') as f:
            json.dump(self._prepare_serializable(status), f, indent=4)

        print(f"Trading data saved to folder: {target_folder}")

    def load_trading_files(self):
        """
        從指定資料夾讀取交易相關檔案。
        :return: 包含交易相關數據的字典。
        """
        data = {}

        # 指定資料夾內最新的子資料夾
        if not os.path.exists(self.data_folder):
            print(f"Data folder {self.data_folder} does not exist.")
            return data

        subfolders = [f.path for f in os.scandir(self.data_folder) if f.is_dir()]
        if not subfolders:
            print(f"No subfolders found in {self.data_folder}.")
            return data

        latest_folder = max(subfolders, key=os.path.getmtime)

        # 讀取交易明細
        trade_log_path = os.path.join(latest_folder, "trade_log.json")
        if os.path.exists(trade_log_path) and os.path.getsize(trade_log_path) > 0:
            with open(trade_log_path, 'r') as f:
                data["trade_log"] = json.load(f)

        # 讀取損益紀錄
        profit_loss_log_path = os.path.join(latest_folder, "profit_loss_log.json")
        if os.path.exists(profit_loss_log_path) and os.path.getsize(profit_loss_log_path) > 0:
            with open(profit_loss_log_path, 'r') as f:
                data["profit_loss_log"] = json.load(f)

        # 讀取停損紀錄
        stop_loss_log_path = os.path.join(latest_folder, "stop_loss_log.json")
        if os.path.exists(stop_loss_log_path) and os.path.getsize(stop_loss_log_path) > 0:
            with open(stop_loss_log_path, 'r') as f:
                data["stop_loss_log"] = json.load(f)

        # 讀取當前狀態
        status_path = os.path.join(latest_folder, "status.json")
        if os.path.exists(status_path) and os.path.getsize(status_path) > 0:
            with open(status_path, 'r') as f:
                status_data = json.load(f)
                data.update(status_data)  # 合併狀態數據

        print(f"Trading data loaded from folder: {latest_folder}")
        return data

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
            self.profit_loss_log.append({'time': time, 'profit_loss': profit_loss})
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
            self.profit_loss_log.append({'time': time, 'profit_loss': profit_loss})
            self.position = 0
            self.entry_price = None
            self.entry_time = None
            self.trailing_stop = None
        if action in ['sell', 'cover']:
            self.portfolio_values.append(self.balance)

    def detect_market_state(self, predicted_return, actual_price, threshold=0.02, osc_window=5):
        if len(self.lookback_prices) < osc_window:
            return 'unknown'

        recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
        recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']

        if recent_low <= actual_price <= recent_high:
            return 'oscillation'
        elif actual_price > recent_high or actual_price < recent_low:
            return 'trend'
        return 'unknown'

    def apply_strategy(self, strategy_func, *args, **kwargs):
        strategy_func(*args, **kwargs)

    def oscillation_strategy(self, predicted_return, actual_price, current_time, recent_high, recent_low):
        if predicted_return > 0 and actual_price < recent_low * (1 + 0.002):
            self.execute_trade('buy', actual_price, current_time, contracts=1)
        elif predicted_return < 0 and actual_price > recent_high * (1 - 0.002):
            self.execute_trade('short', actual_price, current_time, contracts=1)

    def trend_strategy(self, predicted_return, actual_price, current_time, buy_threshold, short_threshold, recent_high, recent_low):
        
        if predicted_return > buy_threshold and (recent_high + recent_low) * 0.333 > actual_price:
            if self.position == 0:
                self.execute_trade('buy', actual_price, current_time, contracts=1)
        elif predicted_return < short_threshold and (recent_high + recent_low) * 0.333 < actual_price:
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
        if self.position > 0:  # Long position
            if self.trailing_stop is None or actual_price > self.lookback_prices[-2]['high']:
                self.trailing_stop = actual_price * (1 - self.trailing_stop_pct)

        elif self.position < 0:  # Short position
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

        # 繪圖邏輯（僅占位，應替換為實際的 Matplotlib 或其他繪圖程式碼）
        # fig, ax = plt.subplots()
        # ax.set_xlim(x_limit)
        # plt.show()


        # Handle empty profit_loss_log
        if not self.profit_loss_log:
            print("Warning: No profit/loss data available.")
            # Create an empty DataFrame with the correct time range
            profit_loss_df = pd.DataFrame({'time': time, 'profit_loss': [0] * len(time)})
        else:
            profit_loss_df = pd.DataFrame(self.profit_loss_log)
            if 'time' in profit_loss_df.columns:
                profit_loss_df['time'] = pd.to_datetime(profit_loss_df['time'])
                profit_loss_df = profit_loss_df.set_index('time').reindex(time, fill_value=0).reset_index()
                profit_loss_df.columns = ['time', 'profit_loss']
            else:
                raise ValueError("Profit/Loss log must contain a 'time' column.")

        # Calculate cumulative profit/loss, ensuring it starts from zero
        profit_loss_df['cumulative_profit_loss'] = profit_loss_df['profit_loss'].cumsum().fillna(0)

        # Debugging: Print the contents of profit_loss_df for cumulative profit/loss
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

    def monitor_stop_loss(self, live_price_data, osc_window, predicted_return):
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
                self.execute_trade('sell', live_price, pd.Timestamp.now(), contracts=self.position)
                
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
                self.execute_trade('cover', live_price, pd.Timestamp.now(), contracts=abs(self.position))

            if self.trailing_stop is not  None:
                if  self.entry_price >= live_price >= self.trailing_stop:
                    print(f"Trigger Trailing Stop (Short -> Cover) at {current_time}, Price: {live_price}")
                    self.trailing_stop_log.append({'Time':current_time,'price': live_price,'trailing_stop_price':self.trailing_stop})
                    self.execute_trade('cover', live_price, current_time, contracts=abs(self.position))

        """ 震盪策略停利機制"""
        if self.strategy_status[-1]['Strategy state'] == 'oscillation':
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










#%%
predicted_returns = pd.Series(y_pred_lstm_rescaled.flatten())
forward_test = ForwardTest(initial_balance=200000, transaction_fee=30, margin_rate=0.1, stop_loss=0.001, trailing_stop_pct=0.003, point_value=50, data_folder = "trading_data", symbol='TX00')
status_indicator  = RealtimeSpeedForceStrategy(speed_window=3, force_window=3, range_window=3)
                    
for cc in range(50):
    predicted_return = predicted_returns.iloc[cc]
    actual_price = data_training[['open', 'high', 'low', 'close', 'volume']][test_date[0]:test_date[1]].iloc[cc]
    actual_price_series = data_training[['open', 'high', 'low', 'close', 'volume']][test_date[0]:test_date[1]].iloc[0:cc+1]
    status_indicator.add_new_data(actual_price_series)
    
    status_indicator_done = status_indicator.data
    
    current_time = data_training['close'][test_date[0]:test_date[1]].index[cc]

    state = forward_test.run_backtest(
        predicted_return=predicted_return,
        real_time_data=actual_price,
        current_time=current_time,
        buy_threshold=0.00025,
        osc_window = 6,
        short_threshold=-0.0005
    )

    print(f"Step {cc + 1}")
    # print(f"Return Rate: {state['return_rate'] * 100:.2f}%")
    # print(f"Balance: {state['current_balance']:.2f}")
    print(f"Position: {state['position']}")
    print("Trade Log:")
    for trade in state['trade_log']:
        print(trade)

    forward_test.plot_profit_loss(
        time=data_training['close'][test_date[0]:test_date[1]].index[:cc + 1],
        return_rate=predicted_returns.iloc[:cc + 1],
        actual=data_training[['open', 'high', 'low', 'close', 'volume']][test_date[0]:test_date[1]].iloc[:cc + 1],
        forward_test=True
    )

forward_test.summary_table()





# %%
