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

data = pd.read_excel(r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API\CME,MNQ0000\CME,MNQ0000_5_20240315_20241124.xlsx")
data = data.set_index('date')


alpha = AlphaFactory(data)
indicator = TAIndicatorSettings()
indicator2 = StockDataScraper()

data_alpha = alpha.add_all_alphas()

filtered_settings, timeperiod_only_indicators = indicator.process_settings()  # 处理所有步骤并获取结果
data_done1 = indicator2.add_indicator_with_timeperiods(data_alpha,timeperiod_only_indicators, timeperiods=[5, 10, 20, 50, 100, 200])
indicator_list = list(filtered_settings.keys())
data_done2 = indicator2.add_specified_indicators(data_alpha, indicator_list, filtered_settings)
data_done2 = add_all_ta_features(data_done2,open='open',high='high',low='low',close='close',volume='volume')

data_done2['predict_15min'] = data_done2['close'].shift(-3)
#%%
label ='predict_15min'
predict_time = 100
X = data_done2.drop(columns=[label])

y = data_done2[label].iloc[:-predict_time]

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
    kf.Q = np.array([[0.5]])  # Process noise covariance (adjustable)

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
plt.plot(y, label="Original avg_return_after_20_days", color='blue')
plt.plot(y_smoothed_kalman, label="Smoothed by Kalman Filter", color='orange')
plt.legend()
plt.title("Kalman Filter Smoothing of Target Variable: profit_or_loss_after_20_days_pp")
plt.show()

# y_smoothed_kalman = (((y_smoothed_kalman.shift(-3) - y_smoothed_kalman)/y_smoothed_kalman)).iloc[:-20]


# Create Date Features
# Add year, month, day, weekday features
X['day'] = X.index.day
X['minute'] = X.index.minute
X['hour'] = X.index.hour

# Create Lag Features


# 使用切片來獲取 data.index 中最後 20 個時間戳，提供未來使用
future_time = data.index[-predict_time:]

# 使用切片來保留不包括最後 20 行的索引
X_future = X.tail(predict_time)
X = X.iloc[:-predict_time]

####################### 篩選資料 #######################################
 
# Calculate R^2 between each feature and target variable, and eliminate features with low R^2
r2_threshold = 0.1
selected_features = []

for column in X.columns:
    # 使用相同的索引来对齐 X 和 y，确保样本数量一致
    common_index = X[column].dropna().index.intersection(y.dropna().index)
    X_column_aligned = X[column].loc[common_index]
    y_aligned = y.loc[common_index]

    # 计算 R²
    if len(y_aligned) > 0:  # 确保对齐后数据不为空
        r2 = r2_score(y_aligned, X_column_aligned)
        if abs(r2) >= r2_threshold:
            selected_features.append(column)

# Select only the features with R² above the threshold

X = X[selected_features]

# Feature Selection - Remove Low Variance Features
# Remove features with variance less than 0.05
selector = VarianceThreshold(threshold=0.05)
X_selected = selector.fit_transform(X)

# Record remaining features after removing low variance features
remaining_low_variance_features = X.columns[selector.get_support()].tolist()
X_future = X_future[remaining_low_variance_features]

# Convert X_selected back to DataFrame with correct column names
X_selected_df = pd.DataFrame(X_selected, columns=remaining_low_variance_features)

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
# # 將 list 轉換為 DataFrame
# df = pd.DataFrame(remaining_highly_correlated_features, columns=['Highly Correlated Features'])

# # 儲存為 Excel
# df.to_excel('highly_correlated_features.xlsx', index=False)
# Split Training and Testing Data
# Use rows -420 to -91 for the training set, and the last 90 rows as the test set

test_interval = 200
train_time_lenght = 1200
X_train = X_reduced[-(train_time_lenght + test_interval):-test_interval]
y_train = y_smoothed_kalman.iloc[-(train_time_lenght + test_interval):-test_interval].reset_index(drop=True)
X_test = X_reduced[-test_interval:]
y_test = y_smoothed_kalman.iloc[-test_interval:].reset_index(drop=True)

#%%

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
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.linear_model import LinearRegression

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Residual LSTM model
class ResidualLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(ResidualLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

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
learning_rate = 0.0005

residual_lstm_model = ResidualLSTMModel(X_train_tab.shape[1], hidden_size, 1, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(residual_lstm_model.parameters(), lr=learning_rate)

X_train_tensor = torch.tensor(X_train_tab, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

epochs = 800
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

# Define LightGBM model with specific parameters
from lightgbm import LGBMRegressor
lgbm_model = LGBMRegressor(boosting_type='gbdt', learning_rate=0.01, max_depth=10, n_estimators=100, num_leaves=15)

# Train the LightGBM model on the entire training set
lgbm_model.fit(X_train_tab, y_train_scaled.ravel())

# Make predictions using the LightGBM model
y_pred_lgbm_scaled = lgbm_model.predict(X_test_tab).reshape(-1, 1)
y_future_lgbm_scaled = lgbm_model.predict(X_future_tab).reshape(-1, 1)

# Rescale LightGBM predictions
y_pred_lgbm_rescaled = scaler_y.inverse_transform(y_pred_lgbm_scaled)
y_future_lgbm_rescaled = scaler_y.inverse_transform(y_future_lgbm_scaled)

# Evaluate the LightGBM model's performance
rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm_rescaled, squared=False)
r2_lgbm = r2_score(y_test, y_pred_lgbm_rescaled)
print(f'LightGBM RMSE: {rmse_lgbm:.4f}, R^2: {r2_lgbm:.4f}')

# Define and train TabNet model with best parameters
best_tabnet_params = {'n_d': 16, 'n_a': 8, 'n_steps': 3, 'gamma': 1.5, 'lr': 0.02}

best_tabnet_model = TabNetRegressor(n_d=best_tabnet_params['n_d'], n_a=best_tabnet_params['n_a'], 
                                    n_steps=best_tabnet_params['n_steps'], gamma=best_tabnet_params['gamma'], 
                                    optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=best_tabnet_params['lr']))

best_tabnet_model.fit(X_train_tab, y_train_scaled, eval_set=[(X_test_tab, y_test_scaled)], patience=10, max_epochs=200, 
                      batch_size=128, virtual_batch_size=64, num_workers=0, drop_last=False)

# Make predictions using the TabNet model
y_pred_tabnet_scaled = best_tabnet_model.predict(X_test_tab).reshape(-1, 1)
y_future_tabnet_scaled = best_tabnet_model.predict(X_future_tab).reshape(-1, 1)

# Rescale TabNet predictions
y_pred_tabnet_rescaled = scaler_y.inverse_transform(y_pred_tabnet_scaled)
y_future_tabnet_rescaled = scaler_y.inverse_transform(y_future_tabnet_scaled)

# Evaluate the TabNet model's performance
rmse_tabnet = mean_squared_error(y_test, y_pred_tabnet_rescaled, squared=False)
r2_tabnet = r2_score(y_test, y_pred_tabnet_rescaled)
print(f'TabNet RMSE: {rmse_tabnet:.4f}, R^2: {r2_tabnet:.4f}')

# Ensemble Model
# Prepare predictions from LSTM, LightGBM, and TabNet as features
ensemble_features_train = np.hstack((y_pred_lstm_rescaled, y_pred_lgbm_rescaled, y_pred_tabnet_rescaled))
ensemble_model = LinearRegression()
ensemble_model.fit(ensemble_features_train, y_test)

# Prepare future predictions from LSTM, LightGBM, and TabNet for ensemble
ensemble_features_future = np.hstack((y_future_lstm_rescaled, y_future_lgbm_rescaled, y_future_tabnet_rescaled))

# Make predictions using the Ensemble model
y_pred_ensemble_rescaled = ensemble_model.predict(ensemble_features_train)
y_future_ensemble_rescaled = ensemble_model.predict(ensemble_features_future)

# Evaluate the Ensemble model's performance
rmse_ensemble = mean_squared_error(y_test, y_pred_ensemble_rescaled, squared=False)
r2_ensemble = r2_score(y_test, y_pred_ensemble_rescaled)
print(f'Ensemble RMSE: {rmse_ensemble:.4f}, R^2: {r2_ensemble:.4f}')

# Print all model results
print(f'Root Mean Squared Error (RMSE) for LSTM: {rmse_lstm}')
print(f'R^2 Score for LSTM: {r2_lstm}')
print(f'Root Mean Squared Error (RMSE) for LightGBM: {rmse_lgbm}')
print(f'R^2 Score for LightGBM: {r2_lgbm}')
print(f'Root Mean Squared Error (RMSE) for TabNet: {rmse_tabnet}')
print(f'R^2 Score for TabNet: {r2_tabnet}')
print(f'Root Mean Squared Error (RMSE) for Ensemble: {rmse_ensemble}')
print(f'R^2 Score for Ensemble: {r2_ensemble}')

# Plot actual vs predicted values
plt.figure(figsize=(14, 8))
plt.plot(y_test.index, y_test, label='Actual Values', color='blue')
plt.plot(y_test.index, y_pred_lstm_rescaled, label='Predicted Values - LSTM', color='orange')
plt.plot(y_test.index, y_pred_lgbm_rescaled, label='Predicted Values - LightGBM', color='purple')
plt.plot(y_test.index, y_pred_tabnet_rescaled, label='Predicted Values - TabNet', color='green')
plt.plot(y_test.index, y_pred_ensemble_rescaled, label='Predicted Values - Ensemble', color='red')
plt.xlabel('Index')
plt.ylabel('prices')
plt.title('Actual vs Predicted Values - LSTM, LightGBM, TabNet, and Ensemble Models')
plt.legend()
plt.show()


# y_pred_series = pd.Series(y_pred_ensemble_rescaled)



# plt.figure(figsize=(14, 8))
# plt.plot(y_test.index, y_pred_series/, label='Predicted Values - Ensemble', color='red')
# plt.plot(y_test.index, y_test, label='Actual Values', color='blue')
# plt.xlabel('Index')
# plt.ylabel('30 min return rate')
# plt.title('Actual vs Predicted Values - LSTM, LightGBM, and Ensemble Models')
# plt.legend()
# plt.show()

# Plot future predictions for LSTM, LightGBM, TabNet, and Ensemble
plt.figure(figsize=(14, 8))
plt.plot(future_time, y_future_lstm_rescaled, label='Future Predictions - LSTM', color='orange')
plt.plot(future_time, y_future_lgbm_rescaled, label='Future Predictions - LightGBM', color='purple')
plt.plot(future_time, y_future_tabnet_rescaled, label='Future Predictions - TabNet', color='green')
plt.plot(future_time, y_future_ensemble_rescaled, label='Future Predictions - Ensemble', color='red')
plt.plot(future_time, data_done2[label].iloc[-predict_time:], label='True value', color='blue')
plt.xlabel('Future Time Index')
plt.ylabel('prices')
plt.xticks(rotation=90)
plt.title('Future Predictions - LSTM, LightGBM, TabNet, and Ensemble Models')
plt.legend()
plt.show()

with open('MNQ_5min_model.pkl', 'wb') as f:
    dill.dump({
        'lstm_model': residual_lstm_model,
        'lgbm_model': lgbm_model,
        'ensemble_model': ensemble_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, f)
    
import dill

with open('full_model.pkl', 'rb') as f:
    saved_objects = dill.load(f)

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

    def plot_profit_loss(self):
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
        ax1.plot(future_time, y_future_ensemble_rescaled, color='#6a5acd', alpha=0.8, label="Ensemble Model Predictions")  # 使用紫色
        ax1.plot(future_time, data_done2["close"].iloc[-predict_time:], color='#1e90ff', alpha=0.8, label="True Values")  # 使用藍色

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




# 計算預測回報率
predicted_returns = (y_future_ensemble_rescaled-data_done2['close'].iloc[-predict_time:])/data_done2['close'].iloc[-predict_time:]

# 使用保證金、停損、移動停利以及多/空倉進行回測
backtest = Backtest(initial_balance=200000, transaction_fee=25, margin_rate=0.1, stop_loss=0.0002, trailing_stop_pct=0.0001, point_value=60)
return_rate, trade_log, portfolio_values, profit_loss_log = backtest.run_backtest(predicted_returns, data_done2['close'].iloc[-predict_time:],
                                                                                  buy_threshold=0.001,
                                                                                  short_threshold=-0.001)

# 輸出回測結果
print(f'Final Return Rate: {return_rate * 100:.2f}%')
print('Trade Log:')
for trade in trade_log:
    print(trade)


# 繪製未來預測與買入/賣出標記以及損益圖
backtest.plot_profit_loss()

# 輸出摘要表
backtest.summary_table()



# %%
