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

data = pd.read_excel(r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API\TX00\TX00_5_20241101_20241205.xlsx")
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

#%%
#增加target variable
data_done2['predict_15min_return'] = (data_done2['close'].shift(-3)-data_done2['close'])/data_done2['close']

# Add year, month, day, weekday features
data_done2['day'] = data_done2.index.day
data_done2['minute'] = data_done2.index.minute
data_done2['hour'] = data_done2.index.hour


#%%
label ='predict_15min_return'

end_day, end_month = 15, 11
start_day, start_month = 15, 10
# 訓練數據和未來數據的切片
data_training = data_done2['2024-11-14':'2024-12-04']
data_future = data_done2['2024-12-04':'2024-12-04']

train_date = ['2024-11-10','2024-12-03']
test_date = ['2024-12-04','2024-12-04']
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

# Ensemble Model
# Prepare predictions from LSTM and LightGBM as features
ensemble_features_train = np.hstack((y_pred_lstm_rescaled, y_pred_lgbm_rescaled))
ensemble_model = LinearRegression()
ensemble_model.fit(ensemble_features_train, y_test)

# Prepare future predictions from LSTM and LightGBM for ensemble
ensemble_features_future = np.hstack((y_future_lstm_rescaled, y_future_lgbm_rescaled))

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
print(f'Root Mean Squared Error (RMSE) for Ensemble: {rmse_ensemble}')
print(f'R^2 Score for Ensemble: {r2_ensemble}')

# Plot actual vs predicted values
plt.figure(figsize=(14, 8))
plt.plot(y_test.index, y_test, label='Actual Values', color='blue')
plt.plot(y_test.index, y_pred_lstm_rescaled, label='Predicted Values - LSTM', color='orange')
plt.plot(y_test.index, y_pred_lgbm_rescaled, label='Predicted Values - LightGBM', color='purple')
plt.plot(y_test.index, y_pred_ensemble_rescaled, label='Predicted Values - Ensemble', color='red')
plt.xlabel('Index')
plt.ylabel('prices')
plt.title('Actual vs Predicted Values - LSTM, LightGBM, and Ensemble Models')
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
plt.plot(X_future.index, y_future_lstm_rescaled, label='Future Predictions - LSTM', color='orange')
plt.plot(X_future.index, y_future_lgbm_rescaled, label='Future Predictions - LightGBM', color='purple')
plt.plot(X_future.index, y_future_ensemble_rescaled, label='Future Predictions - Ensemble', color='red')
plt.plot(X_future.index, y_future, label='True value', color='blue')
plt.xlabel('Future Time Index')
plt.ylabel('prices')
plt.xticks(rotation=90)
plt.title('Future Predictions - LSTM, LightGBM, TabNet, and Ensemble Models')
plt.legend()
plt.show()


# %%
