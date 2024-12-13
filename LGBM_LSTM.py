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

data = pd.read_excel(r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API\TX00\TX00_5_20241101_20241211.xlsx")
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
data_training = data_done2['2024-11-11':'2024-12-11']
data_future = data_done2['2024-12-11':'2024-12-11']

train_date = ['2024-11-11','2024-12-10']
test_date = ['2024-12-11','2024-12-11']
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
hidden_size = 256
num_layers = 2
learning_rate = 0.008

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
lgbm_model = LGBMRegressor(boosting_type='gbdt',
                           learning_rate = 0.011694439174798591,
                            max_depth = 98,
                            n_estimators= 165,
                            num_leaves = 53,
                            min_child_samples = 41,
                            subsample = 0.9922571912667469,
                            colsample_bytree = 0.9900853051847848,
                            reg_alpha = 0.03916124430025413,
                            reg_lambda = 0.004814454696230497)

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
#%%
import dill
# Save the entire ensemble model
with open('TX_1min_model_5.pkl', 'wb') as f: 
    data_to_save = {
        'lstm_model': residual_lstm_model,
        'lstm_optimizer_state_dict': optimizer.state_dict(),
        'lgbm_model': lgbm_model,
        'ensemble_model': ensemble_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    dill.dump(data_to_save, f)
# 检查保存的数据的键
print(f"保存的数据的键: {data_to_save.keys()}")
# %%
import optuna
from lightgbm import LGBMRegressor

# Define the objective function for LightGBM
def objective_lgbm(trial):
    # Define the hyperparameter search space
    param = {
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 5, 100),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'num_leaves': trial.suggest_int('num_leaves', 15, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e-1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e-1),
    }

    # Train the LightGBM model
    model = LGBMRegressor(**param)
    model.fit(X_train_tab, y_train_scaled.ravel())

    # Evaluate the model
    y_pred_scaled = model.predict(X_test_tab).reshape(-1, 1)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred_scaled)
    rmse = mean_squared_error(y_test, y_pred_rescaled, squared=False)
    return rmse

# Optimize LightGBM hyperparameters
study_lgbm = optuna.create_study(direction="minimize")
study_lgbm.optimize(objective_lgbm, n_trials=200)

# Best hyperparameters
print("Best hyperparameters for LightGBM:", study_lgbm.best_params)

# Train the final LightGBM model with the best hyperparameters
best_params_lgbm = study_lgbm.best_params
final_lgbm_model = LGBMRegressor(**best_params_lgbm)
final_lgbm_model.fit(X_train_tab, y_train_scaled.ravel())

# Make predictions using the final LightGBM model
y_pred_lgbm_scaled = final_lgbm_model.predict(X_test_tab).reshape(-1, 1)
y_pred_lgbm_rescaled = scaler_y.inverse_transform(y_pred_lgbm_scaled)

# Evaluate the final LightGBM model's performance
rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm_rescaled, squared=False)
r2_lgbm = r2_score(y_test, y_pred_lgbm_rescaled)
print(f'Final LightGBM RMSE: {rmse_lgbm:.4f}, R^2: {r2_lgbm:.4f}')

# %%
#Baysiean opitimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import optuna
from sklearn.metrics import mean_squared_error, r2_score

# Objective function for Bayesian Optimization
def objective(trial):
    # Define the hyperparameter search space
    hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Create the Residual LSTM model with current hyperparameters
    model = ResidualLSTMModel(
        input_size=X_train_tab.shape[1],
        hidden_size=hidden_size,
        output_size=1,
        num_layers=num_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    X_train_tensor = torch.tensor(X_train_tab, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

    for epoch in range(50):  # Use a smaller epoch count for faster tuning
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate on validation data
    model.eval()
    X_test_tensor = torch.tensor(X_test_tab, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # Rescale predictions
    y_pred_rescaled = scaler_y.inverse_transform(y_pred_scaled)
    rmse = mean_squared_error(y_test, y_pred_rescaled, squared=False)

    return rmse

# Create a study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)

# Retrain the model with the best hyperparameters
best_params = study.best_params
final_model = ResidualLSTMModel(
    input_size=X_train_tab.shape[1],
    hidden_size=best_params['hidden_size'],
    output_size=1,
    num_layers=best_params['num_layers']
).to(device)
final_model.dropout = nn.Dropout(best_params['dropout_rate'])

final_optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
criterion = nn.MSELoss()

X_train_tensor = torch.tensor(X_train_tab, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

# Train the final model
epochs = 600
for epoch in range(epochs):
    final_model.train()
    final_optimizer.zero_grad()
    outputs = final_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    final_optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Final Model Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the final model
final_model.eval()
X_test_tensor = torch.tensor(X_test_tab, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_final_scaled = final_model(X_test_tensor).cpu().numpy()
    y_pred_final_rescaled = scaler_y.inverse_transform(y_pred_final_scaled)

final_rmse = mean_squared_error(y_test, y_pred_final_rescaled, squared=False)
final_r2 = r2_score(y_test, y_pred_final_rescaled)
print(f'Final Residual LSTM RMSE: {final_rmse:.4f}, R^2: {final_r2:.4f}')
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import datetime
import datetime as dt
import pytz
from datetime import time
import copy


class ForwardTest:
    def __init__(self, initial_balance=100000, transaction_fee=1, margin_rate=0.1, stop_loss=0.001, 
                 trailing_stop_pct=0.000001, point_value=1, data_folder="trading_data", symbol="unknown",
                 morning_session=("08:45", "19:25"), night_session=("15:00", "04:00"),
                 thresholds_by_segment=None):
        """
        初始化參數並根據當前時間動態設置交易時間。
        """
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
        self.trailing_stop_log = []
        self.stop_loss_log = []
        self.trade_time = {"start_time": None, "end_time": None}
        self.sTradeSession = None
        self.thresholds_by_segment = thresholds_by_segment or {}

        # 獲取當前時間並轉換為 UTC+8
        current_time = dt.datetime.now(pytz.utc).astimezone(pytz.timezone("Asia/Shanghai"))

        # 動態設置交易時間
        morning_start, morning_end = [dt.datetime.strptime(t, "%H:%M").time() for t in morning_session]
        night_start, night_end = [dt.datetime.strptime(t, "%H:%M").time() for t in night_session]

        if morning_start <= current_time.time() <= morning_end:
            self.trade_time["start_time"], self.trade_time["end_time"] = morning_session
            self.sTradeSession = 1  # 早盤
            print(f"當前交易時段: 早盤 | 開始: {morning_session[0]}, 結束: {morning_session[1]}")
        elif current_time.time() >= night_start or current_time.time() <= night_end:
            self.trade_time["start_time"], self.trade_time["end_time"] = night_session
            self.sTradeSession = 1  # 夜盤
            print(f"當前交易時段: 夜盤 | 開始: {night_session[0]}, 結束: {night_session[1]}")
        else:
            print("非交易時段，目前無有效交易時間。")
            self.trade_time["start_time"], self.trade_time["end_time"] = None, None
            self.trade_time["start_time"], self.trade_time["end_time"] = night_session
            self.sTradeSession = 1  # 早盤


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

    def detect_market_state(self, speed, force, force_std, osc_window):
        """
        根據當前交易時段檢測市場狀態。
        """
        if len(self.lookback_prices) <= osc_window:
            print(f"不足窗口大小 ({osc_window}) 的數據，返回 'default'")
            return 'default'

        # now = dt.datetime.now(pytz.utc).astimezone(pytz.timezone("Asia/Shanghai")).time()
    
        if self.sTradeSession == 0:  # 早盤
            thresholds = self.thresholds_by_segment.get('morning')
            time_segment = 'morning'
        elif self.sTradeSession == 1:  # 夜盤
            thresholds = self.thresholds_by_segment.get('night')
            time_segment = 'night'
        else:
            print(f"當前不在交易時段內，返回 'None'")
            return 'None'

        if not thresholds:
            print(f"未找到 {time_segment} 時段的閾值資料，返回 'None'")
            return 'None'

        # 解構閾值
        osc_force_std = thresholds['osc_force_std']
        osc_force_range = thresholds['osc_force_range']
        osc_speed_range = thresholds['osc_speed_range']
        trend_force_threshold_up = thresholds['trend_force_threshold_up']
        trend_speed_threshold_up = thresholds['trend_speed_threshold_up']
        trend_force_threshold_down = thresholds['trend_force_threshold_down']
        trend_speed_threshold_down = thresholds['trend_speed_threshold_down']

        print(f"檢查條件: force_std={force_std}, force={force}, speed={speed}, 時段={time_segment}")

        if (
            force_std < osc_force_std and
            (osc_force_range[0] <= force <= osc_force_range[1] or
             osc_speed_range[0] <= speed <= osc_speed_range[1])
        ):
            print(f"市場處於盤整狀態 (oscillation) - 時段: {time_segment}")
            return 'oscillation'

        if (
            (force > trend_force_threshold_up and speed > trend_speed_threshold_up) or
            (force < trend_force_threshold_down and speed < trend_speed_threshold_down)
        ):
            print(f"市場處於趨勢狀態 (trend) - 時段: {time_segment}")
            return 'trend'

        print(f"市場狀態無法確定，返回 'None' - 時段: {time_segment}")
        return 'None'

            

    def apply_strategy(self, strategy_func, *args, **kwargs):
        strategy_func(*args, **kwargs)

    def trend_strategy(self, predicted_return, actual_price, current_time, buy_threshold, short_threshold, 
                    recent_high, recent_low, speed=None, force=None, smoothed_range_diff=None,
                    cooldown_loss_threshold=-2000, cooldown_time=180, retrace_ratios=(0.382, 0.618),
                    dynamic_adjustment=True):
        """
        優化版趨勢策略（無續倉邏輯）。
        1. 動態調整條件。
        2. 分段判斷趨勢。
        """
        # 冷卻期檢查
        if self.trade_log:
            last_trade = self.trade_log[-1]
            if 'profit_loss' in last_trade and last_trade['profit_loss'] is not None:
                last_loss = last_trade['profit_loss']
                time_since_last_trade = (current_time - pd.to_datetime(last_trade['time'])).seconds
                if last_loss < cooldown_loss_threshold and time_since_last_trade < cooldown_time:
                    print(f"Cooling down due to recent loss {last_loss}, Time remaining: {cooldown_time - time_since_last_trade} seconds")
                    return  # 跳過交易

        # 動態計算回撤位
        retrace_1 = recent_low + (recent_high - recent_low) * retrace_ratios[0]  # 第一回撤位
        retrace_2 = recent_low + (recent_high - recent_low) * retrace_ratios[1]  # 第二回撤位

        # 確保指標的完整性
        if speed is None or force is None or smoothed_range_diff is None:
            print("缺少必要的指標 (speed, force, smoothed_range_diff)，跳過此交易。")
            return

        # 動態條件調整
        if dynamic_adjustment:
            volatility = abs(recent_high - recent_low)  # 以最近高低價差衡量波動性
            adjusted_buy_threshold = buy_threshold * (1 + volatility / 1000)
            adjusted_short_threshold = short_threshold * (1 - volatility / 1000)
        else:
            adjusted_buy_threshold = buy_threshold
            adjusted_short_threshold = short_threshold

        # 分段條件
        strong_trend_buy = force > 100 and speed > 10 and smoothed_range_diff < 30
        weak_trend_buy = force > 50 and speed > 5 and smoothed_range_diff < 50
        strong_trend_short = force < -100 and speed < -10 and smoothed_range_diff < 30
        weak_trend_short = force < -50 and speed < -5 and smoothed_range_diff < 50

        # 買入策略
        if (predicted_return > adjusted_buy_threshold and actual_price < retrace_1 and
            (strong_trend_buy or weak_trend_buy)):
            if self.position == 0:
                print(f"Trend Buy triggered at {current_time}, Price: {actual_price}")
                self.execute_trade('buy', actual_price, current_time, contracts=1)

        # 做空策略
        elif (predicted_return < adjusted_short_threshold and actual_price > retrace_2 and
            (strong_trend_short or weak_trend_short)):
            if self.position == 0:
                print(f"Trend Short triggered at {current_time}, Price: {actual_price}")
                self.execute_trade('short', actual_price, current_time, contracts=1)


    def oscillation_strategy(self, predicted_return, actual_price, current_time, recent_high, recent_low,
                            speed=None, force=None, smoothed_range_diff=None, cooldown_loss_threshold=-2000, cooldown_time=180,
                            dynamic_adjustment=True):
        """
        優化版震盪策略，加入冷卻期邏輯、動態邊界和分段條件。
        """
        # 冷卻期檢查
        if self.trade_log:
            last_trade = self.trade_log[-1]
            if 'profit_loss' in last_trade and last_trade['profit_loss'] is not None:
                last_loss = last_trade['profit_loss']
                time_since_last_trade = (current_time - pd.to_datetime(last_trade['time'])).seconds
                if last_loss < cooldown_loss_threshold and time_since_last_trade < cooldown_time:
                    print(f"Cooling down due to recent loss {last_loss}, Time remaining: {cooldown_time - time_since_last_trade} seconds")
                    return  # 跳過交易

        # 動態計算上下邊界
        if dynamic_adjustment:
            volatility = abs(recent_high - recent_low)
            lower_bound = recent_low + (volatility * 0.15)
            upper_bound = recent_high - (volatility * 0.15)
        else:
            lower_bound = recent_low + (recent_high - recent_low) * 0.2
            upper_bound = recent_high - (recent_high - recent_low) * 0.2

        # 確保指標的完整性
        if speed is None or force is None or smoothed_range_diff is None:
            print("缺少必要的指標 (speed, force, smoothed_range_diff)，跳過此交易。")
            return

        # 分段條件
        strong_oscillation_buy = speed < -10 and force < -100 and smoothed_range_diff < 30
        weak_oscillation_buy = speed < -5 and force < 0 and smoothed_range_diff < 50
        strong_oscillation_short = speed > 10 and force > 100 and smoothed_range_diff < 30
        weak_oscillation_short = speed > 5 and force > 0 and smoothed_range_diff < 50

        # 買入策略
        if (predicted_return > 0 and actual_price < lower_bound and
            (strong_oscillation_buy or weak_oscillation_buy)):
            if self.position == 0:
                print(f"Oscillation Buy triggered at {current_time}, Price: {actual_price}")
                self.execute_trade('buy', actual_price, current_time, contracts=1)

        # 做空策略
        elif (predicted_return < 0 and actual_price > upper_bound and
            (strong_oscillation_short or weak_oscillation_short)):
            if self.position == 0:
                print(f"Oscillation Short triggered at {current_time}, Price: {actual_price}")
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
                


    def run_backtest(self, predicted_return, current_time, real_time_data,
                    statue_indicator,
                    buy_threshold=0.0002, short_threshold=-0.0001,
                    osc_window=2, contracts=1):
        """
        執行回測並應用交易策略。
        """
        # 提取即時數據和指標
        actual_price = real_time_data['close']
        high, low, open_price, volume = real_time_data['high'], real_time_data['low'], real_time_data['open'], real_time_data['volume']
        speed, force, smoothed_range_diff, force_std = statue_indicator.values()

        # # 檢查是否在交易時間範圍內
        # if not self._in_trade_window(current_time):
        #     self._force_close_positions(current_time, actual_price, contracts)
        #     return self.get_current_state()

        # 更新歷史價格
        self._update_lookback_prices(high, low, actual_price, volume, osc_window)

        # 檢測市場狀態
        market_state = self.detect_market_state(speed, force, force_std, osc_window)
        self.strategy_status.append({'Time': current_time, 'Strategy state': market_state})
        print(f"Current Strategy: {market_state}")

        # 動態更新移動停利
        self._update_trailing_profit(actual_price)

        # 檢查移動停損和止損條件
        self._check_trailing_profit(current_time, actual_price, contracts)
        self._check_stop_loss(current_time, actual_price, contracts)

        # 如果是震盪市場，額外執行震盪策略的停利條件
        if market_state == 'oscillation':
            self._check_oscillation_stop(current_time, actual_price, contracts)

        # 應用策略
        recent_high, recent_low = self._get_recent_high_low(osc_window)
        if market_state == 'oscillation':
            self.apply_strategy(self.oscillation_strategy, predicted_return, actual_price, current_time, recent_high, recent_low,
                                speed, force, smoothed_range_diff)
        elif market_state == 'trend':
            
            self.apply_strategy(self.trend_strategy, predicted_return, actual_price, current_time, buy_threshold, short_threshold,
                                recent_high, recent_low, speed=speed, force=force, smoothed_range_diff=smoothed_range_diff)


        return self.get_current_state()

    # ------------------------------------------
    # Helper functions
    # ------------------------------------------

    def _in_trade_window(self, current_time):
        """
        檢查當前時間是否在交易時間內。
        """
        start_time = pd.to_datetime(self.trade_time["start_time"], format="%H:%M").time()
        end_time = pd.to_datetime(self.trade_time["end_time"], format="%H:%M").time()
        if start_time < end_time:
            return start_time <= current_time.time() <= end_time
        else:
            return current_time.time() >= start_time or current_time.time() <= end_time

    def _force_close_positions(self, current_time, actual_price, contracts):
        """
        強制平倉。
        """
        if self.position > 0:
            print(f"Force close long position at {current_time}, Price: {actual_price}")
            self.execute_trade('sell', actual_price, current_time, contracts=self.position)
        elif self.position < 0:
            print(f"Force close short position at {current_time}, Price: {actual_price}")
            self.execute_trade('cover', actual_price, current_time, contracts=abs(self.position))

    def _update_lookback_prices(self, high, low, close, volume, osc_window):
        """
        更新歷史價格。
        """
        self.lookback_prices.append({'high': high, 'low': low, 'close': close, 'volume': volume})
        if len(self.lookback_prices) > osc_window + 1:
            self.lookback_prices.pop(0)
        
   
    def _check_trailing_profit(self, current_time, actual_price, contracts):
        """
        檢查是否觸發移動停利。
        """
        if self.position > 0 and self.trailing_stop is not None:  # 多單
            if actual_price <= self.trailing_stop and actual_price > self.entry_price:
                print(f"Trigger Trailing Profit (Buy -> Sell) at {current_time}, Price: {actual_price}")
                self.trailing_stop_log.append({
                    'Time': self.trailing_stop,
                    'price': actual_price,
                    'trailing_stop_price': self.trailing_stop
                })
                self.execute_trade('sell', actual_price, current_time, contracts=self.position)

        elif self.position < 0 and self.trailing_stop is not None:  # 空單
            if actual_price >= self.trailing_stop and actual_price < self.entry_price:
                print(f"Trigger Trailing Profit (Short -> Cover) at {current_time}, Price: {actual_price}")
                self.trailing_stop_log.append({
                    'Time': self.trailing_stop,
                    'price': actual_price,
                    'trailing_stop_price': self.trailing_stop
                })
                self.execute_trade('cover', actual_price, current_time, contracts=abs(self.position))
                
    def _update_trailing_profit(self, actual_price, trailing_stop_pct=None):
        """
        更新移動停利價格，確保隨著價格變化動態調整。
        """
        trailing_stop_pct = trailing_stop_pct or self.trailing_stop_pct

        # 確保價格窗口數據足夠
        if len(self.lookback_prices) < 3:
            print(f"[Warning] Not enough data for trailing stop update. Lookback size: {len(self.lookback_prices)}")
            return

        if self.position > 0:  # 多單
            recent_high = max([x['high'] for x in self.lookback_prices[-3:]])
            new_trailing_stop = max(self.trailing_stop or 0, recent_high * (1 - trailing_stop_pct))
            print(f"Updating trailing stop for long: New: {new_trailing_stop}, Recent High: {recent_high}")
            self.trailing_stop = new_trailing_stop

        elif self.position < 0:  # 空單
            recent_low = min([x['low'] for x in self.lookback_prices[-3:]])
            if self.trailing_stop is None or self.trailing_stop > recent_low * (1 + trailing_stop_pct):
                new_trailing_stop = recent_low * (1 + trailing_stop_pct)
            else:
                new_trailing_stop = self.trailing_stop  # 保持之前的值
            print(f"Updating trailing stop for short: New: {new_trailing_stop}, Recent Low: {recent_low}")
            self.trailing_stop = new_trailing_stop

            self.trailing_stop = new_trailing_stop


    def _check_stop_loss(self, current_time, actual_price, contracts):
        """
        檢查是否觸發止損。
        """
        if self.position > 0 and (self.entry_price - actual_price) / self.entry_price >= self.stop_loss:
            print(f"Trigger Stop Loss (Buy -> Sell) at {current_time}, Price: {actual_price}")
            self.stop_loss_log.append({'Time': current_time, 'price': actual_price, 'entry_price': self.entry_price})
            self.execute_trade('sell', actual_price, current_time, contracts=self.position)
        elif self.position < 0 and (actual_price - self.entry_price) / self.entry_price >= self.stop_loss:
            print(f"Trigger Stop Loss (Short -> Cover) at {current_time}, Price: {actual_price}")
            self.stop_loss_log.append({'Time': current_time, 'price': actual_price, 'entry_price': self.entry_price})
            self.execute_trade('cover', actual_price, current_time, contracts=abs(self.position))

    def _check_oscillation_stop(self, current_time, actual_price, contracts):
        """
        檢查震盪策略的停利條件。
        """
        if self.position > 0 and self.entry_price < actual_price:
            if abs(self.entry_price - actual_price) / self.entry_price > 0.0001:
                print(f"Trigger Trailing Stop for oscillation (Buy -> Sell) at {current_time}, Price: {actual_price}")
                self.trailing_stop_log.append({'Time': current_time, 'price': actual_price, 'trailing_stop_price': None})
                self.execute_trade('sell', actual_price, current_time, contracts=self.position)
        elif self.position < 0 and self.entry_price > actual_price:
            if abs(self.entry_price - actual_price) / self.entry_price > 0.0001:
                print(f"Trigger Trailing Stop for oscillation (Short -> Cover) at {current_time}, Price: {actual_price}")
                self.trailing_stop_log.append({'Time': current_time, 'price': actual_price, 'trailing_stop_price': None})
                self.execute_trade('cover', actual_price, current_time, contracts=abs(self.position))

    def _get_recent_high_low(self, osc_window):
        """
        獲取最近窗口內的最高和最低價格。
        
        :param osc_window: 最近窗口的大小。
        :return: (recent_high, recent_low) 元組。
        """
        # 確認是否有足夠的數據
        if len(self.lookback_prices) < osc_window:
            print(f"Not enough data in lookback_prices. Required: {osc_window}, "
                            f"Available: {len(self.lookback_prices)}")
            return None, None
            

        # 獲取最近窗口的數據
        recent_prices = self.lookback_prices[-osc_window:]

        # 計算高低價
        recent_high = max(recent_prices, key=lambda x: x['high'])['high']
        recent_low = min(recent_prices, key=lambda x: x['low'])['low']

        return recent_high, recent_low



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

    def monitor_stop_loss(self, live_price_data, osc_window, predicted_return, buy_threshold, short_threshold,
                          statue_indicator):
        """
        即時監控停損條件，並根據價格自動平倉。
        Paramter:
            live_price_data: 即時價格 (Dataframe)
            osc_window: 盤整窗口 (int)
            predicted_return: 五分鐘回報率 (int)
            
        """
        
        live_price = live_price_data['price'].iloc[-1]
        current_time = live_price_data.index[-1]
        
        speed, force, smoothed_range_diff , force_std = statue_indicator['speed'], statue_indicator['force'], statue_indicator['smoothed_range_diff'], statue_indicator['force_std']

        
        # 將字符串格式時間轉為 datetime.time
        start_time = pd.to_datetime(self.trade_time["start_time"], format="%H:%M").time()
        end_time = pd.to_datetime(self.trade_time["end_time"], format="%H:%M").time()
        
        # 判断当前时间是否在交易时间范围内
        if start_time < end_time:
            # 情况 1: 不跨天
            in_trade_window = start_time <= current_time.time() <= end_time
        else:
            # 情况 2: 跨天
            in_trade_window = current_time.time() >= start_time or current_time.time() <= end_time

        # 如果不在交易时间范围内
        if not in_trade_window:
            # 如果有持仓，强制平仓
            if self.position > 0:  # 平多单
                print(f"Force close long position at {current_time}, Price: {live_price}")
                self.execute_trade('sell', live_price, current_time, contracts=self.position)
            elif self.position < 0:  # 平空单
                print(f"Force close short position at {current_time}, Price: {live_price}")
                self.execute_trade('cover', live_price, current_time, contracts=abs(self.position))
            
            # 阻止任何交易行为，直接返回当前状态
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
            
            # 計算 recent_high 和 recent_low
            recent_high = max(self.lookback_prices[-osc_window:], key=lambda x: x['high'])['high']
            recent_low = min(self.lookback_prices[-osc_window:], key=lambda x: x['low'])['low']

            # 偵測是否突然突破以便隨時轉換狀態
            self.live_adjust_strategy(live_price, current_time, recent_high, recent_low, predicted_return)

            # Trailing Stop 檢查
            if self.position > 0 and self.entry_price < live_price:  # Long position
                if self.trailing_stop is None or live_price > self.trailing_stop:
                    self.trailing_stop = live_price * (1 - self.trailing_stop_pct)
                if abs(self.entry_price - live_price) / self.entry_price > self.stop_loss:
                    print(f"Trigger Trailing Stop for oscillation (Buy -> Sell) at {current_time}, Price: {live_price}")
                    self.trailing_stop_log.append({'Time': current_time, 'price': live_price, 'trailing_stop_price': self.trailing_stop})
                    self.execute_trade('sell', live_price, current_time, contracts=self.position)

            elif self.position < 0 and self.entry_price > live_price:  # Short position
                if self.trailing_stop is None or live_price < self.trailing_stop:
                    self.trailing_stop = live_price * (1 + self.trailing_stop_pct)
                if abs(self.entry_price - live_price) / self.entry_price > self.stop_loss:
                    print(f"Trigger Trailing Stop for oscillation (Short -> Cover) at {current_time}, Price: {live_price}")
                    self.trailing_stop_log.append({'Time': current_time, 'price': live_price, 'trailing_stop_price': self.trailing_stop})
                    self.execute_trade('cover', live_price, current_time, contracts=abs(self.position))           
                       
        if self.position == 0 and self.strategy_status[-1]['Strategy state'] == 'oscillation':
            # Calculate oscillation strategy values from the window of high/low prices
            recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
            recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']  
                      
            self.apply_strategy(self.oscillation_strategy, predicted_return, live_price, current_time, recent_high, recent_low,
                                speed, force, smoothed_range_diff) 

            
        elif self.strategy_status[-1]['Strategy state'] == 'trend':
            
            # Calculate oscillation strategy values from the window of high/low prices
            recent_high = max(self.lookback_prices[-osc_window:-1], key=lambda x: x['high'])['high']
            recent_low = min(self.lookback_prices[-osc_window:-1], key=lambda x: x['low'])['low']  
                      
            self.apply_strategy(self.trend_strategy, predicted_return, live_price, current_time, buy_threshold, short_threshold, 
                                recent_high, recent_low)
        
         
        return self.get_current_state()
    
    def live_adjust_strategy(self, live_price, current_time, recent_high, recent_low, predicted_return):
        """
        如果盤整趨勢被打破就切換成趨勢策略
        """
        if not self.strategy_status or not self.profit_loss_log:
            print("策略狀態或損益紀錄為空，無法執行調整")
            return

        last_profit_loss = self.profit_loss_log[-1].get('profit_loss', None)
        if last_profit_loss is None or not isinstance(last_profit_loss, (int, float)):
            print("無法獲取最近損益記錄，跳過策略調整")
            return

        is_breakout = recent_high < live_price or recent_low > live_price
        cooldown_period = pd.Timedelta(minutes=5)
        if self.strategy_status and (current_time - self.strategy_status[-1]['Time']) < cooldown_period:
            print("冷卻期間，跳過策略調整")
            return

        # 如果策略是盤整且無持倉，則根據條件切換到趨勢策略
        if self.strategy_status[-1]['Strategy state'] == 'oscillation' and self.position == 0:
            if last_profit_loss:
                self.strategy_status.append({'Time': current_time, 'Strategy state': 'trend'})
                self.apply_strategy(self.breakout_strategy, predicted_return, live_price, current_time, recent_high, recent_low)
            elif is_breakout:
                self.strategy_status.append({'Time': current_time, 'Strategy state': 'trend'})
                self.apply_strategy(self.breakout_strategy, predicted_return, live_price, current_time, recent_high, recent_low)
        elif is_breakout:
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
    def __init__(self, speed_window=5, force_window=5, range_window=5, max_allowed_std=10000, extreme_threshold=5000):
        self.speed_window = speed_window
        self.force_window = force_window
        self.range_window = range_window
        self.max_allowed_std = max_allowed_std
        self.extreme_threshold = extreme_threshold

        self.data = pd.DataFrame(columns=[
            'datetime', 'open', 'high', 'low', 'close', 'volume',
            'speed', 'force', 'smoothed_range_diff', 'force_std'
        ])

    def add_new_data(self, new_data):
        """
        添加新數據，並自動處理數據格式問題。
        """
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        # 如果索引名稱為 'Datetime'，將其重置為列並重命名為 'datetime'
        if new_data.index.name == 'date':
            new_data = new_data.reset_index()
            new_data.rename(columns={'date': 'datetime'}, inplace=True)

        # 確保 datetime 列存在
        if 'datetime' not in new_data.columns:
            raise ValueError("Input data must have a 'datetime' column or DatetimeIndex.")

        # 確保 datetime 列是日期時間格式
        new_data['datetime'] = pd.to_datetime(new_data['datetime'], errors='coerce')
        new_data.dropna(subset=['datetime'], inplace=True)

        # 檢查並提取必要的欄位
        missing_columns = [col for col in required_columns if col not in new_data.columns]
        if missing_columns:
            raise ValueError(f"Input data is missing required columns: {missing_columns}")

        # 選取必要欄位
        filtered_data = new_data[required_columns].copy()

        # 排序數據
        filtered_data.sort_values('datetime', inplace=True)

        # 合併數據
        self.data = pd.concat([self.data, filtered_data], ignore_index=True)

        # 計算技術指標
        self.data['speed'] = self.data['close'].diff()
        self.data['price_diff'] = self.data['close'].diff()
        self.data['price_acceleration'] = self.data['price_diff'].diff()
        self.data['force'] = self.data['price_acceleration'] * self.data['volume']

        self.data['force_std'] = self.data['force'].rolling(window=self.force_window, min_periods=1).std()

        self.data['range_diff'] = self.data['high'] - self.data['low']
        self.data['smoothed_range_diff'] = (
            self.data['range_diff']
            .rolling(window=self.range_window, min_periods=1)
            .mean()
        )

        # 填充 NaN 值
        self.data.fillna(0, inplace=True)



    def get_latest_indicators(self):
        """
        獲取最新計算的指標。
        :return: 包含最新 speed, force, smoothed_range_diff, 和 force_std 的字典。
        """
        if not self.data.empty:
            latest_row = self.data.iloc[-1]
            return {
                'speed': latest_row.get('speed', 0),
                'force': latest_row.get('force', 0),
                'smoothed_range_diff': latest_row.get('smoothed_range_diff', 0),
                'force_std': latest_row.get('force_std', 0),
            }
        return {'speed': 0, 'force': 0, 'smoothed_range_diff': 0, 'force_std': 0}

    def get_full_data(self):
        """
        獲取完整的數據框，包含所有計算的指標。
        :return: 包含所有數據和指標的 DataFrame。
        """
        return self.data.copy()

    def determine_thresholds_from_history(self, force_std_multiplier=1.5, range_quantiles=(0.25, 0.75),
                                          trend_quantile=0.9, time_segments=None):
        if self.data.empty:
            raise ValueError("No historical data to determine thresholds.")

        df = self.data
        thresholds_by_segment = {}
        if time_segments is None:
            thresholds_by_segment['default'] = self._calculate_thresholds_for_df(
                df, force_std_multiplier, range_quantiles, trend_quantile
            )
        else:
            for segment_name, (start_str, end_str) in time_segments.items():
                start_t = pd.to_datetime(start_str, format='%H:%M').time()
                end_t = pd.to_datetime(end_str, format='%H:%M').time()

                if end_t < start_t:
                    mask = (df['datetime'].dt.time >= start_t) | (df['datetime'].dt.time <= end_t)
                else:
                    mask = (df['datetime'].dt.time >= start_t) & (df['datetime'].dt.time <= end_t)

                segment_df = df.loc[mask]

                if len(segment_df) < 10:
                    thresholds_by_segment[segment_name] = None
                else:
                    thresholds_by_segment[segment_name] = self._calculate_thresholds_for_df(
                        segment_df, force_std_multiplier, range_quantiles, trend_quantile
                    )

        return thresholds_by_segment

    def _clip_and_limit_value(self, value, lower_percentile=0.1, upper_percentile=0.9, max_limit=None):
        lower_bound = value.quantile(lower_percentile)
        upper_bound = value.quantile(upper_percentile)
        clipped_value = value.clip(lower=lower_bound, upper=upper_bound)

        if max_limit:
            clipped_value = clipped_value.apply(lambda x: min(x, max_limit))
        return clipped_value

    def _calculate_thresholds_for_df(self, df, force_std_multiplier, range_quantiles, trend_quantile):
        force_std_series = df['force_std'].dropna()
        force_series = df['force'].dropna()
        speed_series = df['speed'].dropna()

        if len(force_series) < 10 or len(speed_series) < 10:
            raise ValueError("Not enough data in this segment to determine thresholds.")

        force_std_series = self._clip_and_limit_value(force_std_series, 0.1, 0.9, self.max_allowed_std)
        force_series = self._clip_and_limit_value(force_series, 0.1, 0.9)
        speed_series = self._clip_and_limit_value(speed_series, 0.1, 0.9)

        force_std_mean = force_std_series.mean()
        force_std_std = force_std_series.std(ddof=1)

        adjusted_multiplier = force_std_multiplier
        if force_std_std > self.extreme_threshold:
            adjusted_multiplier = force_std_multiplier / 2

        osc_force_std = min(force_std_mean + adjusted_multiplier * force_std_std, self.max_allowed_std)

        low_q, high_q = range_quantiles
        force_low, force_high = force_series.quantile([low_q, high_q])
        speed_low, speed_high = speed_series.quantile([low_q, high_q])

        trend_force_up = force_series.quantile(trend_quantile)
        trend_force_down = force_series.quantile(1 - trend_quantile)
        trend_speed_up = speed_series.quantile(trend_quantile)
        trend_speed_down = speed_series.quantile(1 - trend_quantile)

        return {
            'osc_force_std': osc_force_std,
            'osc_force_range': (force_low, force_high),
            'osc_speed_range': (speed_low, speed_high),
            'trend_force_threshold_up': trend_force_up,
            'trend_speed_threshold_up': trend_speed_up,
            'trend_force_threshold_down': trend_force_down,
            'trend_speed_threshold_down': trend_speed_down
        }

    def plot_thresholds(self, thresholds):
        """
        將指標與計算出的閾值進行可視化。
        """
        if self.data.empty:
            raise ValueError("No data to plot.")

        plt.figure(figsize=(15, 10))

        # Plot force_std
        plt.subplot(3, 1, 1)
        plt.plot(self.data['datetime'], self.data['force_std'], label='force_std')
        plt.axhline(y=thresholds['osc_force_std'], color='r', linestyle='--', label='osc_force_std')
        plt.legend()
        plt.title('Force Std with Thresholds')

        # Plot force
        plt.subplot(3, 1, 2)
        plt.plot(self.data['datetime'], self.data['force'], label='force')
        plt.axhline(y=thresholds['osc_force_range'][0], color='g', linestyle='--', label='Force Low')
        plt.axhline(y=thresholds['osc_force_range'][1], color='r', linestyle='--', label='Force High')
        plt.legend()
        plt.title('Force with Oscillation Range')

        # Plot speed
        plt.subplot(3, 1, 3)
        plt.plot(self.data['datetime'], self.data['speed'], label='speed')
        plt.axhline(y=thresholds['osc_speed_range'][0], color='g', linestyle='--', label='Speed Low')
        plt.axhline(y=thresholds['osc_speed_range'][1], color='r', linestyle='--', label='Speed High')
        plt.legend()
        plt.title('Speed with Oscillation Range')

        plt.tight_layout()
        plt.show()
#%%
predicted_returns = pd.Series(y_pred_ensemble_rescaled.flatten())

# 初始化類別
indicator = RealtimeSpeedForceIndicators(max_allowed_std=50000)

# 添加測試數據
indicator.add_new_data(data_training[['open', 'high', 'low', 'close', 'volume']][test_date[0]:test_date[1]])

# 定義時間段
time_segments = {
    'morning': ('09:00', '12:30'),
    'night': ('18:00', '04:00')
}

# 計算每個時間段的閾值
thresholds = indicator.determine_thresholds_from_history(
    force_std_multiplier=1.1,
    range_quantiles=(0.25, 0.75),
    trend_quantile=0.6,
    time_segments=time_segments
)

forward_test = ForwardTest(initial_balance=200000, transaction_fee=30, margin_rate=0.1, stop_loss=0.001, trailing_stop_pct=0.01, 
                           point_value=50, data_folder = "trading_data_test", symbol='TX00', thresholds_by_segment= thresholds)
                   

#%%
for cc in range(50):
    predicted_return = predicted_returns.iloc[cc]
    actual_price = data_training[['open', 'high', 'low', 'close', 'volume']][test_date[0]:test_date[1]].iloc[cc]
    actual_price_series = data_training[['open', 'high', 'low', 'close', 'volume']][test_date[0]:test_date[1]].iloc[0:cc+1]
    indicator.add_new_data(actual_price_series)
    
    status_indicator_done = indicator.get_latest_indicators()
    
    current_time = data_training['close'][test_date[0]:test_date[1]].index[cc]

    state = forward_test.run_backtest(
        predicted_return=predicted_return,
        real_time_data=actual_price,
        current_time=current_time,
        statue_indicator= status_indicator_done,
        buy_threshold=0.0002,
        osc_window = 3,
        short_threshold=-0.0002
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
