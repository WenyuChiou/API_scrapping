#%%
import sys
import pandas as pd
from alpha_eric import AlphaFactory
from TAIndicator import TAIndicatorSettings
from scraping_and_indicators import StockDataScraper
import pandas as pd
import numpy as np
from talib import abstract
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

data = pd.read_excel(r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API\CME,MNQ0000\CME,MNQ0000_1_20240701_20241124.xlsx")
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

label ='close'
X = data_done2.drop(columns=[label])

y = data_done2[label].dropna()[:-20]

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
    kf.R = np.array([[1]])  # Measurement noise covariance (adjustable)
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
plt.plot(data['close'], label="Original avg_return_after_20_days", color='blue')
plt.plot(y_smoothed_kalman, label="Smoothed by Kalman Filter", color='orange')
plt.legend()
plt.title("Kalman Filter Smoothing of Target Variable: profit_or_loss_after_20_days_pp")
plt.show()
# Create Date Features
# Add year, month, day, weekday features
X['day'] = data.index.day
X['minute'] = data.index.minute
X['hour'] = data.index.hour

# Create Lag Features


# 使用切片來獲取 data.index 中最後 20 個時間戳，提供未來使用
future_time = data.index[-20:]

# 使用切片來保留不包括最後 20 行的索引
X_future = X.tail(20)


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

# Split Training and Testing Data
# Use rows -420 to -91 for the training set, and the last 90 rows as the test set

test_interval = 200
train_time_lenght = 2000
X_train = X_reduced[-(train_time_lenght + test_interval):-test_interval]
y_train = y_smoothed_kalman.iloc[-(train_time_lenght + test_interval):-test_interval].reset_index(drop=True)
X_test = X_reduced[-test_interval:]
y_test = y_smoothed_kalman.iloc[-test_interval:].reset_index(drop=True)



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
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with zeros (handle both batched and unbatched input)
        batch_size = x.size(0)
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x.unsqueeze(1) if x.ndim == 2 else x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Scale the data
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

# Train the best LSTM model on the entire training set using the best parameters
hidden_size = 256
num_layers = 3
learning_rate = 0.0005

lstm_model = LSTMModel(X_train_tab.shape[1], hidden_size, 1, num_layers).to(device)
criterion = nn.MSELoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

X_train_tensor = torch.tensor(X_train_tab, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

epochs = 800
for epoch in range(epochs):
    lstm_model.train()
    lstm_optimizer.zero_grad()
    outputs = lstm_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    lstm_optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'LSTM Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Make predictions using the LSTM model
lstm_model.eval()

X_test_tensor = torch.tensor(X_test_tab, dtype=torch.float32).to(device)
X_future_tensor = torch.tensor(X_future_tab, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_lstm_scaled = lstm_model(X_test_tensor).cpu().numpy()
    y_future_lstm_scaled = lstm_model(X_future_tensor).cpu().numpy()

# Rescale LSTM predictions
y_pred_lstm_rescaled = scaler_y.inverse_transform(y_pred_lstm_scaled)
y_future_lstm_rescaled = scaler_y.inverse_transform(y_future_lstm_scaled)

# Evaluate the LSTM model's performance
rmse_lstm = mean_squared_error(y_test, y_pred_lstm_rescaled, squared=False)
r2_lstm = r2_score(y_test, y_pred_lstm_rescaled)

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

# Ensemble Model
# Prepare predictions from LSTM and TabNet as features
ensemble_features_train = np.hstack((y_pred_lstm_rescaled, y_pred_tabnet_rescaled))
ensemble_model = LinearRegression()
ensemble_model.fit(ensemble_features_train, y_test)

# Prepare future predictions from LSTM and TabNet for ensemble
ensemble_features_future = np.hstack((y_future_lstm_rescaled, y_future_tabnet_rescaled))

# Make predictions using the Ensemble model
y_pred_ensemble_rescaled = ensemble_model.predict(ensemble_features_train)
y_future_ensemble_rescaled = ensemble_model.predict(ensemble_features_future)

# Evaluate the Ensemble model's performance
rmse_ensemble = mean_squared_error(y_test, y_pred_ensemble_rescaled, squared=False)
r2_ensemble = r2_score(y_test, y_pred_ensemble_rescaled)

# Print all model results
print(f'Root Mean Squared Error (RMSE) for LSTM: {rmse_lstm}')
print(f'R^2 Score for LSTM: {r2_lstm}')
print(f'Root Mean Squared Error (RMSE) for TabNet: {rmse_tabnet}')
print(f'R^2 Score for TabNet: {r2_tabnet}')
print(f'Root Mean Squared Error (RMSE) for Ensemble: {rmse_ensemble}')
print(f'R^2 Score for Ensemble: {r2_ensemble}')

# Plot actual vs predicted values
plt.figure(figsize=(14, 8))
plt.plot(y_test.index, y_test, label='Actual Values', color='blue')
plt.plot(y_test.index, y_pred_lstm_rescaled, label='Predicted Values - LSTM', color='orange')
plt.plot(y_test.index, y_pred_tabnet_rescaled, label='Predicted Values - TabNet', color='green')
plt.plot(y_test.index, y_pred_ensemble_rescaled, label='Predicted Values - Ensemble', color='red')
plt.xlabel('Index')
plt.ylabel('prices')
plt.title('Actual vs Predicted Values - LSTM, TabNet, and Ensemble Models')
plt.legend()
plt.show()

# Plot future predictions for LSTM, TabNet, and Ensemble
plt.figure(figsize=(14, 8))
plt.plot(future_time, y_future_lstm_rescaled, label='Future Predictions - LSTM', color='orange')
plt.plot(future_time, y_future_tabnet_rescaled, label='Future Predictions - TabNet', color='green')
plt.plot(future_time, y_future_ensemble_rescaled, label='Future Predictions - Ensemble', color='red')
plt.xlabel('Future Time Index')
plt.ylabel('prices')
plt.xticks(rotation=90)
plt.title('Future Predictions - LSTM, TabNet, and Ensemble Models')
plt.legend()
plt.show()



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
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import dill

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with zeros (handle both batched and unbatched input)
        batch_size = x.size(0)
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x.unsqueeze(1) if x.ndim == 2 else x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Scale the data
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

# Train the best LSTM model on the entire training set using the best parameters
hidden_size = 256
num_layers = 3
learning_rate = 0.0005

lstm_model = LSTMModel(X_train_tab.shape[1], hidden_size, 1, num_layers).to(device)
criterion = nn.MSELoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

X_train_tensor = torch.tensor(X_train_tab, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

epochs = 800
for epoch in range(epochs):
    lstm_model.train()
    lstm_optimizer.zero_grad()
    outputs = lstm_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    lstm_optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'LSTM Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Make predictions using the LSTM model
lstm_model.eval()

X_test_tensor = torch.tensor(X_test_tab, dtype=torch.float32).to(device)
X_future_tensor = torch.tensor(X_future_tab, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_lstm_scaled = lstm_model(X_test_tensor).cpu().numpy()
    y_future_lstm_scaled = lstm_model(X_future_tensor).cpu().numpy()

# Rescale LSTM predictions
y_pred_lstm_rescaled = scaler_y.inverse_transform(y_pred_lstm_scaled)
y_future_lstm_rescaled = scaler_y.inverse_transform(y_future_lstm_scaled)

# Evaluate the LSTM model's performance
rmse_lstm = mean_squared_error(y_test, y_pred_lstm_rescaled, squared=False)
r2_lstm = r2_score(y_test, y_pred_lstm_rescaled)

# Define LightGBM model with specific parameters
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

# Plot future predictions for LSTM, LightGBM, and Ensemble
plt.figure(figsize=(14, 8))
plt.plot(future_time, y_future_lstm_rescaled, label='Future Predictions - LSTM', color='orange')
plt.plot(future_time, y_future_lgbm_rescaled, label='Future Predictions - LightGBM', color='purple')
plt.plot(future_time, y_future_ensemble_rescaled, label='Future Predictions - Ensemble', color='red')
plt.xlabel('Future Time Index')
plt.ylabel('prices')
plt.xticks(rotation=90)
plt.title('Future Predictions - LSTM, LightGBM, and Ensemble Models')
plt.legend()
plt.show()


# %%
with open('full_model.pkl', 'wb') as f:
    dill.dump({
        'lstm_model': lstm_model,
        'lstm_optimizer_state_dict': lstm_optimizer.state_dict(),
        'lgbm_model': lgbm_model,
        'ensemble_model': ensemble_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, f)

