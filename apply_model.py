#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

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

        out1, _ = self.lstm1(x, (h_0, c_0))
        
        h_0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(x.device)
        c_0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(x.device)
        out2, _ = self.lstm2(out1, (h_0_2, c_0_2))

        # Residual connection
        residual = x[:, -1, :]  # Taking the last step for residual connection
        if residual.size(1) > out2.size(2):
            residual = residual[:, :out2.size(2)]  # Trim residual dimensions to match LSTM output dimensions
        elif residual.size(1) < out2.size(2):
            padding = torch.zeros((batch_size, out2.size(2) - residual.size(1))).to(x.device)
            residual = torch.cat((residual, padding), dim=1)  # Pad residual dimensions to match LSTM output dimensions
        out = out2[:, -1, :] + residual  # Add input as residual
        
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Load trained models and scalers
import dill

with open('TX_1min_model_5.pkl', 'rb') as f:
    saved_data = dill.load(f)

loaded_lstm_model = saved_data['lstm_model']
loaded_lgbm_model = saved_data['lgbm_model']
loaded_ensemble_model = saved_data['ensemble_model']
loaded_scaler_X = saved_data['scaler_X']
loaded_scaler_y = saved_data['scaler_y']



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

data = pd.read_excel(r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API\TX00\TX00_1_20241001_20241124.xlsx")
data = data.set_index('date')
# 刪除包含零的列（行）
data = data.loc[~(data == 0).any(axis=1)]

filter_name = pd.read_excel("highly_correlated_features.xlsx")
filter_name = filter_name["Highly Correlated Features"].to_list()

# %%
import time
import warnings
import matplotlib.pyplot as plt
data_test=data['2024-11-12':'2024-11-18']
new_feed = data["2024-11-19":"2024-11-19"]

predict_data_lstm =[]
predict_data_ensemble =[]
warnings.filterwarnings('ignore')
for i in range(len(new_feed)):
    start_time = time.time()
    
    row = new_feed.iloc[i:i+1]  # 選擇第 i 行，保持 DataFrame 格式
    data_test = pd.concat([data_test, row]).sort_index()  # 合併並排序

    alpha = AlphaFactory(data_test)
    indicator = TAIndicatorSettings()
    indicator2 = StockDataScraper()

    data_alpha = alpha.add_all_alphas()

    filtered_settings, timeperiod_only_indicators = indicator.process_settings()  # 处理所有步骤并获取结果
    data_done1 = indicator2.add_indicator_with_timeperiods(data_alpha,timeperiod_only_indicators, timeperiods=[5, 10, 20, 50, 100, 200])
    indicator_list = list(filtered_settings.keys())
    data_done2 = indicator2.add_specified_indicators(data_alpha, indicator_list, filtered_settings)
    data_done2 = add_all_ta_features(data_done2,open='open',high='high',low='low',close='close',volume='volume')

    y = data_done2['close']
    X = data_done2.drop(columns=['close'])
    X = data_done2.replace([np.inf, -np.inf], np.nan)  # 将 inf 替换为 NaN，以便可以使用 dropna() 删除它们
    X = X.dropna(axis=1, how='any')  # 删除包含 NaN 的列



    # Create Date Features
    # Add year, month, day, weekday features
    X['day'] = data_test.index.day
    X['minute'] = data_test.index.minute
    X['hour'] = data_test.index.hour
    
    


    X = X[filter_name]
    X_filter = pd.DataFrame(X,columns=filter_name)
    
    
    end_time = time.time()
    
    
    # 单步预测
    new_input = X.tail(i+1)


    # 2. Preprocess the new input data using the fitted scaler
    new_input_scaled = loaded_scaler_X.transform(new_input)

    # 3. Convert the scaled input data to a PyTorch tensor
    new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32).to(device)

    # Add a dimension to match LSTM input format (batch_size, sequence_length, input_size)
    new_input_tensor = new_input_tensor.unsqueeze(1)
    # LightGBM 预测（单行输入）
    prediction_lgbm_scaled = loaded_lgbm_model.predict(new_input_scaled).reshape(-1, 1)
    prediction_lgbm = loaded_scaler_y.inverse_transform(prediction_lgbm_scaled)

    # 4. Predict using the LSTM model
    with torch.no_grad():
        prediction_lstm_scaled = loaded_lstm_model(new_input_tensor).cpu().numpy()

    # Rescale the prediction
    prediction_lstm = loaded_scaler_y.inverse_transform(prediction_lstm_scaled)
    print(f'新输入的预测结果 (LSTM): {prediction_lstm}')



    # 准备集成模型的输入特征
    ensemble_features = np.hstack((prediction_lstm, prediction_lgbm))

    # 使用集成模型进行预测
    prediction_ensemble = loaded_ensemble_model.predict(ensemble_features)
    print(f'单步预测的集成模型结果: {prediction_ensemble}')
        
    predict_data_lstm.append(prediction_lstm[-1].flatten())
    predict_data_ensemble.append(prediction_ensemble[-1].flatten())
    plt.close()
    # Plot actual vs predicted values
    plt.figure(figsize=(14, 8))
    plt.plot(X.tail(i+1).index + pd.Timedelta(minutes=5),predict_data_lstm, label='Predict (LSTM)', color='blue')
    plt.plot(X.tail(i+1).index + pd.Timedelta(minutes=5),predict_data_ensemble, label='Predict (Ensemble)', color='black')
    plt.plot(X.tail(i+1).index, data_done2['close'].tail(i+1), label='Acutal', color='orange')
    plt.xlabel('Index')
    plt.ylabel('prices')
    plt.title('Actual vs Predicted Values - LSTM, LightGBM, TabNet, and Ensemble Models')
    plt.legend()
    plt.show()
  

    execution_time = end_time - start_time  # in seconds
    

    print(f"Execution time: {execution_time:.2f} seconds")
#%%
    # Plot actual vs predicted values
    plt.figure(figsize=(14, 8))
    plt.plot(X.tail(i+1).index ,predict_data_lstm, label='Predict (LSTM)', color='blue')
    plt.plot(X.tail(i+1).index ,predict_data_ensemble, label='Predict (Ensemble)', color='black')
    plt.plot(X.tail(i+1).index, data_done2['close'].tail(i+1), label='Acutal', color='orange')
    plt.xlabel('Index')
    plt.ylabel('prices')
    plt.title('Actual vs Predicted Values - LSTM, LightGBM, TabNet, and Ensemble Models')
    plt.legend()
    plt.show()
#%%
new_feed = data["2024-11-20":"2024-11-20"]

# %%
X_time = data_test.index
y = data_done2['close']
X = data_done2.drop(columns=['close'])


X = X.replace([np.inf, -np.inf], np.nan)  # 将 inf 替换为 NaN，以便可以使用 dropna() 删除它们
X = X.dropna(axis=1, how='any')  # 删除包含 NaN 的列


# Create Date Features
# Add year, month, day, weekday features
X['day'] = data_test.index.day
X['minute'] = data_test.index.minute
X['hour'] = data_test.index.hour

X = X[filter_name]
X_filter = pd.DataFrame(X,columns=filter_name)




X_filter = X_filter[-600:-550]

# %%
import torch
import torch.nn as nn
import torch.optim as optim

with open('TX_1min_modelnew.pkl', 'rb') as f:
    model_data = dill.load(f)

# Extract individual components
residual_lstm_model_state_dict = model_data['residual_lstm_model']
lstm_optimizer_state_dict = model_data['lstm_optimizer_state_dict']
lgbm_model = model_data['lgbm_model']
ensemble_model = model_data['ensemble_model']
scaler_X = model_data['scaler_X']
scaler_y = model_data['scaler_y']
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 假設 X_filter 是處理過的資料
# 1. 標準化 X_filter
X_filter_scaled = scaler_X.transform(X_filter)

# 2. 使用 LSTM 和 LightGBM 模型進行預測
# 轉換為 PyTorch tensor
X_filter_tensor = torch.tensor(X_filter_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    # LSTM 預測
    y_filter_lstm_scaled = residual_lstm_model_state_dict(X_filter_tensor).cpu().numpy()

# 使用 LightGBM 進行預測
y_filter_lgbm_scaled = lgbm_model.predict(X_filter_scaled).reshape(-1, 1)

# 3. 反標準化預測結果
y_filter_lstm_rescaled = scaler_y.inverse_transform(y_filter_lstm_scaled)
y_filter_lgbm_rescaled = scaler_y.inverse_transform(y_filter_lgbm_scaled)

# 4. 準備集成模型的輸入特徵（LSTM 和 LightGBM 預測結果）
ensemble_features_filter = np.hstack((y_filter_lstm_rescaled, y_filter_lgbm_rescaled))

# 5. 使用訓練好的集成模型進行預測
y_filter_ensemble_rescaled = ensemble_model.predict(X.tail(1))

# 6. 打印結果
print(f"Predictions from Ensemble Model on Filtered Data: {y_filter_ensemble_rescaled}")

# %%
from back_testing.Backtest import Backtest

def calculate_shifted_return(series, shift_period):
    """
    計算時間序列在指定間隔之後的相對變化百分比。
    
    參數:
    series (pd.Series): 要計算的時間序列
    shift_period (int): 向前或向後滾動的間隔長度
    
    返回:
    pd.Series: 計算出的相對變化百分比
    """
    shifted_series = series.shift(-shift_period)
    shifted_return = (shifted_series - series) / series
    return shifted_return

# 計算預測回報率
predicted_returns = (np.concatenate(predict_data_ensemble)-np.array(data_done2['close'].tail(i+1)))/np.array(data_done2['close'].tail(i+1))

# 使用保證金、停損、移動停利以及多/空倉進行回測
backtest = Backtest(initial_balance=100000, transaction_fee=30, margin_rate=0.1, stop_loss=0.0002, trailing_stop_pct=0.0005, point_value=50)
return_rate, trade_log, portfolio_values, profit_loss_log = backtest.run_backtest(predicted_returns, data_done2['close'].tail(i+1),
                                                                                  buy_threshold=0.0005,
                                                                                  short_threshold=-0.001)

# 輸出回測結果
print(f'Final Return Rate: {return_rate * 100:.2f}%')
print('Trade Log:')
for trade in trade_log:
    print(trade)


# 繪製未來預測與買入/賣出標記以及損益圖
backtest.plot_profit_loss(X.tail(i+1).index,np.concatenate(predict_data_ensemble),data_done2['close'].tail(i+1))

# 輸出摘要表
backtest.summary_table()
# %%


# 单步预测
new_input = X.tail(1)


# 2. Preprocess the new input data using the fitted scaler
new_input_scaled = loaded_scaler_X.transform(new_input)

# 3. Convert the scaled input data to a PyTorch tensor
new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32).to(device)

# Add a dimension to match LSTM input format (batch_size, sequence_length, input_size)
new_input_tensor = new_input_tensor.unsqueeze(0)
# LightGBM 预测（单行输入）
prediction_lgbm_scaled = loaded_lgbm_model.predict(new_input_scaled).reshape(-1, 1)
prediction_lgbm = loaded_scaler_y.inverse_transform(prediction_lgbm_scaled)

# 4. Predict using the LSTM model
with torch.no_grad():
    prediction_lstm_scaled = loaded_lstm_model(new_input_tensor).cpu().numpy()

# Rescale the prediction
prediction_lstm = loaded_scaler_y.inverse_transform(prediction_lstm_scaled)
print(f'新输入的预测结果 (LSTM): {prediction_lstm}')

# LSTM 预测（单行输入）
new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32).to(device).unsqueeze(0)
with torch.no_grad():
    prediction_lstm_scaled = loaded_lstm_model(new_input_tensor).cpu().numpy()
prediction_lstm = loaded_scaler_y.inverse_transform(prediction_lstm_scaled)

# 准备集成模型的输入特征
ensemble_features = np.hstack((prediction_lstm, prediction_lgbm))

# 使用集成模型进行预测
prediction_ensemble = loaded_ensemble_model.predict(ensemble_features)
print(f'单步预测的集成模型结果: {prediction_ensemble}')
# %%
