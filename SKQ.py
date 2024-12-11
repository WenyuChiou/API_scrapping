#%%
import pythoncom
import asyncio
import datetime
import pandas as pd
import comtypes.client as cc
import plotly.graph_objects
import comtypes
# 只有第一次使用 api ，或是更新 api 版本時，才需要呼叫 GetModule
# 會將 SKCOM api 包裝成 python 可用的 package ，並存放在 comtypes.gen 資料夾下
# 更新 api 版本時，記得將 comtypes.gen 資料夾 SKCOMLib 相關檔案刪除，再重新呼叫 GetModule 
cc.GetModule(r'C:\Users\user\OneDrive - Lehigh University\Desktop\investment\CapitalAPI_2.13.51_PythonExample\CapitalAPI_2.13.51_PythonExample\元件\x64\SKCOM.dll')
import comtypes.gen.SKCOMLib as sk
# %%
# login ID and PW
# 身份證
ID = 'F130659713'
# 密碼
PW = 'eric1234'
print(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S,"), 'Set ID and PW')


     # 建立 event pump and event loop
# 新版的jupyterlab event pump 機制好像有改變，因此自行打造一個 event pump機制，
# 目前在 jupyterlab 環境下使用，也有在 spyder IDE 下測試過，都可以正常運行
# %%
# working functions, async coruntime to pump events
async def pump_task():
    '''在背景裡定時 pump windows messages'''
    while True:
        pythoncom.PumpWaitingMessages()
        # 想要反應更快 可以將 0.1 取更小值
        await asyncio.sleep(0.1)

# %%
# get an event loop
loop = asyncio.get_event_loop()
pumping_loop = loop.create_task(pump_task())
print(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S,"), "Event pumping is ready!")

# %%
# %%
# 建立物件，避免重複 createObject
# 登錄物件
if 'skC' not in globals(): 
    skC = cc.CreateObject(sk.SKCenterLib, interface=sk.ISKCenterLib)
# 海期報價物件
if 'skQ' not in globals(): 
    skQ = cc.CreateObject(sk.SKQuoteLib,interface=sk.ISKQuoteLib)
# 回報物件
if 'skR' not in globals(): 
    skR = cc.CreateObject(sk.SKReplyLib, interface=sk.ISKReplyLib)
import sys
import pandas as pd
from package.alpha_eric import AlphaFactory  # 假設您自定義的 AlphaFactory 在這個模塊中
from package.TAIndicator import TAIndicatorSettings  # 假設 TAIndicator 設定在這個模塊中
from package.scraping_and_indicators import StockDataScraper  # 假設爬取和指標工具在這個模塊中
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
import joblib
import torch
import torch.nn as nn
from sklearn.exceptions import NotFittedError
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class ResidualLSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=3):
#         super(ResidualLSTMModel, self).__init__()
#         self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         batch_size = x.size(0)
#         h_0 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size).to(x.device)
#         c_0 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size).to(x.device)

#         out1, _ = self.lstm1(x, (h_0, c_0))
        
#         h_0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(x.device)
#         c_0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(x.device)
#         out2, _ = self.lstm2(out1, (h_0_2, c_0_2))

#         # Residual connection
#         residual = x[:, -1, :]  # Taking the last step for residual connection
#         if residual.size(1) > out2.size(2):
#             residual = residual[:, :out2.size(2)]  # Trim residual dimensions to match LSTM output dimensions
#         elif residual.size(1) < out2.size(2):
#             padding = torch.zeros((batch_size, out2.size(2) - residual.size(1))).to(x.device)
#             residual = torch.cat((residual, padding), dim=1)  # Pad residual dimensions to match LSTM output dimensions
#         out = out2[:, -1, :] + residual  # Add input as residual
        
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out

# # Load trained models and scalers
# import dill

# with open('TX_1min_model_5.pkl', 'rb') as f:
#     saved_data = dill.load(f)

# lstm_model = saved_data['lstm_model']
# lgbm_model = saved_data['lgbm_model']
# ensemble_model = saved_data['ensemble_model']
# scaler_X = saved_data['scaler_X']
# scaler_y = saved_data['scaler_y']


data = pd.read_excel(r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API\TX00\TX00_5_20241101_20241209.xlsx")
data = data.set_index('date')
# 刪除包含零的列（行）
data = data.loc[~(data == 0).any(axis=1)]

filter_name = pd.read_excel("highly_correlated_features.xlsx")
filter_name = filter_name["Highly Correlated Features"].to_list()

data_test=data['2024-11-20':'2024-12-09']

# %%
#Quote event class
#Quote event class
import pandas as pd
from datetime import datetime, timedelta
import time
import joblib
from sklearn.exceptions import NotFittedError
import dill
import requests
from back_testing.Backtest import ForwardTest

#%%
class skQ_event:
    def __init__(self,data_test):
        self.temp = []
        self.code = []       
        self.data = []
        self.test = []
        self.predicted_returns = pd.DataFrame()
        
        # 初始化字典來收集資料
        self.ticks_data = {
            'Datetime': [],
            'price': [],
            'volume': []
        }

        # 用來存儲所有收集的五分鐘K線資料
        self.all_kline_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        #資料長度
        self.len = data_test.shape[0]
        
        # 接收外部歷史數據
        self.data_test = data_test

        # 確保 data_test 的索引名稱和格式正確
        self.data_test.index = pd.to_datetime(self.data_test.index, errors='coerce')
        self.data_test.index.name = 'Datetime'
        
        # 加載已經訓練好的模型
        try:
            with open('TX_1min_model_5.pkl', 'rb') as f:
                saved_data = dill.load(f)
            self.lstm_model = saved_data['lstm_model']
            self.tcn_model = saved_data['tcn_model']
            self.ensemble_model = saved_data['ensemble_model']
            self.scaler_X = saved_data['scaler_X']
            self.scaler_y = saved_data['scaler_y']
            print("Models and scalers loaded successfully.")
        except FileNotFoundError:
            print("Trained model file not found. Please make sure 'TX_1min_model_5.pkl' is available.")
            self.lstm_model = None
            self.tcn_model = None
            self.ensemble_model = None
            self.scaler_X = None
            self.scaler_y = None
            
    def OnConnection(self, nKind, nCode):
        """內期主機回報"""
        print(f'skOSQ_OnConnection nCode={nCode}, nKind={nKind}')
        
    def OnNotifyCommodityListWithTypeNo(self, sMarketNo, bstrCommodityData):
        '''查詢海期/報價下單商品代號'''
        if "##" not in self.temp:
            self.temp.append(sMarketNo)
            self.code.append(bstrCommodityData.split(','))
        else:
            print("skOSQ_OverseaProductsDetail downloading is completed.")
    
    def OnNotifyKLineData(self, bstrStockNo, bstrData):      
        self.data.append(bstrData.split(","))       
        

    def OnNotifyTicksLONG(self, sMarketNo, nIndex, nPtr, nDate, nTimehms,
                          nTimemillismicros, nBid, nAsk, nClose, nQty, nSimulate):
        """
        收集tick資料並進行處理
        """
        print(f"接收到 tick 數據: 日期={nDate}, 時間={nTimehms}, 成交價={nClose/100}, 成交量={nQty}")
            
        # 收集資料
        datetime_str = f"{nDate}{nTimehms:06}"  # 格式化為 YYYYMMDDHHMMSS
        price = nClose / 100  # 成交價格
        volume = nQty  # 成交量

        # 將資料添加到字典中
        self.ticks_data['Datetime'].append(datetime_str)
        self.ticks_data['price'].append(price)
        self.ticks_data['volume'].append(volume)

        # 直接更新 DataFrame
        self.dataframe = pd.DataFrame(self.ticks_data)
        self.dataframe['Datetime'] = pd.to_datetime(self.dataframe['Datetime'], format='%Y%m%d%H%M%S')
        self.dataframe.set_index('Datetime', inplace=True)

        # 每次收到tick數據都嘗試進行5分鐘重取樣和預測
        self.resample_data()
        self.live_price_monitor()


    def resample_data(self):
        """
        將收集到的tick資料進行重取樣為5分鐘K線
        """
        
        # 進行 5 分鐘 K 線的重取樣
        ohlcv = self.dataframe.resample('5T').agg({
            'price': ['first', 'max', 'min', 'last'],  # open, high, low, close
            'volume': 'sum'  # 成交量
        })
        
        # 將每個重取樣的時間向前移動 5 分鐘，使其表示該五分鐘區間的結束時間
        ohlcv.index = ohlcv.index + pd.Timedelta(minutes=5)
        # 給列名稱加上正確的名稱
        
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']

        # 確保所有的 index 都是 datetime 格式
        ohlcv.index = pd.to_datetime(ohlcv.index, errors='coerce')
        ohlcv.index.name = 'Datetime'  # 確保索引名稱統一

        # 確保 data_test 的索引統一
        self.data_test.index = pd.to_datetime(self.data_test.index, errors='coerce')
        self.data_test.index.name = 'Datetime'  # 確保索引名稱統一
        
    
        # 檢查是否有新生成的五分鐘K線
        if not ohlcv.empty:
            # 比較新生成的 ohlcv 資料的行數與之前的行數
            if hasattr(self, 'prev_ohlcv_row_count'):
                prev_row_count = self.prev_ohlcv_row_count
            else:
                prev_row_count = 1

            new_row_count = ohlcv.shape[0]  # 使用 row 數量進行比較
            # print(ohlcv)

            # 確認是否有新增的五分鐘K線資料
            if new_row_count > prev_row_count:
                print(f"新五分鐘 K 線資料已生成，開始更新歷史數據。")

                # 更新最新的 K 線資料，使用倒數第二個
                latest_kline = ohlcv.iloc[[-2]]

                # 合併最新的 K 線資料到歷史數據中，避免重複行
                self.data_test = pd.concat([self.data_test, latest_kline]).sort_index()

                # 確保索引統一成 datetime 格式，防止錯誤
                self.data_test.index = pd.to_datetime(self.data_test.index, errors='coerce')
                self.data_test.index.name = 'Datetime'  # 確保索引名稱一致

                # 刪除重複的行（保留最新的）
                self.data_test = self.data_test[~self.data_test.index.duplicated(keep='last')]

                # # 打印合併後的歷史數據，確認數據是否正確
                # print("更新後的歷史數據：")
                # print(self.data_test.tail())

                # 新增重取樣的 K 線資料到 all_kline_data，避免重複
                new_kline_data = ohlcv[~ohlcv.index.isin(self.all_kline_data.index)]  # 過濾出不重複的資料
                self.all_kline_data = pd.concat([self.all_kline_data, new_kline_data]).sort_index()
                
                            # 更新行數計數器
                self.prev_ohlcv_row_count = new_row_count

                # 進行模型預測
                self.predict_with_latest_kline()
            else:
                print("目前沒有新生成的五分鐘K線資料。等待更多的tick數據以完成新K線生成。")
        else:
            print("重取樣的結果為空，等待更多的 tick 資料...")

            



    def predict_with_latest_kline(self):
        """
        使用最新的五分鐘K線資料進行預測
        """
        print("正在使用最新的 5 分鐘 K 線數據進行預測...")
        if self.lstm_model is not None:
            try:
                # 使用完整的歷史數據進行預測
                new_data = self.data_test
                
                # 資料更新數量
                update_row_num = new_data.shape[0] - self.len
                
                if update_row_num ==1:
                    self.backtest = ForwardTest(initial_balance=200000, transaction_fee=30, margin_rate=0.1, 
                                           stop_loss=0.0005, trailing_stop_pct=0.001, point_value=50,
                                           data_folder= "trading", symbol= "TX00",
                                           start_time="00:00",end_time="23:59")
                    self.backtest.load_trading_files()
                    

                # 前處理步驟
                alpha = AlphaFactory(new_data)
                indicator = TAIndicatorSettings()
                indicator2 = StockDataScraper()

                # 添加所有 alpha 特徵
                data_alpha = alpha.add_all_alphas()
                

                # 添加技術指標
                filtered_settings, timeperiod_only_indicators = indicator.process_settings()

                indicator_list = list(filtered_settings.keys())
                data_done0 = indicator2.add_specified_indicators(data_alpha, indicator_list, filtered_settings)
                data_done1 = add_all_ta_features(data_done0, open='open', high='high', low='low', close='close', volume='volume')

                data_done2 = indicator2.add_indicator_with_timeperiods(data_done1, timeperiod_only_indicators,
                                                                       timeperiods=[5, 10, 20, 60, 120, 240])
                
                self.code = data_done2
                # 處理特徵和標籤
                y = data_done2[['open', 'high', 'low', 'close', 'volume']]
                X = data_done2.drop(columns=['close'])
                
                # X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')

                # 添加日期特徵
                X['day'] = new_data.index.day
                X['minute'] = new_data.index.minute
                X['hour'] = new_data.index.hour
                
                # 選擇最終的特徵列
                filter_name = pd.read_excel("highly_correlated_features.xlsx")
                filter_name = filter_name["Highly Correlated Features"].to_list()
                X = X[filter_name]

                X_filter = pd.DataFrame(X, columns=filter_name)

                # 預處理輸入數據
                new_input_scaled = self.scaler_X.transform(X_filter.tail(update_row_num))

                # LSTM 預測
                new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32).to(device)

                # LightGBM 模型預測
                # prediction_lgbm_scaled = self.lgbm_model.predict(new_input_scaled).reshape(-1, 1)
                # prediction_lgbm = self.scaler_y.inverse_transform(prediction_lgbm_scaled)

                with torch.no_grad():
                    prediction_lstm_scaled = self.lstm_model(new_input_tensor).cpu().numpy()

                # 解縮放 LSTM 預測結果
                prediction_lstm = self.scaler_y.inverse_transform(prediction_lstm_scaled)
                
                new_input_tensor = new_input_tensor.unsqueeze(2)
                
                self.tcn_model.eval()
                with torch.no_grad():
                    prediction_tcn_scaled = self.tcn_model(new_input_tensor).cpu().numpy()               
                              
                prediction_tcn = self.scaler_y.inverse_transform(prediction_tcn_scaled)

                # 集成模型預測
                ensemble_input = np.hstack((prediction_lstm, prediction_tcn))
                prediction_ensemble = self.ensemble_model.predict(ensemble_input)

                # 動態報告預測結果
                self.report_prediction(prediction_lstm[-1].flatten())
                self.report_prediction(prediction_tcn[-1].flatten())
                self.report_prediction(prediction_ensemble[-1].flatten())
                               
                # Plot actual vs predicted values
                plt.figure(figsize=(14, 8))
                # plt.plot(X.tail(update_row_num).index ,prediction_lstm, label='Predict (LSTM)', color='blue')
                plt.plot(X.tail(update_row_num).index ,prediction_ensemble, label='Predict (Ensemble)', color='black')
                #plt.plot(X.tail(i+1).index, data_done2['close'].tail(i+1), label='Acutal', color='orange')
                plt.xlabel('Index')
                plt.ylabel('Return Rate')
                plt.title('Actual vs Predicted Values - LSTM, LightGBM, TabNet, and Ensemble Models')
                plt.legend()
                plt.show()
                
                # 計算預測回報率
                self.predicted_returns = pd.Series(prediction_lstm.flatten())

                # 使用保證金、停損、移動停利以及多/空倉進行回測
                self.bot = self.backtest.run_backtest(predicted_return = self.predicted_returns.iloc[update_row_num-1], 
                                                   real_time_data = y.iloc[self.len+update_row_num-1],
                                                    current_time=y.index[self.len+update_row_num-1],
                                                    buy_threshold=0.0002,
                                                    short_threshold=-0.0005,
                                                    osc_window = 6)

                 #機器人操作開始，才開啟報價監視器
                self.live_price_monitor()
                print("報價監視器更新")
                               
                for trade in self.bot['trade_log']:
                    print(trade)
                    
                print(self.bot['current_balance'])
                
                # 繪製未來預測與買入/賣出標記以及損益圖
                self.time = y.index[self.len:self.len+update_row_num-1]
                self.backtest.plot_profit_loss(time=y.index[self.len:self.len+update_row_num],
                                        return_rate=self.predicted_returns.iloc[:update_row_num],
                                        actual=data_done2[['open','high','low','close','volume']].iloc[self.len:self.len+update_row_num])

                # 輸出摘要表
                self.backtest.summary_table()
                
                if self.backtest.trade_log:
                    self.backtest.save_trading_files()

                
            except NotFittedError:
                print("Loaded model is not fitted yet.")
        else:
            print("Model is not loaded. Cannot make predictions.")
            
    def live_price_monitor(self):
        live_price  = self.dataframe.tail(1)
        
        if not self.predicted_returns.empty:
            print("監測中")
            self.backtest.monitor_stop_loss(live_price_data=live_price,
                                        osc_window=6,
                                        predicted_return=self.predicted_returns.iloc[-1],
                                        buy_threshold=0.0002,
                                        short_threshold=-0.0005)
        else:
            print("尚未有預測值")
            
        

    def report_prediction(self, prediction):
        """
        動態報出每次預測的結果
        """
        # 直接打印預測結果到控制台
        print(f"動態預測值報告: {prediction}")


               
    def OnNotifyHistoryTicksLONG(self, sMarketNo, nIndex, nPtr,  
                                 nDate, nTimehms,nTimemillismicros,
                                 nBid, nAsk, nClose, nQty, nSimulate):
        
        self.test.append(nClose)
        
    def OnNotifyBest5LONG(self, sMarketNo, nStockidx, nBestBid1, nBestBidQty1, nBestBid2,  nBestBidQty2, nBestBid3, nBestBidQty3, nBestBid4, nBestBidQty4, nBestBid5, nBestBidQty5, nExtendBid, nExtendBidQty, nBestAsk1, nBestAskQty1, nBestAsk2, nBestAskQty2, nBestAsk3, nBestAskQty3, nBestAsk4, nBestAskQty4, nBestAsk5, nBestAskQty5, nExtendAsk, nExtendAskQty, nSimulate):
        if (nSimulate == 0):
            labelnSimulate = "一般揭示"
        elif (nSimulate == 1):
            labelnSimulate = "試算揭示"


# SKReplyLib event handler
class skR_events:
    def OnReplyMessage(self, bstrUserID, bstrMessage):
        '''API 2.13.17 以上一定要返回 sConfirmCode=-1'''
        sConfirmCode = -1
        print('skR_OnReplyMessage ok')
        return sConfirmCode
#%%
# # 建立 event 跟 event handler 的連結

# Event sink, 事件實體化
# EventQ = skQ_event()
# 創建 skQ_event 的實例，並傳遞外部歷史數據
EventQ = skQ_event(data_test)
EventR = skR_events()
# %%
# 建立 event 跟 event handler 的連結
ConnOSQ = cc.GetEvents(skQ, EventQ)
ConnR = cc.GetEvents(skR, EventR)

# %%
# login
print('Login', skC.SKCenterLib_GetReturnCodeMessage(skC.SKCenterLib_Login(ID,PW)))


# %%
nCode = skQ.SKQuoteLib_EnterMonitorLONG()
print("SKOSQuoteLib_EnterMonitor", skC.SKCenterLib_GetReturnCodeMessage(nCode))
#%%
nCode = skQ.SKQuoteLib_RequestLiveTick(-1,'TX00')

print("SKQuoteLib_RequestTicks", skC.SKCenterLib_GetReturnCodeMessage(nCode[1]))
#%%
skQ.SKQuoteLib_LeaveMonitor()
# %%
nCode = skQ.SKQuoteLib_RequestStockList(2)
print("SKOSQuoteLib_RequestStockList", skC.SKCenterLib_GetReturnCodeMessage(nCode))
# %%
#儲存清單
percent_strings=[]
# Creating an empty list to store processed dictionaries
processed_entries = []

for num,tt in enumerate(EventQ.code):
# Iterating through each raw data list


    tt[0] = tt[0].rsplit('%',1)[1]
        # Iterating over the raw data in chunks of three
        
    for i in range(0, len(tt), 3):
        if len(tt) - i >= 3:
            # Creating a dictionary for each set of three elements
            entry = {
                    'Number': tt[i],
                    'Name': tt[i + 1],
                    'Date': tt[i + 2]
                }
            processed_entries.append(entry)

        # Converting the processed data to a DataFrame
    df_processed_list = pd.DataFrame(processed_entries)
    df_processed_list.to_excel(os.path.join(r'C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\scrapping\API\清單.xlsx'))


# %%

import pandas as pd
from datetime import datetime
import os
min_str = [1,5,60]

class Scraping_HistKline():
    def __init__(self):
        
        pass
    # sKlineType: 0 分 4 日 5 周 6月
    def RequestKLineAMByDate(self, bstrStockNo:str,
                             time_interval,
                             bstrStartDate,
                             bstrEndDate):
        
        """發出請求"""
        
        self.bstrStockNo = bstrStockNo
        self.freq = time_interval
        self.StartDate = bstrStartDate
        self.EndDate = bstrEndDate
        
        self.RequestKLineInfo = {'bstrStockNo':bstrStockNo,
                                 'freq':time_interval,
                                 'StartDate':bstrStartDate,
                                 'EndDate': bstrEndDate}
        
        #先檢查是否有資料在裏頭
        if not EventQ.data:
            print("Kline data is empty")
        else:
            EventQ.data.clear()
            
        nCode = skQ.SKQuoteLib_RequestKLineAMByDate(bstrStockNo=bstrStockNo,
                                                    sKLineType=0,
                                                    sOutType=1,
                                                    sTradeSession=0,
                                                    bstrStartDate=bstrStartDate,
                                                    bstrEndDate=bstrEndDate,
                                                    sMinuteNumber=time_interval)
        
        print("SKOSQuoteLib_RequestStockList", skC.SKCenterLib_GetReturnCodeMessage(nCode))
        
        


    # 定義函數來處理多行股票數據並存入 DataFrame

    def process_kline_data(self, kline_data, saving_path):
        """
        將包含多行股票數據的列表轉換為 DataFrame。

        參數:
            kline_data (list): 包含多行股票數據的列表，每行格式為
                            [日期時間, 開盤價, 最高價, 最低價, 收盤價, 成交量]

        返回:
            pd.DataFrame: 格式化後的數據 DataFrame，包含日期時間、開盤價、最高價、最低價、收盤價和成交量。
        """
        

        # 定義處理單行數據的函數
        def process_stock_data(data):
            timestamp = datetime.strptime(data[0].strip(), '%Y/%m/%d %H:%M')
            open_price = float(data[1].strip())
            high_price = float(data[2].strip())
            low_price = float(data[3].strip())
            close_price = float(data[4].strip())
            volume = int(data[5].strip())

            processed_data = {
                'date': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
            return processed_data
        
        # 處理多行數據
        processed_data_list = [process_stock_data(row) for row in kline_data]
        df = pd.DataFrame(processed_data_list).set_index('date')
        
            # 創建資料夾
        stock_folder = os.path.join(saving_path, self.bstrStockNo)
        os.makedirs(stock_folder, exist_ok=True)
        
        df.to_excel(os.path.join(stock_folder, f'{self.bstrStockNo}_{self.freq}_{self.StartDate}_{self.EndDate}.xlsx'))
        print(os.path.join(stock_folder, f'{self.bstrStockNo}_{self.freq}_{self.StartDate}_{self.EndDate}.xlsx'))
        return df
 


test = Scraping_HistKline()
#%%
time_list = {'time1':["20241101", "20241209"]}
min_str = [1,5,60]

for time in time_list:
    start_date, end_date = time_list[time]
    for freq in min_str:
        test.RequestKLineAMByDate("TX00",freq, start_date, end_date)
        df = test.process_kline_data(EventQ.data, r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API")


# %%
