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
# %%
#Quote event class
class skQ_event:
    def __init__(self):
        self.temp = []
        self.code = []       
        self.data = []
        
        
    
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
EventQ = skQ_event()
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
                                                    sTradeSession=1,
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
time_list = {'time1':["20240315", "20241124"],
             'time2':["20231115", "20240314"]}
min_str = [1,5,60]

for time in time_list:
    start_date, end_date = time_list[time]
    for freq in min_str:
        test.RequestKLineAMByDate("MTX00",freq, start_date, end_date)
        df = test.process_kline_data(EventQ.data, r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API")


# %%
