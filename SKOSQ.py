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
# 登錄物件
if 'skC' not in globals(): skC = cc.CreateObject(sk.SKCenterLib, interface=sk.ISKCenterLib)
# 下單物件
if 'skO' not in globals(): skO = cc.CreateObject(sk.SKOrderLib , interface=sk.ISKOrderLib)
# 海期報價物件
if 'skOSQ' not in globals(): skOSQ = cc.CreateObject(sk.SKOSQuoteLib , interface=sk.ISKOSQuoteLib)
# 回報物件
if 'skR' not in globals(): skR = cc.CreateObject(sk.SKReplyLib, interface=sk.ISKReplyLib)
# %%
#Quote event class
class skOSQ_events:
    def __init__(self):
        self.temp = []
        self.code = []       
        self.data = []
        
        
    
    def OnConnection(self, nKind, nCode):
        """內期主機回報"""
        print(f'skOSQ_OnConnection nCode={nCode}, nKind={nKind}')
        
    def OnOverseaProducts(self,  bstrValue):
        '''查詢海期/報價下單商品代號'''
        if "##" not in self.temp:
            self.temp.append( bstrValue)
            # self.code.append(bstrCommodityData.split(','))
        else:
            print("skOSQ_OverseaProductsDetail downloading is completed.")
    
    def OnKLineData(self, bstrStockNo, bstrData):      
        self.data.append(bstrData.split(","))       
        

# SKReplyLib event handler
class skR_events:
    def OnReplyMessage(self, bstrUserID, bstrMessage):
        '''API 2.13.17 以上一定要返回 sConfirmCode=-1'''
        sConfirmCode = -1
        print('skR_OnReplyMessage ok')
        return sConfirmCode


# # 建立 event 跟 event handler 的連結

# Event sink, 事件實體化
EventOSQ = skOSQ_events()
EventR = skR_events()

# 建立 event 跟 event handler 的連結
ConnOSQ = cc.GetEvents(skOSQ, EventOSQ)
ConnR = cc.GetEvents(skR, EventR)


# # 登入及各項初始化作業

# login
print('Login', skC.SKCenterLib_GetReturnCodeMessage(skC.SKCenterLib_Login(ID,PW)))

# 海期商品初始化
nCode = skOSQ.SKOSQuoteLib_Initialize()
print("SKOSQuoteLib_Initialize", skC.SKCenterLib_GetReturnCodeMessage(nCode))


###################################################################################
# 以下皆以手動輸入
# 登入海期報價主機
nCode = skOSQ.SKOSQuoteLib_LeaveMonitor()
nCode = skOSQ.SKOSQuoteLib_EnterMonitorLONG()
print('SKOSQuoteLib_EnterMonitorLONG()', skC.SKCenterLib_GetReturnCodeMessage(nCode))

#%%
ncode = skOSQ.SKOSQuoteLib_RequestOverseaProducts()
print('SKOSQuoteLib_RequestOverseaProducts()', skC.SKCenterLib_GetReturnCodeMessage(nCode))
# %%
# Creating an empty list to store processed dictionaries
processed_entries = []

for num, tt in enumerate(EventOSQ.temp):
    # Split the data by comma
    tt = tt.split(',')
    
    # Iterating through the list in chunks of 6
    for i in range(0, len(tt), 6):
        if len(tt) - i >= 6:  # Ensure there are enough elements to create an entry
            # Creating a dictionary for each set of six elements
            entry = {
                '代號': tt[i],
                '交易所': tt[i + 1],
                'code': tt[i + 2],
                '中文': tt[i + 3],
                '日期': tt[i + 4],
                '日期2': tt[i + 5]
            }
            processed_entries.append(entry)

# Converting the processed data to a DataFrame
df_processed_list = pd.DataFrame(processed_entries)

#%%
# Saving the DataFrame to an Excel file
output_file_path = r'C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\scrapping\API\清單海期.xlsx'
df_processed_list.to_excel(output_file_path, index=False)

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
        if not EventOSQ.data:
            print("Kline data is empty")
        else:
            EventOSQ.data.clear()
            
        nCode = skOSQ.SKOSQuoteLib_RequestKLineByDate(bstrStockNo=bstrStockNo,
                                                    sKLineType=0,
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
                'low': low_price,""
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
time_list = {'time1':["20241101", "20241203"]}
min_str = [1,5,60]

for time in time_list:
    start_date, end_date = time_list[time]
    for freq in min_str:
        test.RequestKLineAMByDate("CME,MNQ0000",freq, start_date, end_date)
        df = test.process_kline_data(EventOSQ.data, r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API")


# %%
