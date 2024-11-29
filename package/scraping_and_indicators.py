#%% Class to scrape prices and add indicators using yfinance
import yfinance as yf
import pandas as pd
import os
from talib import abstract
import talib
import requests
from FinMind.data import DataLoader
import numpy as np

class StockDataScraper:
    def __init__(self, dir_path=r"C:\Users\user\Desktop\investment\python\scrapping\scraping_data"):
        # Directory for saving files
        self.dir = dir_path
        os.makedirs(self.dir, exist_ok=True)

    def scrape_prices_yfinance(self, ticker, start_date, end_date):
        """
        Scrapes historical prices using yfinance and returns a DataFrame.

        Parameters:
            ticker (str): Stock ticker symbol (e.g., 'YINN').
            start_date (str): Start date for scraping in the format 'YYYY-MM-DD'.
            end_date (str): End date for scraping in the format 'YYYY-MM-DD'.

        Returns:
            pd.DataFrame: DataFrame containing the stock price data.
        """
        try:
            # Step 1: Scrape historical price data
            print(f"Scraping historical data for {ticker} from {start_date} to {end_date}...")
            df = yf.download(ticker, start=start_date, end=end_date)

            # Check if data is available
            if df.empty:
                print(f"No data found for {ticker}. Please verify the ticker symbol or date range.")
                return pd.DataFrame()

            # Step 2: Rename columns to lowercase to ensure consistency
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }, inplace=True)

            return df

        except Exception as e:
            print(f"Error occurred while scraping data for {ticker}: {e}")
            return pd.DataFrame()

    def add_indicator_with_timeperiods(self, df, indicator_list, timeperiods=[20]):
        new_columns = {}

        for ind in indicator_list:
            ind_upper = ind.upper()
            indicator_function = abstract.Function(ind_upper)

            # Iterate over each time period and apply the indicator
            for period in timeperiods:
                # Calculate the indicator for the given period
                output = indicator_function(df, timeperiod=period)

                # Assign the calculated indicator to the new_columns dictionary
                column_name = f'{ind_upper}_{period}'
                new_columns[column_name] = output

        # Use pd.concat to add all new columns to the DataFrame at once
        new_columns_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_columns_df], axis=1)

        return df

    def add_specified_indicators(self, df, indicator_list, setting=None):
        available_indicators = talib.get_functions()

        for ind in indicator_list:
            ind_upper = ind.upper()
            if ind_upper not in available_indicators:
                print(f"警告: 指標 '{ind}' 不支援。")
                continue

            try:
                # 獲取指標函數
                indicator_function = abstract.Function(ind_upper)
                if setting is None or ind_upper not in setting:
                    output = indicator_function(df)
                else:
                    output = indicator_function(df, **setting[ind_upper])

                # 步驟 1：確認輸出是 Series 還是 DataFrame 或 tuple
                if isinstance(output, pd.Series):
                    output.name = ind
                    df = pd.concat([df, output], axis=1)

                elif isinstance(output, pd.DataFrame):
                    # 如果是 DataFrame，將其每一列加入 df
                    for col in output.columns:
                        if output[col].isna().all():
                            print(f"警告: 指標 '{ind}' 的欄位 '{col}' 返回的結果都是 NaN，跳過。")
                        else:
                            df[f"{ind.lower()}_{col}"] = output[col]

                elif isinstance(output, tuple):
                    # 處理返回多個結果的情況（如 MACD, STOCH 等）
                    for idx, item in enumerate(output):
                        if isinstance(item, pd.Series):
                            df[f'{ind.lower()}_{idx}'] = item
                        elif isinstance(item, pd.DataFrame):
                            for col in item.columns:
                                df[f'{ind.lower()}_{idx}_{col}'] = item[col]

            except Exception as e:
                print(f"無法加入指標 '{ind}': {e}")

        return df

    def calculate_profit_or_loss(self, df, days=20, column_name='adj_close'):
        """
        Calculate the profit or loss after a given number of days and add it as a new column to the dataframe.

        Parameters:
        df (pd.DataFrame): DataFrame containing price data with specified column.
        days (int): Number of days after which to calculate profit or loss.
        column_name (str): Column name to use for profit/loss calculation (e.g., 'adj_close' or 'close').

        Returns:
        pd.DataFrame: Updated DataFrame with a new column for profit or loss.
        """
        df[f'profit_or_loss_after_{days}_days'] = df[column_name].shift(-days) - df[column_name]
        return df

    def calculate_avg_return(self, df, days=20, column_name='adj_close'):
        """
        Calculate the average return over a given number of days and add it as a new column to the dataframe.

        Parameters:
        df (pd.DataFrame): DataFrame containing price data with the specified column.
        days (int): Number of days over which to calculate the average return.
        column_name (str): Column name to use for the return calculation (e.g., 'adj_close' or 'close').

        Returns:
        pd.DataFrame: Updated DataFrame with a new column for the average return over the given days.
        """
        # Initialize a list to hold the average returns
        returns = []

        # Iterate over the DataFrame to calculate daily returns over the next 'days' period
        for i in range(len(df) - days):
            # Calculate daily returns for the next 'days' days
            daily_returns = [(df[column_name].iloc[i + j + 1] - df[column_name].iloc[i + j]) / df[column_name].iloc[i + j]
                             for j in range(days)]
            
            # Calculate the average return for the next 'days' period
            avg_return = sum(daily_returns) / days
            returns.append(avg_return)

        # Check the length of returns vs. the rows from 'days' onward
        print(f"Returns length: {len(returns)}, DataFrame length: {len(df)}")

        # Add the calculated returns to the dataframe starting from index 'days'
        df[f'avg_return_after_{days}_days'] = pd.NA  # Initialize with NA

        # Assign the 'returns' list starting from index 0
        df.iloc[:len(returns), df.columns.get_loc(f'avg_return_after_{days}_days')] = returns
        return df

    def filter_data_by_date_range(self, dataframe, start_date, end_date):
        """
        Filters the data within the specified date range.
        
        Parameters:
        - dataframe (pd.DataFrame): DataFrame indexed by 'date'.
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.
        
        Returns:
        - pd.DataFrame: Filtered DataFrame containing rows within the date range.
        """
        mask = (dataframe.index >= start_date) & (dataframe.index <= end_date)
        return dataframe[mask]

    # 修改 process_taiwan_margin_purchase_short_sale 方法
    def process_taiwan_margin_purchase_short_sale(self, start_date, end_date=None, additional_df=None, token=None, user_id=None, password=None):
        """
        Retrieves Taiwan stock margin purchase and short sale data, pivots it by investor type, and returns the processed DataFrame.
        
        Parameters:
        - start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.
        - end_date (str): End date for data retrieval in 'YYYY-MM-DD' format.
        - additional_df (pd.DataFrame, optional): Additional DataFrame to merge with the processed data.
        - token (str, optional): API token for authentication.
        - user_id (str, optional): User ID for login.
        - password (str, optional): Password for login.
        
        Returns:
        - pd.DataFrame: DataFrame containing the processed margin purchase and short sale data.
        """
        api = DataLoader()
        if token:
            api.login_by_token(api_token=token)
        elif user_id and password:
            api.login(user_id=user_id, password=password)
        
        df = api.taiwan_stock_margin_purchase_short_sale_total(
            start_date=start_date,
            end_date=end_date
        )
        
        # Pivot the data by investor type
        pivot_df = df.pivot_table(
            index=['date'],
            columns='name',
            values=['TodayBalance', 'YesBalance', 'buy', 'Return', 'sell'],
            aggfunc='first'
        )
        pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
        pivot_df.index = pd.to_datetime(pivot_df.index).strftime('%Y-%m-%d')

        # Merge the pivoted DataFrame with the additional input DataFrame if provided
        if additional_df is not None:
            additional_df.index = pd.to_datetime(additional_df.index).strftime('%Y-%m-%d')
            merged_df = pd.merge(pivot_df, additional_df, left_index=True, right_index=True, how='outer')
        else:
            merged_df = pivot_df

        # Fill NaN values to handle missing data more effectively
        merged_df.fillna(0, inplace=True)

        return merged_df
    


# %%