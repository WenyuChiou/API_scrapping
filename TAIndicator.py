#%%
import talib
from talib import abstract


class TAIndicatorSettings:
    def __init__(self):
        # 初始化存储指标参数的字典和只包含 timeperiod 参数的指标列表
        self.custom_settings = {}
        self.timeperiod_only_indicators = []

    def process_settings(self):
        """初始化指标参数，移除只包含 'timeperiod' 参数并且具有多个输出的指标，以及特定指标"""
        # 初始化指标参数
        for category in talib.get_function_groups().keys():
            # 获取每个分类下的所有指标
            indicators = talib.get_function_groups()[category]

            # 遍历每个指标
            for indicator in indicators:
                # 获取指标的函数对象
                func = abstract.Function(indicator)

                # 获取该指标的参数
                params = func.info['parameters']
                
                # 将所有指标添加到 custom_settings 中
                self.custom_settings[indicator] = {key: value for key, value in params.items()}

        # 过滤只包含 'timeperiod' 参数的指标
        timeperiod_only_indicators = [
            indicator for indicator, params in self.custom_settings.items()
            if len(params) == 1 and 'timeperiod' in params
        ]

        # 从 custom_settings 中移除只包含 'timeperiod' 参数 的指标
        self.custom_settings = {
            indicator: params for indicator, params in self.custom_settings.items()
            if indicator not in timeperiod_only_indicators
        }
        
        # 只保留输出数量为 1 的那些 'timeperiod' 参数的指标
        self.timeperiod_only_indicators = [
            indicator for indicator in timeperiod_only_indicators
            if len(abstract.Function(indicator).info['output_names']) == 1
        ]


        # 初始化要移除的特定指标列表（包含 MAVP, ACOS, ASIN，以及所有 CDL 开头的指标）
        indicators_to_remove = ['MAVP', 'ACOS', 'ASIN']
        indicators_to_remove += [indicator for indicator in self.custom_settings if indicator.startswith('CDL')]

        # 从 custom_settings 中移除这些特定指标
        self.custom_settings = {
            indicator: params for indicator, params in self.custom_settings.items()
            if indicator not in indicators_to_remove
        }

        # 返回过滤后的 custom_settings 和包含 timeperiod 的指标列表
        return self.custom_settings, self.timeperiod_only_indicators

    def update_indicator_params(self, indicator_name, new_params):
        """
        更新 filtered_custom_settings 中某个指标的参数
        参数:
        - indicator_name: 指标的名称 (str)
        - new_params: 新的参数字典 (dict)
        """
        if indicator_name in self.custom_settings:
            # 更新参数
            self.custom_settings[indicator_name].update(new_params)
            print(f"Updated {indicator_name} with new parameters: {new_params}")
        else:
            print(f"Indicator '{indicator_name}' not found in filtered settings.")

# 示例用法
if __name__ == "__main__":
    # 初始化类对象并处理设置
    indicator_settings = TAIndicatorSettings()
    filtered_settings, timeperiod_only_indicators = indicator_settings.process_settings()  # 处理所有步骤并获取结果

    # 打印结果
    print("Filtered Custom Settings (without CDL, MAVP, ACOS, ASIN indicators):")
    print(filtered_settings)
    print("\nIndicators with only 'timeperiod' parameter and single output:")
    print(timeperiod_only_indicators)

    # 更新 filtered_settings 中的参数示例
    indicator_settings.update_indicator_params('RSI', {'timeperiod': 20})  # 修改 RSI 的 timeperiod 参数为 20
    indicator_settings.update_indicator_params('MACD', {'fastperiod': 10, 'slowperiod': 30})  # 修改 MACD 的参数

    # 打印更新后的 filtered_settings
    print("\nUpdated Filtered Custom Settings:")
    updated_filtered_settings, _ = indicator_settings.process_settings()
    print(updated_filtered_settings)

# %%
