#获得可传递的特征，可传递tensor
import torch,pandas as pd,numpy as np,random
from torch import tensor
from fc import timer,disk_cache
@timer
@disk_cache
def get_feature(x,):
    def feature_window_size(df,window_size):
        # 定义窗口大小（示例设为5，可根据需求调整）
        window_size = window_size
        # 批量计算基础统计指标（均值/标准差/极差等）
        rolling_stats = df['close'].rolling(window=window_size).agg(
            ['mean', 'std', 'min', 'max', 'median'])
        # 自定义统计指标（极差/IQR）
        def custom_range(x):
            return x.max() - x.min()
        def iqr(x):
            return np.quantile(x, 0.75) - np.quantile(x, 0.25)
        def slope(y_values):#滚动斜率
            # 若窗口内数据不足，返回NaN
            if len(y_values) < 2:
                return np.nan
            # 定义x为0到window-1（时间点或位置索引）
            x = np.arange(len(y_values))
            # 一阶线性拟合，返回斜率
            slope, _ = np.polyfit(x, y_values, 1)
            return slope
        rolling_stats['range'] = df['close'].rolling(window_size).apply(custom_range)
        rolling_stats['iqr'] = df['close'].rolling(window_size).apply(iqr)
        rolling_stats['slope'] = df['close'].rolling(window_size).apply(slope)
        rolling_stats['diff_col'] = df['close'].diff(periods=window_size)
        # 重命名列
        rolling_stats.columns = [f'rolling_{col}_{window_size}' for col in rolling_stats.columns]
        # 合并所有统计指标列
        return rolling_stats
    # 输入Series x转换为DataFrame
    df_x= x#pd.DataFrame({'x': x})
    rolling_stats=[]
    for i in [5,15,30,60]:
        rolling_stats+=[feature_window_size(df_x,window_size=i)]
    df = pd.concat([df_x]+rolling_stats, axis=1)
    df['diff_1'] = df['close'].diff(periods=1)
    return df

if __name__=='__main__':
    # 生成时间索引（从指定时间开始，间隔1分钟，共100个时间点）
    start_time = "2024-02-19 14:28:00"  # 初始时间点
    time_index = pd.date_range(         # 生成时间范围
        start=start_time,               # 起始时间
        periods=100,                    # 生成100个时间点
        freq="min"                        # 频率：每分钟（'T'或'min'）
    )

    # 创建DataFrame，填充随机数据（均值为0，标准差为1的正态分布）
    x = pd.DataFrame(
        data=np.random.randn(100, 1),   # 生成100行1列的随机数
        index=time_index,               # 设置时间索引
        columns=["close"]               # 列名
    )
    df=get_feature(x,)
    print(df.shape)
