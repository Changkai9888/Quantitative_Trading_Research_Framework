import pandas as pd,fc,numpy as np,pickle
file_path=r'C:\Quant工坊\TB数据下载\新版_TB录数据程序\to_save_data\\'
file_name=r'_get_data_easy_60s_000_20250507.11230181多的品种.tbf.feather'
df=pd.read_feather(file_path+file_name)
x=(df['a9000.DCE_close']/df['m9000.DCE_close']).dropna().iloc[:-1]
sp_diff=(df['a9000.DCE_close']-df['m9000.DCE_close']).dropna().diff().dropna().to_numpy()
# 保存对象到 .pkl 文件
with open('x.pkl', 'wb') as f:  # 注意：使用 'wb' 模式（写入二进制模式）
    pickle.dump(x, f)
