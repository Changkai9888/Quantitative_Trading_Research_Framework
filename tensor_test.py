import torch,pickle,fc,pandas as pd,numpy as np,time
from fc import plot_trade
from get_result_tensor import get_right
from torch import tensor
import torch,torch.optim as optim
from get_feature_tensor import get_feature
from get_pos_NET import get_pos_NET
import torch.nn.functional as F
with open("x.pkl", "rb") as f:
    close = pickle.load(f)
#close=close.iloc[:1000]#测试截断
df=pd.DataFrame({'close': close})
df.index=pd.to_datetime((df.index*10000).round(), format='%Y%m%d%H%M')##转换为时间格式
df=get_feature(df)
print (df.shape)
for k,i in enumerate(df.isna().any(axis=1)):
  if i==False:
      break
batch_lon=100
df=df.iloc[k:(len(df-k-1)//batch_lon-1)*batch_lon+k]#截断
if df.isnull().values.any()==True:
    raise ValueError('df含有异常值！')
####
device = torch.device("cuda");print(f"训练设备: {device}")
x=torch.tensor(df.values, dtype=torch.float32, device=device)
x=x.reshape((len(x)//batch_lon, batch_lon, x.shape[-1]))
net=get_pos_NET(x).to(device)
optimizer = optim.NAdam(net.parameters(), lr=0.001)
ti=time.time()
for i in range(40000):
    pos=net(x).reshape(len(x)*batch_lon)
    cost=0.001
    ri,right_bonus=get_right(df.close,pos,cost=cost, bonus=max(0,cost*(1-0.0001*i)) )#退火方法
    loss=-right_bonus
    loss.backward();    optimizer.step();    optimizer.zero_grad()
    if i%1000==0:
        print(i,ri[-1].item(), right_bonus.item(), time.time()-ti);ti=time.time()
        plot_trade(df.close,pos.cpu(),ri.cpu(),save=f'plot\\{i}')
plot_trade(df.close,pos.cpu(),ri.cpu())
