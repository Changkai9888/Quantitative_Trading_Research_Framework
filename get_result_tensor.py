#用于获得结果多元分析，兼容tenso传递
import torch,fc,pandas as pd,numpy as np
from torch import tensor
def get_right(x,pos,cost,bonus):
    #bonus:做单奖励
    if len(x)!=len(pos):
        raise ValueError("输入序列长度不等！")
    if str(type(x))=="<class 'pandas.core.series.Series'>":
        x=torch.tensor(x.values, device=pos.device)
    for k,i in enumerate((x,pos)):
        if str(type(i))!="<class 'torch.Tensor'>" or len(i.shape)!=1:
            raise ValueError(f"输入变量 {['x','pos'][k]} 的格式错误！")
    right_diff=torch.diff(x)*pos[:-1]-cost*abs(torch.diff(pos[:-1],prepend=tensor([0.], device=pos.device)))
    right=torch.cumsum(right_diff,dim=0)
    right=torch.cat([tensor([0.], device=pos.device),right],dim=0)#补充统一长度
    right_bonus=torch.sum(torch.diff(x)*pos[:-1]+(bonus-cost)*abs(torch.diff(pos[:-1],prepend=tensor([0.], device=pos.device))))
    return right,right_bonus
