import torch,fc,pandas as pd,numpy as np
import numpy as np
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
class get_pos_NET(nn.Module): # 网络属于nn.Module类
    def __init__(net,x):
        super(get_pos_NET, net).__init__()
        net.input_size=x.shape[-1]
        mid_size=net.input_size//8*8+8
        def mod_line(insize, outsize):
            return nn.Sequential(nn.Linear(insize, mid_size),nn.ReLU(),  nn.Linear(mid_size, 16),\
                                 nn.ReLU(),nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, outsize) )
        net.lin1=mod_line(net.input_size,2)
        print('参数个数：'+str(sum(p.numel() for p in net.parameters())))
    def forward(net, x):
        x = net.lin1(x)
        x = F.tanh(x[...,1])
        pos=x.squeeze(-1)
        return pos
if __name__=='__main__':
    import pandas as pd
    device = torch.device("cuda");print(f"训练设备: {device}")
    x=torch.rand(5, 1000, 15, dtype=torch.float32, device=device)
    pos_net=get_pos_NET(x).to(device)
    pos=pos_net(x)
    print(pos.shape)
