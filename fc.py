import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import copy
from scipy.interpolate import make_interp_spline
import os
import sys,time
import pickle
import time,math,copy,inspect,hashlib,gzip
from datetime import datetime, timedelta
from functools import wraps
import statsmodels as sm
import multiprocessing
#import statsmodels.api
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
####
def sign_threshold(arr,threshold=0):#对array进行，带阈值的符号化处理。
    arr = np.copy(arr)
    if threshold<0:
        raise ValueError("threshold应该大于等于0")
    arr[abs(arr)<=threshold]=0
    return np.sign(arr)
def StandardScaler(df_x,y=None, shift=20,if_std=True):#对样本df, 滑动平均标准化
    x=(df_x-df_x.rolling(window=shift).mean())
    if if_std ==True:
        x=x/(df_x.rolling(window=shift).std())#+1e-10)
    if 'const' in df_x.keys():
        x['const']=1
    if '1d' in df_x.keys():
        x['1d']=df_x['1d']
    x=x.iloc[shift:]
    if x.isnull().values.any():# 检查DataFrame中是否有NaN值
        raise ValueError("DataFrame contains NaN values.")
    if np.isinf(x.values).any():# 检查DataFrame中是否有无穷大值
        raise ValueError("DataFrame contains infinite values.")
    if type(y)!=type(None):
        y=y.iloc[shift:]
        return x,y
    else:
        return x
def LR_std(y):
    '计算线性回归的残差的方差'
    slope, intercept = np.polyfit(range(len(y)), y, 1)
    residuals = y - slope * np.array(range(len(y))) - intercept
    return np.std(residuals)
###数列画图###
def plot(lit,k=0,save='0',s=0,figsize=0,zoom='none'):
    #k表示是一组还是多组，默认一组,多组时候，lit为list of list,save值为保存路径#
    #zoom表示自适应缩放，只在k=1时有效
        #zoom='max'按照最大值最小值差值进行归一
        #zoom='auto'按照减均值，除方差，进行标准化
    if str(type(lit))=="<class 'torch.Tensor'>":
        lit=lit.tolist()
    if zoom=='auto' and k==1:
        lit_real=[]
        for i in lit:
            mean_temp=np.mean(i)
            std_temp=np.std(i)
            lit_real+=[[(t-mean_temp)/std_temp for t in i]]
        lit=lit_real
    if zoom=='max' and k==1:
        lit_real=[]
        for i in lit:
            mean_temp=np.mean(i)
            width_temp=max(i)-min(i)
            lit_real+=[[(t-mean_temp)/width_temp for t in i]]
        lit=lit_real
    if figsize!=0:
        plt.figure(figsize=figsize)
    if s==0:
        if k==0:
            lon=range(len(lit))
            plt.plot(lon,lit)
        else:
            for i in lit:
                plot(i,save='1')
        if save =='0':
            if k==1:
                pass
                #plot(len(lit[0])*[0])
            plt.show();plt.close()
        elif save!='1':
            plt.savefig(save);plt.close()
    #s!=0,二值图表分析
    if s!=0:
        a,b=lit
        b=np.array(b)
        c=(max(a)-min(a))/(max(b)-min(b))
        plot([a,(b-min(b))*c+min(a)],k=1)
#交易结果画图
def plot_trade(close,pos,right,save=None):
    a,b,c=(i.detach().numpy() if str(type(i))=="<class 'torch.Tensor'>" else i for i in (close,pos,right))
    # 检验数据
    if not(len(a)==len(b)==len(c)):
        raise ValueError("输入序列长度不等！")
    lon=len(a)
    x = list(range(lon))
    # 创建共享坐标轴的子图
    fig, axs = plt.subplots(3, 1, figsize=(9, 6),  height_ratios=[3, 1, 1],
                            sharex=True, sharey=False)  # 关键参数[1,7](@ref)
    plt.subplots_adjust(hspace=0.05, right=0.9)  # 控制子图间距[6](@ref)
    # 绘制每个子图
    axs[0].plot(x, a, 'r', label='close')
    #axs[0].set_ylim(min_val - margin, max_val + margin)
    #axs[0].legend()#loc='upper left')               # 去掉图例边框（可选）[7](@ref))
    axs[0].legend(loc='upper left',bbox_to_anchor=(1.0, 1),borderaxespad=-0.5)  # 右侧外定位[4,7](@ref)) 
    axs[0].set_title('交易结果', y=1.1)  # 标题上移

    axs[1].plot(x, [0]*len(x),'k--',linewidth=0.5)
    axs[1].plot(x, [1]*len(x),'k--',linewidth=0.5)
    axs[1].plot(x, [-1]*len(x),'k--',linewidth=0.5)
    axs[1].plot(x, b, 'g', label='pos') 
    axs[1].set_ylim(-1.2, 1.2)
    axs[1].legend(loc='upper left',bbox_to_anchor=(1.0, 1),borderaxespad=-0.5)#loc='upper left')
    axs[1].tick_params(labelbottom=False)  # 隐藏中间图的x轴标签[6](@ref)

    axs[2].plot(x, c, 'b', label='right')
    axs[2].legend(loc='upper left',bbox_to_anchor=(1.0, 1),borderaxespad=-0.5)#loc='upper left')
    axs[2].set_xlabel('索引')  # 仅底部显示x轴标签[1](@ref)

    # 同步缩放设置
    for ax in axs:
        ax.label_outer()  # 隐藏内部冗余标签[7](@ref)
        ax.grid(alpha=0.3)  # 添加辅助网格
    if save ==None:
        plt.show();plt.close()
    elif str(type(save))!="<class 'str'>":
        raise ValueError('save应该是str格式！')
    else:
        plt.savefig(save);plt.close()
#装饰器：函数的计时功能。
def timer(func):
    """装饰器：计算函数运行时间"""
    @wraps(func)  # 保留原函数元信息（如函数名、文档说明）[4,7](@ref)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 使用高精度计时器（推荐）[6](@ref)
        result = func(*args, **kwargs)    # 执行原函数
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"函数 {func.__name__} 运行耗时: {duration:.6f} 秒")  # 输出微秒级精度[6](@ref)
        return result
    return wrapper

#装饰器：装饰器：加快反复调用 很耗时的 相同函数 相同参数 时的速度。
def disk_cache(func):
    '''装饰器：加快反复调用 很耗时的 相同函数 相同参数 时的速度。
    相同的 函数 传递相同的 参数 时候，第一次调用计算结果保存到硬盘，第二次则不计算直接调用保存的结果进行输出。
    保存在"./function_cache"，每次会比对函数的代码内容是否变化？参数传递是否变化？
    每日首次运行时，自动检测，如果保存的结果在近3日没有被调用，则自动删除，防止硬盘空间不足。
    对大文件进行压缩。'''
    """硬盘缓存装饰器（集成GZIP压缩+函数指纹+参数指纹+过期清理）"""
    CACHE_ROOT = "./function_cache"
    CACHE_DAYS = 3
    def cleanup_old_cache():
        """清理过期缓存（基于网页9的目录时间戳比对）"""
        now = datetime.now()
        for func_dir in os.listdir(CACHE_ROOT):
            dir_path = os.path.join(CACHE_ROOT, func_dir)
            timestamp_file = os.path.join(dir_path, "timestamp.txt")
            if os.path.exists(timestamp_file):
                with open(timestamp_file, 'r') as f:
                    cache_time = datetime.fromisoformat(f.read())
                if abs((now - cache_time).days) > CACHE_DAYS:
                    # 删除整个过期目录（网页10的shutil.rmtree方法）
                    shutil.rmtree(dir_path)
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 1. 生成函数版本指纹（网页5的SHA256哈希方法）
        func_code = inspect.getsource(func).encode()
        func_hash = hashlib.sha256(func_code).hexdigest()[:16]
        
        # 2. 生成参数指纹（网页6的pickle序列化+MD5哈希）
        param_data = pickle.dumps((args, kwargs))
        param_hash = hashlib.md5(param_data).hexdigest()
        
        # 3. 创建缓存目录
        cache_dir = os.path.join(CACHE_ROOT, f"func_{func_hash}")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 4. 缓存文件路径（保持.pkl后缀但实际为gzip压缩文件）
        cache_file = os.path.join(cache_dir, f"param_{param_hash}.pkl")
        timestamp_file = os.path.join(cache_dir, "timestamp.txt")
        
        # 5. 检查缓存有效性（网页7的流式压缩读法）
        if os.path.exists(cache_file):
            # 检查时间戳是否在3天内（网页9的时间比对逻辑）
            with open(timestamp_file, 'r') as f:
                cache_time = datetime.fromisoformat(f.read())
            with open(timestamp_file, 'w') as f:
                f.write(datetime.now().isoformat())#更新时间戳
            if datetime.now() - cache_time >= timedelta(days=CACHE_DAYS):
                # 运行过期清理
                cleanup_old_cache()
            # 使用gzip流式解压读取（网页3的GzipFile方法）
            with gzip.open(cache_file, 'rb') as f:
                return pickle.load(f)
        # 6. 执行计算并保存结果（网页7的流式压缩写法）
        result = func(*args, **kwargs)
        # 使用gzip压缩序列化（网页6的pickle+gzip组合）
        with gzip.open(cache_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        # 更新时间戳（网页9的时间记录方法）
        with open(timestamp_file, 'w') as f:
            f.write(datetime.now().isoformat())
        # 7. 触发过期清理（网页10的目录级清理逻辑）
        cleanup_old_cache()
        return result
    return wrapper

######AR模型##
def ar_a(lit,k=0):#k为滑动取值
    a=0
    b=0
    lits=copy.deepcopy(lit)
    lits=np.array(lits)
    if k==0:
        lits=lits-lits.mean()
        for i in range(len(lits)):
            if i >=1:
                a+=lits[i]*lits[i-1]
            if 1<=i<=len(lits)-2:
                b+=lits[i]**2
        return a/b
    else:
        lon=len(lits)
        t=[]
        for i in range (lon-k+1):
            t+=[ar_a(lits[i:i+k])]
        return t
###############Bolling布林通道#参数m,n,k
def Boll_index(tradtolist,m=26,n=26,k=2,bar=120):
    smooth_mbar=mean_smooth(tradtolist,bar*m)
    smooth_1bar=mean_smooth(tradtolist,bar)
    p_mean=np.mean(tradtolist)
    y=[]
    z=[]
    a=copy.deepcopy(smooth_mbar)
    p0=copy.deepcopy(smooth_1bar)
    singnal=[]
    for i in range(len(tradtolist)):
        if i>=bar*(1+max(m,n)):
            y+=[smooth_mbar[i]+k*np.std(smooth_1bar[i-bar*n:i])]
            z+=[smooth_mbar[i]-k*np.std(smooth_1bar[i-bar*n:i])]
            singnal+=[-1 if p0[i]>=y[-1] else 1 if  p0[i]<=z[-1]  else 0 ]
        else:
            y+=[p_mean]
            z+=[p_mean]
            p0[i]=p_mean
            a[i]=p_mean
            singnal+=[0]
    x=[y,z,a,p0,singnal]#y高,z低,a长均,p0短均，singnal,符号
    return x
####
def Boll_sig(a,m=26,k=2):
    a=a[-m:]
    mean=np.mean(a)
    std=np.std(a)
    sort=[mean-std*k, mean-std*(k-1), mean, mean,mean+std*(k-1),mean-std*k]
    c=-3
    for i in sort:
        if i>a[-1]:
            c+=1
    return c
####
def SAR_sig(a,k0=1):
    c=0;sig=0
    acc=0.02;k=1
    sar=min(a[:4])
    sar_list=[]
    for i in range(4,len(a)):
        if sig!=-1:
            if a[i]>max(a[c:i]):
                k=min(k+k0,10)
            if a[i]<sar:
                sar=max(a[c:i+1])
                c=i;k=k0;sig=-1
            else:
                sar=sar+acc*k*(a[i]-sar)
        elif sig!=1:
            if a[i]<min(a[c:i]):
                k=min(k+k0,10)
            if a[i]>sar:
                sar=min(a[c:i+1])
                c=i;k=k0;sig=1
            else:
                sar=sar+acc*k*(a[i]-sar)
        sar_list+=[[a[i],sar]]
    return [sig,a[-1]-sar,i-c]
###############SAR模型
def SAR_index(tradp,a=4, b=2, c=20, bar=120):
    index=[0]*len(tradp);s=0;temp=0;af=0.01*c;t=0
    for i in range(bar*(a+1),len(tradp)):
        if s==0 and tradp[i]>max(tradp[i-bar*a:i])*(1-af)+af*min(tradp[i-bar*a:i]):
            s=1;index[i]=min(tradp[i-bar*a:i])*s; af=0.01*b;continue
        if s==0 and tradp[i]<min(tradp[i-bar*a:i])*(1-af)+af*max(tradp[i-bar*a:i]):
            s=-1;index[i]=max(tradp[i-bar*a:i])*s; af=0.01*b; continue
        if s==1:
            t+=1
            if t%bar==0:
                temp=abs(index[i-1])+af*(max(tradp[i-bar*a:i])-abs(index[i-1]))
                minp=min(tradp[i-bar:i])
                if temp>minp:
                    s=-1;index[i]=max(tradp[i-bar*a:i])*s;af=0.01*b;t=0;continue
                elif temp<=minp:
                    index[i]=temp*s
                    if max(tradp[i-bar*(a+1):i-bar])<max(tradp[i-bar:i]):
                        af+=0.01*b if af<=c*0.01 else 0
            else:
                index[i]=index[i-1]
        if s==-1:
            t+=1
            if t%bar==0:
                temp=abs(index[i-1])+af*(min(tradp[i-bar*a:i])-abs(index[i-1]))
                maxp=max(tradp[i-bar:i])
                if temp<maxp:
                    s=1;index[i]=min(tradp[i-bar*a:i])*s;af=0.01*b;t=0;continue
                elif temp>=maxp:
                    index[i]=temp*s;
                    if min(tradp[i-bar*(a+1):i-bar])>min(tradp[i-bar:i]):
                        af+=0.01*b if af<=c*0.01 else 0
            else:
                index[i]=index[i-1]
    return index
#################
##############
def make_easy(bar,k1=120,k3=0,k4=0):#转为1分钟bar，取close
    if k3==0:
        k3=len(bar)-k4
    bar=bar[k4:k4+k3]#只有price和volum
    cbar=[];kk=0
    for i in range(int(len(bar)//k1)):
        kk+=1
        cbar+=[[kk,bar[kk*k1-1,1],sum(bar[kk*k1-k1:kk*k1,2])]]
    bar=cbar;bar=np.array(bar)
    return bar
##############
def make_easy_mean(bar,k1=120,k3=0,k4=0):#转为1分钟bar，取平均值
    if k3==0:
        k3=len(bar)-k4
    bar=bar[k4:k4+k3]#只有price和volum
    cbar=[];kk=0
    for i in range(int(len(bar)//k1)):
        kk+=1
        cbar+=[[kk,np.mean(bar[kk*k1-k1:kk*k1,1]),sum(bar[kk*k1-k1:kk*k1,2])]]
    bar=cbar;bar=np.array(bar)
    return bar
########################
########################
class Market():####模拟交易系统，【非常重要！！】
    '''
    bar格式：
    array([[0.0000000e+00, 2.9390000e+03, 1.2090000e+03, ..., 3.2340014e+04,
            1.0000000e+00, 0.0000000e+00],...,
           [9.6978300e+06, 3.2790000e+03, 0.0000000e+00, ..., 5.4000545e+04,
            2.4300000e+02, 1.0000000e+00]])
            注：【序号，价格，本bar成交量，bid价格，ask价格，bid量，ask量，time日内第几秒，交易日.合约代号】
            **夜盘算做下一个交易日的。
    '''
    __bar_time_number=0;
    right=0;pos=0;__right_record=0
    pos_max=0;deal_times=0#最大持仓数，交易比数
    this_bar_order=(0,0);
    def __init__(self,bar,cost,max_pos=999):#最大允许开仓手数
        self.__bar=bar
        self.cost=cost
        self.max_pos=max_pos
        ####
        self.__order_holding=[]#[[(0,1),3]]--->[[order,waiting in queue]]
        self.__last_bar_fufill=[]
        self.__last_bar_error=''
        self.__pos_list=[]
        self.__win_times=0#盈利次数
        self.order_sent=(0,0)
        return
    def call_this_bar(self):#返回本bar盘口信息
        if self.__bar_time_number>=len(self.__bar):
            print('Test_ending!')
            return []
        return self.__bar[self.__bar_time_number]
    def end_this_bar(self):#计算本bar结果，并进入下一帧
        if self.__bar_time_number>=len(self.__bar):
            print('Test_ending!')
            return
        self.__last_bar_error=''#错误信息清零，等待更新
        self.__last_bar_fufill=[]#成交信息清零，等待更新
        self.order_sent=self.this_bar_order
        self.__calculate_this_bar()
        self.__bar_time_number+=1
        self.this_bar_order=(0,0)#本bar的委托信息清零
        return
    def send_order(self,order):#接收客户下单指令，每个bar一次只能下一个order指令！多余的指令以最后一次为准~
        '''order格式：（0,0）什么都不做；（1,p）以价格p，发1手多单；（-1,p）以价格p，发1手空单；
                                    （0，777）撤所有多单，（0，888）撤所有空单，（0，999）撤所有单
                            *【注】目前只能支持委托和持仓不超过两单的情况
        '''
        self.this_bar_order=order
        return
    def get_account_state(self):#返回当前交易信息
        account_state={'right':round(self.right,3),'pos':self.pos,\
                       'order_holding':self.__order_holding.copy(),\
                       'order_sent':self.order_sent,\
                       'deal_times':self.deal_times,\
                       'winning_rate':'{:.3%}'.format(self.__win_times/(self.deal_times+0.0001)*2),\
                       'pos_max':self.pos_max,\
                       'last_bar_fufill':self.__last_bar_fufill.copy(),\
                       'last_bar_error':self.__last_bar_error}
        return account_state
    def __calculate_this_bar(self):#计算客户成交、持仓、收益等情况。
        price=self.__bar[self.__bar_time_number][1]
        vol=self.__bar[self.__bar_time_number][2]
        bid=self.__bar[self.__bar_time_number][3]
        ask=self.__bar[self.__bar_time_number][4]
        bid_size=self.__bar[self.__bar_time_number][5]
        ask_size=self.__bar[self.__bar_time_number][6]
        #####
        if self.__bar_time_number!=0 and \
           self.__bar[self.__bar_time_number][8] !=self.__bar[self.__bar_time_number-1][8]:
            self.__order_holding=[]#新交易日，昨天未成交挂单作废。
        #####
        if self.__order_holding!=[]:#处理委托单存量
            for i in self.__order_holding:
                if i[0][0]>0 and i[0][1]>=ask:#表示买进排队单，被吃单成交
                    self.__last_bar_fufill+=[i[0]]
                    i[1]=0#成交了,并做好删除标记
                elif i[0][0]<0 and i[0][1]<=bid:#表示卖出进排队单，被吃单成交
                    self.__last_bar_fufill+=[i[0]]
                    i[1]=0#成交了,并做好删除标记
                elif i[0][1] in (bid,price,ask):#表示排队单
                    if i[0][1]==price:
                        i[1]-=vol
                    if i[0][1]==bid:
                        i[1]=min(i[1],bid_size)#排队单处理，用最长可能时间法
                    if i[0][1]==ask:
                        i[1]=min(i[1],ask_size)#排队单处理，用最长可能时间法
                    if i[1]<=0:
                        self.__last_bar_fufill+=[i[0]]#成交了
            order_holding_new=[]#要删除记录，并要不影响下一次迭代
            for i in self.__order_holding:
                if i[1]>0:
                    order_holding_new+=[i]
            self.__order_holding=order_holding_new
        #####
        if self.this_bar_order!=(0,0):#处理新委托
            if round(abs(self.this_bar_order[0]))+round(abs(self.pos))+len(self.__order_holding)>self.max_pos:
                self.__last_bar_error=str(self.__bar_time_number)+'：发单超过持仓限制，作废！'
            elif self.this_bar_order[0]==0:
                if self.this_bar_order[1]==999:#删除所有订单
                    self.__order_holding=[]
                order_holding_new=[]#要删除记录，准备好记录
                if self.this_bar_order[1]==777:#删除做多订单
                    for i in self.__order_holding:
                        if i[0][0]<0:
                            order_holding_new+=[i]
                elif self.this_bar_order[1]==888:#删除做空订单
                    for i in self.__order_holding:
                        if i[0][0]>0:
                            order_holding_new+=[i]
                self.__order_holding=order_holding_new
            elif self.this_bar_order[0]>0 and self.this_bar_order[1]>=ask:#表示对手价买进，成交
                self.__last_bar_fufill+=[self.this_bar_order]
            elif self.this_bar_order[0]<0 and self.this_bar_order[1]<=bid:#表示对手价卖出，成交
                self.__last_bar_fufill+=[self.this_bar_order]
            else:
                if self.this_bar_order[0]>0 and self.this_bar_order[1]==bid:
                    hp=bid_size
                elif self.this_bar_order[0]<0 and self.this_bar_order[1]==ask:
                    hp=ask_size
                else:
                    hp=9999
                self.__order_holding+=[[self.this_bar_order, hp ]]#转入排队队列[[(0,1),3]]--->[[order,waiting in queue]]
            self.this_bar_order=(0,0)
        if self.__last_bar_fufill!=[]:
            for i in self.__last_bar_fufill:
                self.__pos_list+=[-i[1]*abs(i[0])/i[0]]*abs(i[0])
                #print(self.__pos_list);print(self.pos);print(self.get_account_state())
                while self.__pos_list!=[] and self.__pos_list[0]*self.__pos_list[-1]<0:
                    if self.__pos_list[0]+self.__pos_list[-1]-2*self.cost>0:
                        self.__win_times+=1
                    self.__pos_list=self.__pos_list[1:-1]
                self.pos+=i[0]
                self.pos_max=max(abs(self.pos),self.pos_max)
                self.deal_times+=abs(i[0])
                self.__right_record+=-i[0]*i[1]-abs(i[0])*self.cost
        self.right=self.__right_record+self.pos*price
        return
####################
####################
def dt_rate(right_list,risk_free_interest_perbar,right_list_len=-1): #下行偏差率__简化版；
    #【注意】：由于涉及无风险利率的计算，right_list要包含不开仓的0点，以统计bar时长。
    a=0;pos_times=0
    if right_list_len==-1:
        right_list_len=len(right_list)
    sum_right=sum(right_list)
    min_time=0
    for i in right_list:
        if i<0:
            a+=i**2;min_time+=1
        if i!=0:
            pos_times+=1
    right_relative=sum_right-right_list_len*risk_free_interest_perbar
    if a==0:
        if pos_times<=6 and right_relative>0:
            a=-0.0001
        else:
            a=0.0001
    else:
        a=(a/min_time)**0.5
    return right_relative/a
###################
def rank(b):#前百分之多少小
    a=b.copy()
    tar=a[-1]
    a.sort()
    c=0
    for i in a:
        c+=1
        if i==tar:
            break
    return c/len(a)
def Lin_Slope(y):#最小二乘法求线性回归,得到的是系数a，b
    x=range(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c
#####
def Lin_list(y,k=1):#最小二乘法求线性回归,分k段
    if k==1:
        x=range(len(y))
        m, c =Lin_Slope(y)
        return m*x + c
    elif k>1:
        c=len(y)//k
        x=[]
        for i in range(k):
            xs=Lin_list(y[c*i:c*(i+1)])
            x+=xs
        return x
#####
def Lin_std(y):
    a=Lin_list(y)
    p=[]
    for i in range(len(a)):
        p+=[a[i]-y[i]]
    return np.std(p)
#####
def plot_Lin_num(y,i,s=0,k=1):#画图，价格，线性拟合
	d=y[s:i]
	a=Lin_Slope_list(y[s:i],k=k)
	plot([d,a],k=1)
#####
def price_time(a):#价格频次统计
    p={}
    for i in a:
        if i in p:
            p[i]+=1
        else:
            p[i]=0
    s=[]
    for i in sorted(p,key=p.__getitem__,reverse=True):
        s+=[[i,p[i]]]
    return s
######
def rank_real(a):#价格排名百分比
    if len(a)==0:
        return 0.5
    p=a[-1]
    b=sorted(a)
    c1=0;c2=0
    for i in range(len(b)):#防止重复数字出现，做出保守处理
        if b[i]<=p:
            c1+=1
        if b[i]>=p:
            c2+=1
    if c1<c2:
        return c1/len(b)
    elif c2<c1:
        return 1-c2/len(b)
    else:
        return 0.5
######ATR指数
def ATR(a,lon,ln2=0):#ln2:TR个数
    p=[]
    TR=[]
    for i in range(1,len(a)):
        if i%lon==0:
            p+=[[max(a[i-lon:i]),min(a[i-lon:i]),a[i-1]]]
    for i in range(1,len(p)):
        z,x,c=p[i]
        v,b,n=p[i-1]
        TR+=[max(z-x,abs(z-n),abs(x-n))]
    if ln2!=0 and len(TR)-ln2>=0:
        TR=TR[len(TR)-ln2:]
    return sum(TR)/len(TR)
def AR1(aa0,c=0):#均值自回归参数 aa0:list
    m=np.mean(aa0)
    if c==0:
        c=m*0.001
    t=[]
    for i in aa0:
        t+=[i-m]
    '''x1=0;x2=0
    for k in range(len(t)-1):
        x1+=t[k]*t[k+1]
        if k!=0:
              x2+=t[k]**2
    b=x1/x2'''
    t=pd.Series(list(t))
    m = sm.tsa.ar_model.AR(t)
    res =m.fit(1,mothed='mle',trend='nc')
    b=list(res.params)[0]
    if b>=1:
        b=0.999
    alpha=-math.log(b)
    ita=((2*alpha*np.var(t))/(1-b*b))**(0.5)
    kaba=((c*alpha)**3+24*c*(alpha*ita)**2-4*(3*c**4*alpha**5*ita**2+36*c**2*alpha**4*ita**4)**(0.5))**(1/3)
    a=abs(-c/4-(c*c*alpha)/(4*kaba)-kaba/(4*alpha))
    return a,b
########        
def envelope(bar,k=5):
    def arith(a,b,m):#生成a到b有m个的等差数列
        lis=[]
        for i in range(m):
            lis+=[a+(b-a)*(i+1)/m]
        return lis
    maxp=bar[0];    minp=bar[0]
    rec_max=[bar[0]];    rec_min=[bar[0]]
    new=0
    c=0;    c2=0
    findp=[bar[0]]
    for i in bar:
        c+=1
        if i>=max(minp+k,maxp):
            if new==-1:
                rec_max+=arith(rec_max[-1],maxp,c2-len(rec_max))
                rec_min+=arith(rec_min[-1],minp,c2-len(rec_min))
            if (i>=maxp+1 and new!=-1) or new==-1:
                maxp=i; minp=maxp-5; new=1; c2=c
            findp+=[maxp]
        elif i<=min(maxp-k,minp):
            if new==1:
                rec_max+=arith(rec_max[-1],maxp,c2-len(rec_max))
                rec_min+=arith(rec_min[-1],minp,c2-len(rec_min))
            if (i<=minp-1 and new!=1) or new==1:
                minp=i; maxp=minp+5; new=-1; c2=c
            findp+=[minp]
        elif c>1:
            findp+=[findp[-1]]
    rec_max+=arith(rec_max[-1],maxp,c2-len(rec_max))
    rec_min+=arith(rec_min[-1],minp,c2-len(rec_min))
    return [bar,findp,rec_max,rec_min]
#######
def FDD_d(bar,d=0.35):#求序列的分数阶导数
    x_=[]
    w=1;k=1;a=[1]
    while abs(w)>1e-5:
        w*=-(d-k+1)/k
        a+=[w]
        k+=1
    a=np.array(a)[::-1] 
    for i in range(1,len(bar)+1):
        if i<k:
            x_+=[0]
        else:
            x_+=[sum(a*bar[i-k:i])]
    return x_
####
def process_list(lst, func,k=0):
    """
    递归遍历嵌套列表，对其中的元素进行函数运算并重新赋值
    """
    if k==0:
        lst=copy.deepcopy(lst)
    for i, item in enumerate(lst):
        if isinstance(item, list):
            process_list(item, func,k=1)  # 递归处理嵌套列表
        else:
            lst[i] = func(item)  # 对元素进行函数运算并重新赋值
    return lst
def deep(bar,func):#序列深度运算
    type_str=str(type(bar))
    if str(type(bar)) in ["<class 'float'>","<class 'int'>","<class 'numpy.float64'>"]:
        return func(bar)
    if str(type(bar))=="<class 'numpy.ndarray'>":
        bar=bar.tolist()
    if str(type(bar))=="<class 'list'>":
        x=process_list(bar, func)
        if type_str=="<class 'list'>":
            return x
        if type_str=="<class 'numpy.ndarray'>":
            return np.array(x)
def sig(x):
    '求序列的符号序列，支持各种格式'
    return deep(x,lambda i: int(abs(i)/i) if i!=0 else 0)
def sig_log(x):
    '求序列的符号对数，支持各种格式'
    return deep(x,lambda i: sig(i)*np.log(abs(i)) if abs(i)>=1 else 0)
####
def cal(bar,weit=[1],de=1):#求积分序列
    'weit为序列权重,de为指数衰减值'
    if weit==[1]:
        weit=len(bar)*[1]
    a=[0]
    for i,k in zip(*(bar,weit)):
        a+=[a[-1]*de+i*k]
    a=a[1:]
    return a
####
def FFT(x,plot=1):
    '傅里叶变换'
    N=len(x)# 时间序列的长度为N
    # 对x进行傅里叶变换
    X = np.fft.fft(x)
    # 计算频率轴
    freq = np.fft.fftfreq(N, d=1)
    # 取前一半的数据（由于FFT结果是对称的，因此只需要取前一半）
    X_half = X[:N//2]
    freq_half = freq[:N//2]
    # 绘制原始信号和FFT结果
    if plot==1:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(range(len(x)), x)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax2.plot(freq_half, np.abs(X_half))
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Magnitude')
        plt.show()
    return np.abs(X_half)
###
def get_right(close,pos,cost=0):
    '计算盈利序列'
    if not(type(close)==type(pos)==type(np.array([]))):
        print('【错误】：参数需要统一np.array格式。')
        return
    if len(close)!=len(pos):
        print('【错误】：参数close,pos_list长度不相等。')
        return
    if cost=='auto':
        cost=close*0.0001
    right_list=np.append(pos[:-1]*np.diff(close),0)
    cost_list=abs(np.insert(np.diff(pos),0,pos[0]))*cost
    right_list=right_list-cost_list
    return cal(right_list)
def get_right_diff(close_diff,pos,cost=0,price_mean=0):
    '计算盈利序列'
    if not(type(close_diff)==type(pos)==type(np.array([]))):
        print('【错误】：参数需要统一np.array格式。')
        return
    if len(close_diff)!=len(pos):
        print('【错误】：参数close,pos_list长度不相等。')
        return
    if cost=='auto':
        cost=price_mean*0.0001
    right_list=pos*close_diff
    cost_list=abs(np.insert(np.diff(pos),0,pos[0]))*cost
    right_list=right_list-cost_list
    return cal(right_list)
def normal_2D(arr):
    w=[];std=[]
    for i in range(arr.shape[1]):
        w+=[np.mean(arr[:,i])];std+=[np.std(arr[:,i])]
    return (arr-np.array(w).reshape(1, -1))/np.array(std).reshape(1, -1)
####
def plot_tree(rfc):
    from sklearn.tree import plot_tree
    trees=rfc.estimators_
    tree=trees[0]
    fig, ax = plt.subplots(figsize=(18, 18))
    plot_tree(tree, ax=ax, feature_names=np.array(range(1,14)), filled=True
              ,class_names=['-1,','1'])
    plt.show()
def Return_drawdown_ratio(y):
    '输出：收益最大回撤比，最大回撤；'
    h=0
    i=y[0]
    for k in y:
        if k>i:
            i=k
        elif i-k>h:
            h=i-k
    drawdown=h#以下处理h为0的特殊情况，故此记录回撤
    if h==0 and len(np.diff(y)[np.diff(y)!=0])!=0:
        h=np.min(np.abs(np.diff(y)[np.diff(y)!=0]) )*0.05#防止h为0，令其为最小差值的5%
    #若h仍然为0，那么说明是常数列，其增长为0，因此此时输出0
    return (y[-1]-y[0])/h if h!=0 else 0,drawdown
def Sharp_ratio(y):
    '输出：夏普率'
    h=np.std(y)
    #波动率为0，只能是常数列，其增长为0，因此此时输出0
    return (y[-1]-y[0])/h if h!=0 else 0


####
if __name__=='__main__':
    y=np.random.randn(30)
    a=np.array([1,1,1,1,1])
    b=np.array([1,2,3,4,5])
    c=np.array([1,2,5,5,6])
