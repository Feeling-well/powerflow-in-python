# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:33:58 2017
lastest edit：2017-12-1
功能：极坐标潮流程序
@author: 苏
"""

import numpy as np
from scipy import sparse
import datetime
import matplotlib.pyplot as plt 
import seaborn as sns
#import math
#import scipy.io as sio
#import pprint
#np.set_printoptions(threshold=np.inf)     #输出全部矩阵数据。
#np.set_printoptions(threshold = 1e6)      #输出精度与全部数据控制语句
#from numpy import *
#import pandas as pd
#from gurobipy import *
#import time

#%% 符合说明
# node_number：节点数量;    Jacobi：雅可比矩阵;       blanNode：平衡节点号;    
# data：系统参数;          lineblock：线路参数;      branchblock：接地支路参数;
# transblock：变压器参数;   pvNodepara：pv节点参数
# y：节点导纳矩阵           pvNode：PV节点号;         pvNum：PV节点数量
# pis：节点注入有功;        qis：节点注入无功   
# iteration：迭代次数;      accuracy：迭代精度;      jie：修正量
# deltP：有功不平衡量       deltQ：无功不平衡量       delta：相角修正量    deltv：电压修正量
# va:相角                   v0：幅值
#%% 数据读取
data=[]
for l in open('D:\\pythonfiles\\1047.txt','r'):
    row = [float(x) for x in l.split()]
    row+=[0]*(8-len(row))
    data.append(row)
data_np=np.array(data)
start = datetime.datetime.now()     #开始计时
node_number = int(data_np[0][0])    #节点数量
data[2][1]                          #平衡节点号
myreturn_index = []                                
for i in np.arange(len(data)):
    if data[i][0] == 0:
        myreturn_index.append(i)
myreturn = np.array(myreturn_index)
lineN=myreturn[1]-myreturn[0]-1     #线路参数的行数
knum=myreturn[1]-1                  #线路参数结束行数
#读取线路参数
lineblock=data_np[myreturn[0]+1:knum+1]   #线路参数整体切片
lineNo=lineblock[0:lineN,0]
linei=lineblock[0:lineN,1]          #线路参数的母线i
linej=lineblock[0:lineN,2]          #线路参数的母线j
liner=lineblock[0:lineN,3]          #线路参数的R
linex=lineblock[0:lineN,4]          #线路参数的X
lineb=lineblock[0:lineN,5]          #线路参数的B
#接地支路参数读取
branch=myreturn[2]-myreturn[1]-1    #接地支路共有行数
k1=knum+2                           #接地支路开始行
k2=knum+1+branch                    #接地支路结束行
branchblock=data_np[k1:k2+1,:]      #接地支路参数整体切片
branchi=branchblock[0:branch,0]     #接地支路节点号
branchb=branchblock[0:branch,1]     #接地支路导纳
branchg=branchblock[0:branch,2]
#变压器参数读取
trans=myreturn[3]-myreturn[2]-1     #变压器参数共有行数
k1=k2+2                             #变压器参数开始行
k2=myreturn[2]+trans                #变压器参数结束行
transblock=data_np[k1:k2+1,:]       #变压器参数整块切片
transi=transblock[0:trans,1]        #变压器参数的母线i
transj=transblock[0:trans,2]        #变压器参数的母线j
transr=transblock[0:trans,3]        #变压器参数的R
transx=transblock[0:trans,4]        #变压器参数的X
transk=transblock[0:trans,5]        #变压器参数的变比
#节点功率参数读取
pow=myreturn[4]-myreturn[3]-1       #节点功率共有行数
k1=k2+2                             #节点功率开始行
k2=k2+1+pow  
powblock=data_np[k1:k2+1,:]        #节点功率参数整块切片 
powi=powblock[0:pow,0]             #节点功率参数的节点号
powpgi=powblock[0:pow,1]           #节点功率参数的PG
powqgj=powblock[0:pow,2]           #节点功率参数的QG
powpdi=powblock[0:pow,3]           #节点功率参数的PD
powqdj=powblock[0:pow,4]           #节点功率参数的QD
pv=myreturn[5]-myreturn[4]-1       #PV节点共有行数
k1=k2+2                            #PV节点开始行
k2=k2+1+pv;                        #PV节点结束行
#读取pv节点参数
pvblock=data_np[k1:k2+1,:]
pvi=pvblock[0:pv,0]                #PV节点参数的节点号
pvv=pvblock[0:pv,1]                #PV节点参数的电压
pvqmin=pvblock[0:pv,2]             #PV节点参数的Qmin
pvqmax=pvblock[0:pv,3]             #PV节点参数的Qmax
#%% 数据读取完毕，导纳矩阵的形成
#符合说明：linei为行，linej为列
#线路导纳矩阵
z1=1.*(liner+1j*linex)**-1                                       #矩阵的除法
z11=1.*(liner+1j*linex)**-1+1j*lineb
y1_1=-sparse.coo_matrix((z1,(linei-1,linej-1)),shape=(node_number,node_number))
y1_2=-sparse.coo_matrix((z1,(linej-1,linei-1)),shape=(node_number,node_number))
y1_3=sparse.coo_matrix((z11,(linei-1,linei-1)),shape=(node_number,node_number))
y1_4=sparse.coo_matrix((z11,(linej-1,linej-1)),shape=(node_number,node_number))    
y1=y1_1+y1_2+y1_3+y1_4                                           #线路导纳矩阵
#变压器导纳矩阵
z2=1*(transr+1j*transx)**-1*(transk)**-1                         #  含义为1./(transr+j*transx)./transk
z22=(1-transk)*(transr+1j*transx)**-1*(transk)**-1*(transk)**-1+z2
z23=(transk-1)*(transr+1j*transx)**-1*(transk)**-1+z2
y2_1=-sparse.coo_matrix((z2,(transi-1,transj-1)),shape=(node_number,node_number))
y2_2=-sparse.coo_matrix((z2,(transj-1,transi-1)),shape=(node_number,node_number)) 
y2_3=sparse.coo_matrix((z22,(transi-1,transi-1)),shape=(node_number,node_number))
y2_4=sparse.coo_matrix((z23,(transj-1,transj-1)),shape=(node_number,node_number))
y2=y2_1+y2_2+y2_3+y2_4                                           #变压器导纳矩阵
#接地支路导纳矩阵
y3=sparse.coo_matrix((branchg+1j*branchb,(branchi-1,branchi-1)),shape=(node_number,node_number))
y=y1+y2+y3                                                        #节点导纳矩阵        
y_abs=abs(y)
pis_1=(powpgi-powpdi)/100                                         #基准值处理
pis_2=(powqgj-powqdj)/100                                         #基准值处理
powi0=powi*0
powi0[0:node_number-1]=0 
pis=sparse.coo_matrix((pis_1,(powi-1,powi0)))/100                #pis与qis的求解结果正确
qis=sparse.coo_matrix((pis_2,(powi-1,powi0)))/100                #
powi0=np.transpose(powi0)
v0=(node_number,1)                                               #单位矩阵
v0=np.ones(v0)                                                   #初始化电压值
va=powi0*0                                                       #初始化电压相角
v0[int(data[2][1])-1]=1                                          #电压初始化
n=0
#为pv节点电压赋值
for i in range(0,len(pvi)):
    v0[int(pvi[i])-1]=pvv[n]
    n=n+1
accuracy=1  #精度
iteration=1     #迭代次数
#%% 开始迭代=================================
while (accuracy>data[1][0] and iteration<20):
#%% 不平衡量    
    a=np.angle(y.toarray())
    va0=(node_number,1)
    va0=np.zeros(va0)                  #初始化电压相角
    va=va0
    ij=va-va.T-a
    v00=np.cos(va)+1j*np.sin(va)
    v=v0*v00
    delt0=np.conj(y*v)
    delt1=v*delt0                       #代入节点电压求出的功率
    deltp=pis-np.real(delt1)            #有功修正量
    deltq=qis-np.imag(delt1)            #无功修正量
    deltp[int(data[2][1]-1)]=0          #有功修正量(处理平衡节点)
    deltq[int(data[2][1]-1)]=0          #无功修正量(处理平衡节点)
    pvi_int=pvi.astype(int)             #将pvi变为整型
    deltq[[pvi_int-1]]=0                # 无功修正量（处理pv节点）
    delt=np.vstack((deltp,deltq))       #将deltp与deltq合并在一起，横向拼接
    #np.diag(x.flat)与matlab中diag（x）相同
    #np.dots满阵的情况下表示矩阵的乘法
#%% 雅可比矩阵的形成
    h1_1=np.dot(np.diag(v0.flat),y_abs.toarray()*np.sin(ij))
    h1_2=np.dot(np.diag(v0.flat),y_abs.toarray()*np.sin(ij))
    hh=np.diag(np.dot(h1_1,v0).flat)-np.dot(h1_1,np.diag(v0.flat))    # H矩阵形成
    nn_1=np.dot(np.diag(v0.flat),y_abs.toarray()*np.cos(ij))  
    nn_2=np.dot(y_abs.toarray()*np.cos(ij),v0)
    nn=-nn_1-np.diag(nn_2.flat)                                       # N矩阵形成
    jj_1=np.dot(np.diag(v0.flat),y_abs.toarray()*np.cos(ij)) 
    jj_2=np.dot(np.diag(v0.flat),y_abs.toarray()*np.cos(ij))
    jj=-np.diag(np.dot(jj_1,v0).flat)+np.dot(jj_1,np.diag(v0.flat))   # J矩阵形成
    ll_1=np.dot(np.diag(v0.flat),y_abs.toarray()*np.sin(ij))
    ll_2=np.dot(y_abs.toarray()*np.sin(ij),v0)
    ll=-ll_1-np.diag(ll_2.flat)                                       # L矩阵初步形成
    #对四子矩阵处理
    nn[:,[pvi_int-1]]=0
    jj[[pvi_int-1],:]=0
    ll[[pvi_int-1],:]=0
    ll[:,[pvi_int-1]]=0
    data1=(len(pvi_int),)                                                    #单位矩阵
    data1=np.ones(data1)
    ll=ll+sparse.coo_matrix((data1,((pvi-1).T,(pvi-1).T)),shape=(node_number,node_number))    # L矩阵初步形成
    Jacobi1=np.hstack((hh,nn))                                                   #纵向拼接
    Jacobi2=np.hstack((jj,ll))                                                   #横向拼接
    Jacobi=np.vstack((Jacobi1,Jacobi2))                                          #初步形成雅可比
    Jacobi[[int(data[2][1])-1],:]=0                                              #处理平衡节点
    Jacobi[:,[int(data[2][1])-1]]=0                                              #处理平衡节点
    Jacobi[[int(data[2][1])-1],[int(data[2][1])-1]]=1                            #处理平衡节点
    Jacobi[[int(data[2][1])+node_number-1],:]=0                                  #处理平衡节点
    Jacobi[:,[int(data[2][1])+node_number-1]]=0                                  #处理平衡节点
    Jacobi[[int(data[2][1])+node_number-1],[int(data[2][1])+node_number-1]]=1    #处理平衡节点
#%% 修正量    
    jie=np.linalg.solve(Jacobi,delt)
    delta=jie[0:node_number,]
    deltv=jie[(node_number):2*node_number]
    va=va-delta
    v0=np.array(v0-deltv)
    accuracy=np.max(np.abs(deltv))
    iteration=iteration+1
#循环结束================================================
#%% 结果输出
print('电压幅值：\n',v0)
x_values = list(range(1, node_number+1))
plt.figure(2)
ax1=plt.subplot(2,2,1)#在图表2中创建子图1  
ax1=plt.scatter(x_values,va,50,c='red',marker='x',alpha=1)
plt.title('Vangle')
plt.xlabel('variables x')
plt.ylabel('variables y')
plt.legend(loc='upper right')   #显示图例

ax2=plt.subplot(2,2,2)#在图表2中创建子图2  
ax2=plt.scatter(x_values,v0,50,c='blue',marker='o',alpha=1)
plt.title('V0')
plt.xlabel('variables x')
plt.ylabel('variables y')
plt.legend(loc='upper right')   #显示图例

print('电压相角：\n',va)
end = datetime.datetime.now()
print ('迭代次数:',iteration)

