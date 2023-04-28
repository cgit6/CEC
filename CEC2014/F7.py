'''F7绘图函数'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import opfunu

def F7_2014(X):
    dim = X.shape[0]
    F7 = opfunu.cec_based.cec2014.F72014(ndim = dim)
    F7_2014=F7.evaluate(X)
    # print("解",F7_2014)
    return F7_2014


def F7Plot():
    fig = plt.figure(1) #定义figure
    ax = Axes3D(fig) #将figure变为3d
    x1=np.arange(10,100,2) #定义x1，范围为[-100,100],间隔为2
    x2=np.arange(-40,50,2) #定义x2，范围为[-100,100],间隔为2
    X1,X2=np.meshgrid(x1,x2) #生成网格
    nSize = x1.shape[0]
    Z=np.zeros([nSize,nSize])
    for i in range(nSize):
        for j in range(nSize):
            X=[X1[i,j],X2[i,j]] #构造F3输入
            X=np.array(X) #将格式由list转换为array
            Z[i,j]= F7_2014(X)  #计算F3的值
    #绘制3D曲面
    # rstride:行之间的跨度  cstride:列之间的跨度
    # rstride:行之间的跨度  cstride:列之间的跨度
    # cmap参数可以控制三维曲面的颜色组合
    ax.plot_surface(X1, X2, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))
    ax.contour(X1, X2, Z, zdir='z', offset=0)#绘制等高线
    ax.set_xlabel('X1')#x轴说明
    ax.set_ylabel('X2')#y轴说明
    ax.set_zlabel('Z')#z轴说明
    ax.set_title('F3_space')
    plt.show()

F7Plot()