'''F1绘图函数'''
import numpy as np
import opfunu
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

M_1_D2 = [9.0583875284671056e-001 ,-4.2362265501400698e-001]
shift_data_1 = [ 5.0355789822908633e+001 , 6.4926709932099072e+001]
def Elliptic(X,dim):
    Results = 0
    for i in range(dim):
        Results = Results + (10 ** 6) ** (i / dim - 1) * X[i] ** 2
    return Results

def cec2014F1(X):
    dim = X.shape[0]
    f_bais = 100.
    Z = np.dot(M_1_D2 , X - shift_data_1)
    Results = Elliptic(X,dim) + f_bais


    return Results

def F1_2014(X):
    dim = X.shape[0]
    F1 = opfunu.cec_based.cec2014.F12014(ndim = dim)
    F1_2014=F1.evaluate(X)
    # print("解",F1_2014)
    return F1_2014

def F1Plot():
    fig = plt.figure(1) #定义figure
    ax = Axes3D(fig) #将figure变为3d
    x1=np.arange(-100,100,5) #定义x1，范围为[-100,100],间隔为2
    x2=np.arange(-100,100,5) #定义x2，范围为[-100,100],间隔为2
    X1,X2=np.meshgrid(x1,x2) #生成网格
    nSize = x1.shape[0]
    Z=np.zeros([nSize,nSize])
    for i in range(nSize):
        for j in range(nSize):
            X=[X1[i,j],X2[i,j]] #构造F1输入
            X=np.array(X) #将格式由list转换为array
            Z[i,j]= F1_2014(X)  #计算F1的值
    #绘制3D曲面
    # rstride:行之间的跨度  cstride:列之间的跨度
    # rstride:行之间的跨度  cstride:列之间的跨度
    # cmap参数可以控制三维曲面的颜色组合
    ax.plot_surface(X1, X2, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))
    ax.contour(X1, X2, Z, zdir='z', offset=0)#绘制等高线
    ax.set_xlabel('X1')#x轴说明
    ax.set_ylabel('X2')#y轴说明
    ax.set_zlabel('Z')#z轴说明
    ax.set_title('F1_space')
    plt.show()


F1Plot()


















