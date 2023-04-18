import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def F32005 (X):
    '''F3_2005绘图函数'''
    dim = X.shape[0]
    f_bias_3 = -450

    # np.dot : np.dot()函数主要有两个功能，向量点积和矩阵乘 
    result = 0
    for i in range(dim):
        Z = X
        # print(Z)
        result =  result + ((1000000)**(i/dim-1)) * Z[i]**2
        # print( result)
    return  result + f_bias_3

def F3Plot():
    fig = plt.figure(1) #定义figure
    ax = Axes3D(fig) #将figure变为3d
    # 遍歷解空間
    x1=np.arange(-100,100,2) #定义x1，范围为[-100,100],间隔为2
    x2=np.arange(-100,100,2) #定义x2，范围为[-100,100],间隔为2
    X1,X2=np.meshgrid(x1,x2) #生成网格

    nSize = x1.shape[0]
    # 初始化空間
    Z=np.zeros([nSize,nSize])

    for i in range(nSize):
        for j in range(nSize):
            #构造F1输入
            X=[X1[i,j],X2[i,j]] 
            #将格式由list转换为np.array
            X=np.array(X) 
            # 计算F1的值
            Z[i, j] = F32005(X) 
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

F3Plot()