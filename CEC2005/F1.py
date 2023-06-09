import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import opfunu

print("====================F1")
problem = opfunu.cec_based.F12005(ndim=2)
x = [-39.3119 , 58.8999]
print(x)
print(problem.evaluate(x))
print(problem.x_global)

dim = 30 
F1 = opfunu.cec_based.cec2005.F12005(ndim = dim)

def F12005(X):
    result = opfunu.cec_based.F12022(ndim=2)
    return result



def F12005(X):
    '''F1_2005绘图函数'''
    f_bias_1 = -450
    # 全局
    O = 0
    Z = X
    # print(Z)
    Fitness = np.sum(Z**2) 
    return Fitness 

def F1Plot():
    fig = plt.figure(1) #定义figure
    ax = Axes3D(fig) #将figure变为3d
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
            #将格式由list转换为array
            X=np.array(X) 
            # 计算F1的值
            Z[i, j] = F12005(X)  

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