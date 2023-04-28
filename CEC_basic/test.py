"""
依照Slime mould algorithm A new method for stochastic optimization 實作該測試函數集
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#===============================================#
#                     F1                        #
#===============================================#
f1_dim = 30
f1_lb = -100 * np.ones(f1_dim)
f1_ub = 100 * np.ones(f1_dim)
def F1(X):
    Results= np.sum(X**2)
    return Results

#===============================================#
#                     F2                        #
#===============================================#
f2_dim = 30
f2_lb = -10 * np.ones(f2_dim)
f2_ub = 10 * np.ones(f2_dim)
def F2(X):
    Results=np.sum(np.abs(X))+np.prod(np.abs(X))
    return Results


f3_dim = 30
f3_lb = -100 * np.ones(f3_dim)
f3_ub = 100 * np.ones(f3_dim)

#===============================================#
#                     F3                        #
#===============================================#
def F3(X):
    dim = X.shape[0]
    Results = 0
    for i in range(dim):
        Results = Results+np.sum(X[0:i+1])**2

    return Results
#===============================================#
#                     F4                        #
#===============================================#


def F4(X):
    Results = np.max(np.abs(X))

    return Results
#===============================================#
#                     F5                        #
#===============================================#
f5_dim = 30
f5_lb = -30 * np.ones(f5_dim)
f5_ub = 30 * np.ones(f5_dim)


def F5(X):
    dim = X.shape[0]
    Results = np.sum(100*(X[1:dim]-(X[0:dim-1]**2))**2+(X[0:dim-1]-1)**2)
    return Results
#===============================================#
#                     F6                        #
#===============================================#
f6_dim = 30
f6_lb = -30 * np.ones(f6_dim)
f6_ub = 30 * np.ones(f6_dim)
def F6(X):
    Results=np.sum(np.abs(X+0.5)**2)
    return Results
#===============================================#
#                     F7                        #
#===============================================#
f7_dim = 30
f7_lb = -1.28 * np.ones(f7_dim)
f7_ub = 1.28 * np.ones(f7_dim)
def F7(X):
    dim = X.shape[0]
    Temp = np.arange(1,dim+1,1)
    Results=np.sum(Temp*(X**4))+np.random.random()
    return Results
#===============================================#
#                     F8                        #
#===============================================#
f8_dim = 30
f8_lb = -500 * np.ones(f8_dim)
f8_ub = 500 * np.ones(f8_dim)
def F8(X):
    Results=np.sum(-X*np.sin(np.sqrt(np.abs(X))))
    return Results
#===============================================#
#                     F9                        #
#===============================================#
f9_dim = 30
f9_lb = -5.12 * np.ones(f9_dim)
f9_ub = 5.12 * np.ones(f9_dim)
def F9(X):
    dim=X.shape[0]
    Results=np.sum(X**2-10*np.cos(2*np.pi*X))+10*dim
    return Results


#===============================================#
#                     F10                       #
#===============================================#
f10_dim = 30
f10_lb = -32 * np.ones(f10_dim)
f10_ub = 32 * np.ones(f10_dim)


def F10(X):
    dim = X.shape[0]
    Results = -20*np.exp(-0.2*np.sqrt(np.sum(X**2)/dim)) - \
        np.exp(np.sum(np.cos(2*np.pi*X))/dim)+20+np.exp(1)
    return Results


#===============================================#
#                     F11                       #
#===============================================#
f11_dim = 30
f11_lb = -600 * np.ones(f11_dim)
f11_ub = 600 * np.ones(f11_dim)


def F11(X):
    dim = X.shape[0]
    # print("dim",dim)
    Temp = np.arange(1, dim+1, 1)
    # print("Temp",Temp)
    # print(X)
    Results = np.sum(X**2)/4000-np.prod(np.cos(X/np.sqrt(Temp)))+1
    # print("result",Results)
    return Results


#===============================================#
#                     F12                       #
#===============================================#
f12_dim = 3
f12_lb = -50 * np.ones(f12_dim)
f12_ub = 50 * np.ones(f12_dim)

def F12_Ufun(X, a, k, m, dim):
    Results = []
    for i in range(dim):
        if (X[i]):
            Results.append(k*(X[i] - a)**m)
        elif (-a < X[i] < a):
            Results.append(0)
        elif (X[i] < a):
            Results.append(k*(-X[i] - a)**m)
    # print(f"F12_Ufun {Results}")
    return Results


def F12(X):
    dim = X.shape[0]
    y = 1+(X + 1)/4
    part_1 = (np.pi/dim) * (10*np.sin(np.pi*y[0]))
    part_2 = np.sum(((y[0:dim-2]-1)**2) *
                    (1+10*np.sin(np.pi*X[1:dim])**2 + (y[dim-1]-1)**2))
    part_3 = np.sum(F12_Ufun(X, 10, 100, 4, dim))
    Results = part_1 + part_2 + part_3
    # print(f"F12 {part_1},{part_2},{part_3},{Results}")

    return Results


#===============================================#
#                     F13                       #
#===============================================#
f13_dim = 30
f13_lb = -50 * np.ones(f13_dim)
f13_ub = 50 * np.ones(f13_dim)

def F13_Ufun(X, a, k, m, dim):
    Results = []
    for i in range(dim):
        if (X[i]):
            Results.append(k*(X[i] - a)**m)
        elif (-a < X[i] < a):
            Results.append(0)
        elif (X[i] < a):
            Results.append(k*(-X[i] - a)**m)
    # print(f"F13_Ufun {Results}")
    return Results


def F13(X):
    dim = X.shape[0]
    Results = 0.1*((np.sin(3*np.pi*X[0]))**2+np.sum((X[0:dim-1]-1)**2*(1+(np.sin(3*np.pi*X[1:dim]))**2)) +
                   ((X[dim-1]-1)**2)*(1+(np.sin(2*np.pi*X[dim-1]))**2))+np.sum(F13_Ufun(X, 5, 100, 4, dim))
    # print(Results)
    return Results
#===============================================#
#                     CEC2014F1                 #
#===============================================#
cec2014f1_dim = 30
cec2014f1_lb = -100 * np.ones(cec2014f1_dim)
cec2014f1_ub = 100 * np.ones(cec2014f1_dim)

def Elliptic(X,dim):
    Results = 0
    for i in range(dim):
        Results = Results + (10 ** 6) ** (i / dim - 1) * X[i] 
    return Results

def cec2014F1(X):
    dim = X.shape[0]
    Results = Elliptic(X,dim)


    return
#===============================================#
#                     CEC2014F2                 #
#===============================================#
cec2014f2_dim = 30
cec2014f2_lb = -50 * np.ones(cec2014f2_dim)
cec2014f2_ub = 50 * np.ones(cec2014f2_dim)
def cec2014F2(X):

    pass
#===============================================#
#                     CEC2014F5                 #
#===============================================#
cec2014f5_dim = 30
cec2014f5_lb = -50 * np.ones(cec2014f5_dim)
cec2014f5_ub = 50 * np.ones(cec2014f5_dim)
def cec2014F5(X):

    pass
#===============================================#
#                     CEC2014F6                #
#===============================================#
cec2014f6_dim = 30
cec2014f6_lb = -50 * np.ones(cec2014f6_dim)
cec2014f6_ub = 50 * np.ones(cec2014f6_dim)
def cec2014F6(X):

    pass


#===============================================#
#                     CEC2014F13                #
#===============================================#
cec2014f13_dim = 30
cec2014f13_lb = -50 * np.ones(cec2014f13_dim)
cec2014f13_ub = 50 * np.ones(cec2014f13_dim)
def cec2014F13(X):

    pass
#===============================================#
#                     CEC2014F14                #
#===============================================#
cec2014f14_dim = 30
cec2014f14_lb = -50 * np.ones(cec2014f14_dim)
cec2014f14_ub = 50 * np.ones(cec2014f14_dim)
def cec2014F14(X):

    pass
#===============================================#
#                     CEC2014F15                #
#===============================================#
cec2014f15_dim = 30
cec2014f15_lb = -50 * np.ones(cec2014f15_dim)
cec2014f15_ub = 50 * np.ones(cec2014f15_dim)
def cec2014F15(X):

    pass
#===============================================#
#                     CEC2014F16                #
#===============================================#
cec2014f16_dim = 30
cec2014f16_lb = -50 * np.ones(cec2014f16_dim)
cec2014f16_ub = 50 * np.ones(cec2014f16_dim)
def cec2014F16(X):

    pass



def Fun_Plot():
    fig = plt.figure(1) #定义figure
    ax = Axes3D(fig) #将figure变为3d
    x1=np.arange(-100,100,2) #定义x1，范围为[-100,100],间隔为2
    x2=np.arange(-100,100,2) #定义x2，范围为[-100,100],间隔为2
    X1,X2=np.meshgrid(x1,x2) #生成网格
    nSize = x1.shape[0]
    Z=np.zeros([nSize,nSize])
    for i in range(nSize):
        for j in range(nSize):
            X=[X1[i,j],X2[i,j]] #构造F1输入
            X=np.array(X) #将格式由list转换为array
            Z[i,j]=F1(X)  #计算F1的值
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

Fun_Plot()
















