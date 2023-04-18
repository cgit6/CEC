'''F1绘图函数'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def F1(X):
    Results= np.sum(X**2)
    return Results

def F2(X):
    Results=np.sum(np.abs(X))+np.prod(np.abs(X))
    return Results

def F3(X):
    dim=X.shape[0]
    Results=0
    for i in range(dim):
        Results=Results+np.sum(X[0:i+1])**2

    return Results

def F4(X):
    Results=np.max(np.abs(X))

    return Results

def F5(X):
    dim=X.shape[0]
    Results=np.sum(100*(X[1:dim]-(X[0:dim-1]**2))**2+(X[0:dim-1]-1)**2)

    return Results

def F6(X):
    Results=np.sum(np.abs(X+0.5)**2)

    return Results

def F7(X):
    dim = X.shape[0]
    Temp = np.arange(1,dim+1,1)
    Results=np.sum(Temp*(X**4))+np.random.random()

    return Results

def F8(X):
    
    Results=np.sum(-X*np.sin(np.sqrt(np.abs(X))))

    return Results

def F9(X):
    dim=X.shape[0]
    Results=np.sum(X**2-10*np.cos(2*np.pi*X))+10*dim

    return Results

def F10(X):
    dim=X.shape[0]
    Results=-20*np.exp(-0.2*np.sqrt(np.sum(X**2)/dim))-np.exp(np.sum(np.cos(2*np.pi*X))/dim)+20+np.exp(1)

    return Results

def F11(X):
    dim=X.shape[0]
    Temp=np.arange(1,dim,1)
    Results=np.sum(X**2)/4000-np.prod(np.cos(X/np.sqrt(Temp)))+1

    return Results

def Ufun(x,a,k,m):
    Results=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<-a)
    return Results

def F12(X):
    dim=X.shape[0]
    Results=(np.pi/dim)*(10*((np.sin(np.pi*(1+(X[0]+1)/4)))**2)+\
             np.sum(((X[0:dim-1]+1)/4)**2)*(1+10*((np.sin(np.pi*(1+X[1:dim]+1)/4))**2))+\
             ((X[dim-1]+1)/4)**2)+np.sum(Ufun(X,10,100,4))

    return Results

def F13(X):
    dim=X.shape[0]
    Results=0.1*((np.sin(3*np.pi*X[0]))**2+np.sum((X[0:dim-1]-1)**2*(1+(np.sin(3*np.pi*X[1:dim]))**2))+\
                 ((X[dim-1]-1)**2)*(1+(np.sin(2*np.pi*X[dim-1]))**2))+np.sum(Ufun(X,5,100,4))

    return Results

def F14(X):
    aS=np.array([[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],\
                 [-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]])
    bS=np.zeros(25)
    for i in range(25):
        bS[i]=np.sum((X-aS[:,i])**6)
    Temp=np.arange(1,26,1)
    Results=(1/500+np.sum(1/(Temp+bS)))**(-1)
    
    return Results

def F15(X):
    aK=np.array([0.1957,0.1947,0.1735,0.16,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
    bK=np.array([0.25,0.5,1,2,4,6,8,10,12,14,16])
    bK=1/bK
    Results=np.sum((aK-((X[0]*(bK**2+X[1]*bK))/(bK**2+X[2]*bK+X[3])))**2)
    
    return Results

def F16(X):
    Results=4*(X[0]**2)-2.1*(X[0]**4)+(X[0]**6)/3+X[0]*X[1]-4*(X[1]**2)+4*(X[1]**4)
    return Results

def F17(X):
    Results=(X[1]-(X[0]**2)*5.1/(4*(np.pi**2))+(5/np.pi)*X[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(X[0])+10
    return Results

def F18(X):
    Results=(1+(X[0]+X[1]+1)**2*(19-14*X[0]+3*(X[0]**2)-14*X[1]+6*X[0]*X[1]+3*X[1]**2))*\
    (30+(2*X[0]-3*X[1])**2*(18-32*X[0]+12*(X[0]**2)+48*X[1]-36*X[0]*X[1]+27*(X[1]**2)))
    return Results

def F19(X):
    aH=np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
    cH=np.array([1,1.2,3,3.2])
    pH=np.array([[0.3689,0.117,0.2673],[0.4699,0.4387,0.747],[0.1091,0.8732,0.5547],[0.03815,0.5743,0.8828]])
    Results=0
    for i in range(4):
        Results=Results-cH[i]*np.exp(-(np.sum(aH[i,:]*((X-pH[i,:]))**2)))
    return Results

def F20(X):
    aH=np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
    cH=np.array([1,1.2,3,3.2])
    pH=np.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],[0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],\
                 [0.2348,0.1415,0.3522,0.2883,0.3047,0.6650],[0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
    Results=0
    for i in range(4):
        Results=Results-cH[i]*np.exp(-(np.sum(aH[i,:]*((X-pH[i,:]))**2)))
    return Results

def F21(X):
    aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],\
                  [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH=np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    Results=0
    for i in range(5):
        Results=Results-(np.dot((X-aSH[i,:]),(X-aSH[i,:]).T)+cSH[i])**(-1)
    return Results

def F22(X):
    aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],\
                  [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH=np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    Results=0
    for i in range(7):
        Results=Results-(np.dot((X-aSH[i,:]),(X-aSH[i,:]).T)+cSH[i])**(-1)
    return Results


def F23(X):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    Results = 0
    for i in range(10):
        Results = Results-(np.dot((X-aSH[i, :]), (X-aSH[i, :]).T)+cSH[i])**(-1)
    return Results


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

