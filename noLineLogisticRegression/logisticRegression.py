from numpy import loadtxt,where
from pylab import scatter, show, legend, xlabel, ylabel

#load the dataset
data = loadtxt("F:/PythonCode/LogisticRegression/data2.txt", delimiter=",")
#可以看出数据是一个二维数组，维度是100*3
print(data)

X = data[:,0:2]
#X存放的是数据的特征，维度是：100*2
# print(X.shape)
y = data[:, 2]
#y存放的是数据的标签，维度是：100*1
# print(y)

pos = where(y == 1)
#pos是y中数据等于1的下标索引
# print(pos)
neg = where(y==0)
#neg是y中数据等于0的下标索引
# print(neg)

#python中数据可视化函数scatter(数据的横坐标向量，数据的纵坐标向量，marker='0'数据以点的形式显示，c='b'数据点是blue颜色)
scatter(X[pos,0],X[pos, 1],marker='o', c='b')
scatter(X[neg,0],X[neg, 1],marker='x', c='r')

#说明二维坐标中o表示Pass,x表示Fail
legend(["y==1","y==0"])
show()