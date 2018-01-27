from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

filename = "F:/PythonCode/LogisticRegression/data1.txt"

def loadDataSet():
    # load the dataset
    data = loadtxt("F:/PythonCode/LogisticRegression/data1.txt", delimiter=",")
    # np.c_按colunm来组合array
    X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
    y = np.c_[data[:, 2]]
    return data,X,y


#计算Sigmoid函数
def sigmoid(X):
    '''Compute sigmoid function'''
    den = 1.0 + exp(-1.0*X)
    gz = 1.0/den
    return gz

#定义损失函数
def compute_cost(theta,X,y):
    '''computes cost given predicted and actual values'''
    m = y.size #number of training examples
    h = sigmoid(X.dot(theta))

    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    #optimize.fmin expects a single value, so cannot return grad
    if np.isnan(J[0]):
        return (np.inf)
    return  J[0]#grad

#求解梯度
def compute_grad(theta, X, y):
    '''compute gradient'''
    m = y.size
    h=sigmoid(X.dot(theta.reshape(-1,1)))

    grad = (1.0/m)*X.T.dot(h-y)
    return (grad.flatten())

def gradAscent(X, y):
    initial_theta = np.zeros(X.shape[1])
    cost = compute_cost(initial_theta, X, y)
    grad = compute_grad(initial_theta, X, y)
    print('Cost: \n', cost)
    print('Grad: \n', grad)
    res = minimize(compute_cost, initial_theta, args=(X, y), jac=compute_grad, options={'maxiter': 400})
    return res

def plotBestFit(data,res,X,score):  #画出最终分类的图
    plt.scatter(score[1], score[2], s=60, c='r', marker='v', label='('+str(score[1])+','+str(score[2])+')')
    plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max(),
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res.x))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.show()

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)

def predict(theta,input,threshold=0.5):
    p = sigmoid(input.dot(theta.T)) >= threshold
    return (p.astype('int'))


def main():
    data, X, y = loadDataSet()
    res = gradAscent(X, y)
    print(res)
    print("请输入您要预测的值(用空格隔开)：")
    input_score = input()
    num = [int(n) for n in input_score.split()]
    score = np.array(num,dtype=int)
    print("您预测的成绩:%d" %predict(res.x,score))
    plotBestFit(data,res,X,score)


if __name__=='__main__':
    main()


