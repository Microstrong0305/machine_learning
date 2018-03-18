from numpy import *
import xlrd
import matplotlib.pyplot as plt

# 计算欧氏距离
def distEclud(vector1, vector2):
    '''
    :param vector1: 第j个均值向量
    :param vector2: 第i个样本
    :return: 距离值
    '''
    return sqrt(sum(power(vector2 - vector1, 2)))

#构建聚簇中心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    #首先创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = mat(zeros((m,2)))
    #----创建一个初始簇----
    #计算整个数据集的质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]

    #使用一个列表来保留所有的质心
    centList = [centroid0]
    #遍历数据集中所有点来计算每个点到质心的误差值
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    #while循环会不停的对簇进行划分，直到得到想要的簇数目为止
    while(len(centList) < k):
        lowestSSE = inf    #python中inf表示正无穷
        #遍历簇列表centList中的每一个簇
        for i in range(len(centList)):
            #-----尝试划分每一簇----
            #对每个簇，将该簇中的所有点看成一个小的数据集ptsInCurrCluster
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #将ptsInCurrCluster输入到函数KMeans()中进行处理(k=2)。K-均值算法会生成两个质心(簇)，同时给出每个簇的误差值
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)

            #划分后的样本误差之和
            sseSplit = sum(splitClustAss[:, 1])
            #剩余数据集的误差之和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            #将划分后的样本误差之和+剩余数据集的误差之和作为本次划分的误差
            if(sseSplit + sseNotSplit) < lowestSSE:
                #决定要划分某一个簇
                bestCentToSplit = i
                #划分后的质心
                bestNewCents = centroidMat
                #划分后的簇分配结果矩阵，包含两列：第一列记录簇索引，第二列存储误差
                bestClustAss = splitClustAss.copy()
                # 本次划分的误差
                lowestSSE = sseSplit + sseNotSplit
            #----更新簇的分配结果----
        #将划分簇中所有点的簇分配结果进行修改，当使用KMeans()函数并且指定簇数为2时，会得到两个编号分别为0和1的结果簇
        #需要将这些簇编号修改为划分簇及新加簇的编号，该过程可以通过两个数组过滤器来完成。
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        #新的质心会被添加到centList中
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        # 更新SSE的值(sum of squared errors)
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment

#k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))    #用于存放该样本属于哪类及质心距离
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True;
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = mean(ptsInClust, axis = 0)
    return centroids, clusterAssment

# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print ("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print ("Sorry! Your k is too large! please contact Zouxy")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()

def main():
    ## step 1: load data
    print ("step 1: load data...")
    dataSet = []
    data = xlrd.open_workbook('C:/Users/Microstrong/Desktop/watermelon4.0.xlsx')
    table = data.sheets()[0]
    for line in range(0,table.nrows):
        lineArr = table.row_values(line)
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

    ## step 2: clustering...
    print ("step 2: clustering...")
    dataSet = mat(dataSet)
    k = 3
    centroids, clusterAssment = biKmeans(dataSet, k)

    ## step 3: show the result
    print ("step 3: show the result...")
    showCluster(dataSet, k, centroids, clusterAssment)

if __name__ == '__main__':
 main()



