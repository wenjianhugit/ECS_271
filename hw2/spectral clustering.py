import csv
import random
import math
import operator
import numpy as np
from sklearn.cluster import KMeans

# loadDataset1 will load the training data and split the training data into
# two parts, with one training set and another test set 
def loadDataset1(filename, test_mark, trainingSet, testSet):
    with open(filename, 'rU') as csvfile: 
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            gotoint=[]
            for y in range(len(dataset[x+1])-1):
                gotoint.append(int(dataset[x+1][y]))
            if (x % 100 == test_mark):
                testSet.append(gotoint)
            else:
                trainingSet.append(gotoint)

# loadDataset2 will load the training data from one file and load the test data
# from another file 
def loadDataset2(filename_training, filename_test, trainingSet, testSet):
    with open(filename_training, 'rU') as csvfile_training:
        lines_training= csv.reader(csvfile_training)
        dataset_training = list(lines_training)
        for x in range(len(dataset_training)-1):
            gotoint_training=[]
            for y in range(len(dataset_training[x+1])-1):
                gotoint_training.append(int(dataset_training[x+1][y]))
            trainingSet.append(gotoint_training)
    with open(filename_test, 'rU') as csvfile_test:
        lines_test= csv.reader(csvfile_test)
        dataset_test = list(lines_test)
        for x in range(len(dataset_test)):
            gotoint_test=[]
            for y in range(len(dataset_test[x])-2):
                gotoint_test.append(int(dataset_test[x][y]))
            testSet.append(gotoint_test)
                
# getmatrix will collect necessary data from training set and store them in 
# more organized matrixes. temp_matrix consists of lists with first item being its
# user ID, while the rest items being its movie IDs. temp_matrix2 consists of 
# lists with first item being its user ID, while the rest items being the corresponding
# movie rating. temp_matrix3 consists of lists with first item being its user ID,
# while the second item being the absolute length of rating vector .
def getmatrix(dataSet, temp_matrix, temp_matrix2, temp_matrix3):
    nodenum=0
    for x in range(len(dataSet)):
        mark=0
        for y in range(len(temp_matrix)):
            if((dataSet[x][1]==temp_matrix[y][0])):
                mark=1
                temp_matrix[y].append(dataSet[x][0])
                temp_matrix2[y].append(dataSet[x][2])
        if(mark==0):
            temp_matrix.append([dataSet[x][1],dataSet[x][0]])
            temp_matrix2.append([dataSet[x][1],dataSet[x][2]])
            temp_matrix3.append([dataSet[x][1]])
    nodenum=len(temp_matrix)
    for x in range(len(temp_matrix2)):
        sum=0
        for y in range(len(temp_matrix2[x])-1):
            sum+=temp_matrix2[x][y+1]**2
        temp_matrix3[x].append(math.sqrt(sum))
    print 'There are '+repr(nodenum)+' users !'
    
def average_rating(dataSet):
    aver_rating=0
    for x in range(len(dataSet)):
        aver_rating+=dataSet[x][2]
    aver_rating=aver_rating/(1.00*len(dataSet))
    return aver_rating
    
# Build up a huge matrix table, with the row number being its user ID, the 
# colume number being its movie ID and the content being its movie rating.
def huge_matrix_cal(dataSet):
    L=len(dataSet)
    movieID_max=int(dataSet[0][0])
    userID_max=int(dataSet[0][1])
    for x in range(L):
        if(movieID_max<dataSet[x][0]):
            movieID_max=int(dataSet[x][0])
        if(userID_max<dataSet[x][1]):
            userID_max=int(dataSet[x][1])
    huge_matrix=np.zeros((userID_max+1,movieID_max+1),int)
    for x in range(L):
        userID=dataSet[x][1]
        movieID=dataSet[x][0]
        huge_matrix[userID][movieID]=dataSet[x][2]
    return huge_matrix

# return different clusters according to the cluster prediction of each user ID 
def clusters_cal(k,Y_pred,temp_matrix):
    clusterList=[]
    for x in range(k):
        clu_list=[]
        for y in range(len(Y_pred)):
            if(Y_pred[y]==x):
                clu_list.append(temp_matrix[y][0])
        clusterList.append(clu_list)
    return clusterList

# Build up a simple search function, directly returning what user ID it is by 
# knowing its place (location) in the matrix temp_matrix
def userID_quick_search(dataSet,temp_matrix):
    L=len(dataSet)
    userID_max=int(dataSet[0][1])
    for x in range(L):
        if(userID_max<dataSet[x][1]):
            userID_max=int(dataSet[x][1])
    userID_search=np.zeros((userID_max+1),int)
    for x in range(len(temp_matrix)):
        userID_search[temp_matrix[x][0]]=x
    return userID_search
        
# Return the adjacency matrix. One user corresponds one node. the kernel function is Gaussian similarity function
def similarity_cal(temp_matrix,temp_matrix2,std):
    f=open('check_similarity_cal','w')
    L=len(temp_matrix)
    similarity_matrix=np.zeros((L,L),float)
    for x in range(L):
        print 'x='+repr(x)
        for y in range(L):
            if (x>y):
                similarity=0
                for z1 in range(len(temp_matrix[x])-1):
                    check=0
                    for z2 in range(len(temp_matrix[y])-1):
                        if(temp_matrix[x][z1+1]==temp_matrix[y][z2+1]):
                            similarity+=math.exp(-(temp_matrix2[x][z1+1]-temp_matrix2[y][z2+1])**2/(2*std*std))
                            check+=1
                            if(check>=2):
                                f.write('x='+repr(x)+'   y='+repr(y)+'   '+repr(temp_matrix[y][z2+1])+'\n')
                    if(check>=2):
                        f.write("The wired user id is "+repr(temp_matrix[y][0])+'! \n')
                similarity_matrix[x][y]=similarity
    for x in range(L):
        for y in range(L):
            if (x<y):
                similarity_matrix[x][y]=similarity_matrix[y][x]
    return similarity_matrix

# Return the adjacency matrix. One user corresponds one node. The kernel function is inner product
# Use the huge matrix returned by "huge_matrix_cal" to accelerate the calculation
def similarity_cal2(huge_matrix,temp_matrix,temp_matrix2,temp_matrix3):
    L=len(temp_matrix)
    similarity_matrix=np.zeros((L,L),float)
    for x in range(L):
        print 'x='+repr(x)
        for y in range(L):
            if (x<y):
                similarity=0
                for z1 in range(len(temp_matrix[x])-1):
                    y_grade=huge_matrix[temp_matrix[y][0]][temp_matrix[x][z1+1]]
                    if(y_grade!=0):
                        similarity+=temp_matrix2[x][z1+1]*y_grade
                similarity=similarity/(temp_matrix3[x][1]*temp_matrix3[y][1]*1.00)
                similarity_matrix[x][y]=similarity
    for x in range(L):
        for y in range(L):
            if (x>y):
                similarity_matrix[x][y]=similarity_matrix[y][x]
    return similarity_matrix

# Return the adjacency matrix. One user corresponds one node. the kernel function is Gaussian similarity function
# Use the huge matrix returned by "huge_matrix_cal" to accelerate the calculation
def similarity_cal3(huge_matrix,temp_matrix,temp_matrix2,std):
    L=len(temp_matrix)
    similarity_matrix=np.zeros((L,L),float)
    for x in range(L):
        print 'x='+repr(x)
        for y in range(L):
            if (x<y):
                similarity=0
                for z1 in range(len(temp_matrix[x])-1):
                    y_grade=huge_matrix[temp_matrix[y][0]][temp_matrix[x][z1+1]]
                    if(y_grade!=0):
                        similarity+=math.exp(-(temp_matrix2[x][z1+1]-y_grade)**2/(2*std*std))
                similarity_matrix[x][y]=similarity
    for x in range(L):
        for y in range(L):
            if (x>y):
                similarity_matrix[x][y]=similarity_matrix[y][x]
    return similarity_matrix

# Return degree matrix
def degree_cal(similarity_matrix):
    L=len(similarity_matrix)
    degree_matrix=np.zeros((L,L),float)
    for x in range(L):
        sum=0
        for y in range(L):
            sum+=similarity_matrix[x][y]
        degree_matrix[x][x]=sum
    return degree_matrix

# Return Graph Laplacians
def L_cal(degree_matrix, similarity_matrix):
    L=len(similarity_matrix)
    L_matrix=np.zeros((L,L),float)
    for x in range(L):
        for y in range(L):
            L_matrix[x][y]=degree_matrix[x][y]-similarity_matrix[x][y]
    return L_matrix


# make predictions for the test set.
def predictions_cal(huge_matrix,clusterList,userID_search,Y_pred,testSet,aver_rating):
    predictions=[]
    f10=open('weird_predictions','w')
    for x in range(len(testSet)):
        movieID=testSet[x][0]
        userID=testSet[x][1]
        cluster=Y_pred[userID_search[userID]]
        grade=0
        counter=0
        for y in range(len(clusterList[cluster])):
            newuserID=clusterList[cluster][y]
            if ((newuserID != userID)and(huge_matrix[newuserID][movieID]!=0)):
                grade+=huge_matrix[newuserID][movieID]
                counter+=1
        if(counter!=0):
            grade=grade/(counter*1.00)
        else:
            grade=aver_rating
            f10.write('This is cluster '+repr(cluster)+' and it has '+repr(len(clusterList[cluster]))+' members\n')
            f10.write('user ID= '+repr(userID)+'\n')
            f10.write('movie ID= '+repr(movieID)+'\n\n\n')
        predictions.append(round(grade,1))
    return predictions

# Call loadDataset1 and split the training data into two sets, the training set and the test set
# Predict the test set and compare the prediction with real rating. Return the relative accuracy (loss function)
def main():
    f1=open('clusterList','w')
    f3=open('user id','w')
    f5=open('test data predictions','w')
    f6=open('new test set','w')
    f7=open('eigenvalues and best cluster number k','w')
    trainingSet=[]
    testSet=[]
    nodes=[]
    loadDataset1('train.csv',2,trainingSet,testSet)
    #loadDataset2('train.csv','test.csv',trainingSet,testSet)
    print len(trainingSet)
    print len(testSet)
    print 'We are going to calculate the huge matrix !'
    aver_rating=average_rating(trainingSet)
    huge_matrix=huge_matrix_cal(trainingSet)
    temp_matrix=[]
    temp_matrix2=[]
    temp_matrix3=[]
    getmatrix(trainingSet,temp_matrix,temp_matrix2,temp_matrix3)
    #similarity_matrix=similarity_cal(temp_matrix,temp_matrix2,1.40)
    similarity_matrix=similarity_cal2(huge_matrix,temp_matrix,temp_matrix2,temp_matrix3)
    #similarity_matrix=similarity_cal3(huge_matrix,temp_matrix,temp_matrix2,0.5)
    L=len(temp_matrix)
    for x in range(L):
        f3.write(repr(temp_matrix[x][0])+'\n')
    degree_matrix=degree_cal(similarity_matrix)
    L_matrix=L_cal(degree_matrix,similarity_matrix)
    #choose k clusters
    k=3
    v,w=np.linalg.eig(L_matrix)
    v=v.real
    w=w.real
    ev_list = zip(v,w)
    ev_list.sort(key=lambda tup:tup[0], reverse=False)
    v,w= zip(*ev_list)
    for x in range(len(v)):
        f7.write(repr(v[x])+'\n')
    Y=[]
    for x in range(L):
        YY=[]
        for y in range(k):
            YY.append(w[y][x])
        Y.append(YY)
    Y_pred=KMeans(n_clusters=k, max_iter=600).fit_predict(Y)
    print 'We are going to calculate the cluster matrix !'
    clusterList=clusters_cal(k,Y_pred,temp_matrix)
    for x in range(len(clusterList)):
        f1.write('This is '+repr(x)+'th cluster and it has members:\n')
        f1.write(repr(clusterList[x])+'\n\n\n\n')
    print 'We are going to calculate the userID quick search list!'
    userID_search=userID_quick_search(trainingSet,temp_matrix)
    predictions=predictions_cal(huge_matrix,clusterList,userID_search,Y_pred,testSet,aver_rating)
    print 'Average rating is '+repr(aver_rating)
    for x in range(len(predictions)):
        f5.write(repr(predictions[x])+'\n')
        testSet[x].append(predictions[x])
        f6.write(repr(testSet[x])+'\n')
    loss_function=0
    for x in range(len(predictions)):
        loss_function+=(predictions[x]-testSet[x][2])**2
    loss_function=loss_function/(1.00*len(testSet))
    print 'The average loss function is '+repr(loss_function)
    f5.write('The average loss function is '+repr(loss_function)+'\n')

# Call loadDataset2 and load the training data from one file, the test data from another file. 
# Predict the test set.
def main2():
    f1=open('clusterList','w')
    f3=open('user id','w')
    f5=open('test data predictions','w')
    f6=open('new test set','w')
    f7=open('eigenvalues and best cluster number k','w')
    trainingSet=[]
    testSet=[]
    nodes=[]
    #loadDataset1('train.csv',2,trainingSet,testSet)
    loadDataset2('train.csv','test.csv',trainingSet,testSet)
    print len(trainingSet)
    print len(testSet)
    print 'We are going to calculate the huge matrix !'
    aver_rating=average_rating(trainingSet)
    huge_matrix=huge_matrix_cal(trainingSet)
    temp_matrix=[]
    temp_matrix2=[]
    temp_matrix3=[]
    getmatrix(trainingSet,temp_matrix,temp_matrix2,temp_matrix3)
    #similarity_matrix=similarity_cal(temp_matrix,temp_matrix2,1.40)
    similarity_matrix=similarity_cal2(huge_matrix,temp_matrix,temp_matrix2,temp_matrix3)
    #similarity_matrix=similarity_cal3(huge_matrix,temp_matrix,temp_matrix2,0.5)
    L=len(temp_matrix)
    for x in range(L):
        f3.write(repr(temp_matrix[x][0])+'\n')
    degree_matrix=degree_cal(similarity_matrix)
    L_matrix=L_cal(degree_matrix,similarity_matrix)
    #choose k clusters
    k=3
    v,w=np.linalg.eig(L_matrix)
    v=v.real
    w=w.real
    ev_list = zip(v,w)
    ev_list.sort(key=lambda tup:tup[0], reverse=False)
    v,w= zip(*ev_list)
    for x in range(len(v)):
        f7.write(repr(v[x])+'\n')
    Y=[]
    for x in range(L):
        YY=[]
        for y in range(k):
            YY.append(w[y][x])
        Y.append(YY)
    Y_pred=KMeans(n_clusters=k, max_iter=600).fit_predict(Y)
    print 'We are going to calculate the cluster matrix !'
    clusterList=clusters_cal(k,Y_pred,temp_matrix)
    for x in range(len(clusterList)):
        f1.write('This is '+repr(x)+'th cluster and it has members:\n')
        f1.write(repr(clusterList[x])+'\n\n\n\n')
    print 'We are going to calculate the userID quick search list!'
    userID_search=userID_quick_search(trainingSet,temp_matrix)
    predictions=predictions_cal(huge_matrix,clusterList,userID_search,Y_pred,testSet,aver_rating)
    print 'Average rating is '+repr(aver_rating)
    for x in range(len(predictions)):
        f5.write(repr(predictions[x])+'\n')
        testSet[x].append(predictions[x])
        f6.write(repr(testSet[x])+'\n')
        

# call main2() to make predictions for the test data
main2()