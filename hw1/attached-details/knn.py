import csv
import random
import math
import operator
import numpy as np

# loadDataset1 will load the training data and split the training data into
# two parts, with one training set and another test set 
def loadDataset1(filename, test_mark, trainingSet, testSet):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            if (x % 10 == test_mark):
                testSet.append(dataset[x])
            else:
                trainingSet.append(dataset[x])

# loadDataset2 will load the training data from one file and load the test data
# from another file  
def loadDataset2(filename_training, filename_test, trainingSet, testSet):
    with open(filename_training, 'rb') as csvfile_training:
        lines_training= csv.reader(csvfile_training)
        dataset_training = list(lines_training)
        for x in range(len(dataset_training)-1):
            trainingSet.append(dataset_training[x])
    with open(filename_test, 'rb') as csvfile_test:
        lines_test= csv.reader(csvfile_test)
        dataset_test = list(lines_test)
        for x in range(len(dataset_test)-1):
            testSet.append(dataset_test[x])

# loadDataset3 will only load the training data from one file 
def loadDataset3(filename_training, trainingSet):
    with open(filename_training, 'rb') as csvfile_training:
        lines_training= csv.reader(csvfile_training)
        dataset_training = list(lines_training)
        for x in range(len(dataset_training)-1):
            trainingSet.append(dataset_training[x])
#Use loadDataset1 or loadDataset2 or loadDataset3          

# Calculate the similarity(distance) between any two given data instances
def euclideanDistance(instance1, instance2, length):
    distance=0
    for x in range(length):
        distance += pow((int(instance1[x])- int(instance2[x])),2)
    return math.sqrt(distance)

# Use it to collect the k most similar(closest) neighbors for a given instance
def getNeighbors(trainingSet, testInstance, k):
    distances=[]
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
        
 
# Give a predicted response(class) based on its k most similar neighbors
# We can do this by allowing each neighbor to vote for their class attribute, 
# and take the majority vote as the prediction
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    sortedVotes=sorted(classVotes.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

# Evaluate the accuracy of the model by calculating a ratio of the total correct
# predictions out of all predictions made, called the classification accuracy
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct +=1
    return (correct/float(len(testSet)))*100.0

# Define the main function: return 10 cross validation results (with confusion matrix)
def main1():
    f1=open('knn output','w')
    f2=open('knn confusion matrix','w')
    confumatrix_tot=np.zeros((10,10),int)
    accuracy_overall=0
    for counter in range(10):
        trainingSet=[]
        testSet=[]
        loadDataset1('pendigits-train.csv' ,counter, trainingSet ,testSet )
        print 'Counter='+repr(counter)
        print 'Train set:'+repr(len(trainingSet))
        print 'Test set:'+repr(len(testSet))
        f1.write('Counter='+repr(counter)+'\n')
        f1.write('Train set:'+repr(len(trainingSet))+'\n')
        f1.write('Test set:'+repr(len(testSet))+'\n')
        # generate predictions
        predictions=[]
        # choose 5 closest neighbors
        k=5
        for x in range(len(testSet)):
            neighbors=getNeighbors(trainingSet,testSet[x],k)
            result=getResponse(neighbors)
            predictions.append(result)
            print('> predicted'+repr(result)+',actual='+repr(testSet[x][-1]))
            f1.write('> predicted'+repr(result)+',actual='+repr(testSet[x][-1])+'\n')
        accuracy=getAccuracy(testSet,predictions)
        accuracy_overall+=accuracy
        print('Accuracy: '+repr(accuracy)+'%')
        f1.write('Accuracy: '+repr(accuracy)+'%'+'\n\n\n')
        confumatrix=np.zeros((10,10),int)
        for x in range(len(testSet)):
            x_p=int(testSet[x][-1])
            y_p=int(predictions[x])
            confumatrix[x_p][y_p]+=1
        print confumatrix
        f2.write('    This is the '+repr(counter)+'th confusion matrix ! \n\n')
        for x in range(10):
            for y in range(10):
                confumatrix_tot[x][y]+=confumatrix[x][y]
                f2.write('   '+repr(confumatrix[x][y]))
            f2.write('\n\n')
    accuracy_overall=accuracy_overall/10.00
    print('Overall Accuracy: '+repr(accuracy_overall)+'%')
    f1.write('Overall Accuracy: '+repr(accuracy_overall)+'%'+'\n\n\n')
    f2.write('    This is the overall confusion matrix ! \n\n')
    print confumatrix_tot
    for x in range(10):
        for y in range(10):
            f2.write('      '+repr(confumatrix_tot[x][y]))
        f2.write('\n\n')
  

# Define the main function: get the predictions for the test data
def main2():
    f1=open('knn predict test data','w')
    trainingSet=[]
    testSet=[]
    loadDataset2('pendigits-train.csv' ,'pendigits-test-nolabels.csv' ,trainingSet ,testSet )
    print 'Train set: '+repr(len(trainingSet))
    print 'Test set: '+repr(len(testSet))
    f1.write('Train set: '+repr(len(trainingSet))+'\n')
    f1.write('Test set: '+repr(len(testSet))+'\n')
    # generate predictions
    predictions=[]
    # choose 5 closest neighbors
    k=5
    for x in range(len(testSet)):
        neighbors=getNeighbors(trainingSet,testSet[x],k)
        result=getResponse(neighbors)
        predictions.append(result)
        print result
        f1.write(result+'\n')


# I want to know the predictions of the test data, so I call main2()
main2()




















       
        
