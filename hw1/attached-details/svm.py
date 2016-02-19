import csv
import random
import math
import operator
import numpy as np
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
from sklearn import svm

# loadDataset1 will load the training data and split the training data into
# two parts, with one training set and another test set 
def loadDataset1(filename, test_mark, trainingSet, testSet):
    with open(filename, 'rU') as csvfile: 
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            gotoint=[]
            for y in range(len(dataset[x])):
                gotoint.append(int(dataset[x][y]))
            if (x % 10 == test_mark):
                testSet.append(gotoint)
            else:
                trainingSet.append(gotoint)

# loadDataset2 will load the training data from one file and load the test data
# from another file  
def loadDataset2(filename_training, filename_test, trainingSet, testSet):
    with open(filename_training, 'rU') as csvfile_training:
        lines_training= csv.reader(csvfile_training)
        dataset_training = list(lines_training)
        for x in range(len(dataset_training)):
            gotoint_training=[]
            for y in range(len(dataset_training[x])):
                gotoint_training.append(int(dataset_training[x][y]))
            trainingSet.append(gotoint_training)
    with open(filename_test, 'rU') as csvfile_test:
        lines_test= csv.reader(csvfile_test)
        dataset_test = list(lines_test)
        for x in range(len(dataset_test)):
            gotoint_test=[]
            for y in range(len(dataset_test[x])):
                gotoint_test.append(int(dataset_test[x][y]))
            testSet.append(gotoint_test)

# loadDataset3 will load the source and target data. It will make the source data 
#as one training set and also split the target data into two parts (another training set and one test set).
def loadDataset3(filename_source, filename_target, trainingSet, trainingSet2, test_mark, testSet):
    with open(filename_source, 'rU') as csvfile_source:
        lines_source= csv.reader(csvfile_source)
        dataset_source = list(lines_source)
        for x in range(len(dataset_source)):
            gotoint_source=[]
            for y in range(len(dataset_source[x])):
                gotoint_source.append(int(dataset_source[x][y]))
            trainingSet.append(gotoint_source)
    with open(filename_target, 'rU') as csvfile_target:
        lines_target= csv.reader(csvfile_target)
        dataset_target= list(lines_target)
        for x in range(len(dataset_target)):
            gotoint_target=[]
            for y in range(len(dataset_target[x])):
                gotoint_target.append(int(dataset_target[x][y]))
            if (x % 10 == test_mark):
                testSet.append(gotoint_target)
            else:
                trainingSet2.append(gotoint_target)

# loadDataset4 will load the source data from one file, the target data from one file and
# the test data from one file
def loadDataset4(filename_source, filename_target, filename_test ,trainingSet, trainingSet2, testSet):
    with open(filename_source, 'rU') as csvfile_source:
        lines_source= csv.reader(csvfile_source)
        dataset_source = list(lines_source)
        for x in range(len(dataset_source)):
            gotoint_source=[]
            for y in range(len(dataset_source[x])):
                gotoint_source.append(int(dataset_source[x][y]))
            trainingSet.append(gotoint_source)
    with open(filename_target, 'rU') as csvfile_target:
        lines_target= csv.reader(csvfile_target)
        dataset_target= list(lines_target)
        for x in range(len(dataset_target)):
            gotoint_target=[]
            for y in range(len(dataset_target[x])):
                gotoint_target.append(int(dataset_target[x][y]))
            trainingSet2.append(gotoint_target)
    with open(filename_test, 'rU') as csvfile_test:
        lines_test= csv.reader(csvfile_test)
        dataset_test= list(lines_test)
        for x in range(len(dataset_test)):
            gotoint_test=[]
            for y in range(len(dataset_test[x])):
                gotoint_test.append(int(dataset_test[x][y]))
            testSet.append(gotoint_test)
#Use loadDataset1 or loadDataset2 or loadDataset3 or loadDataset4

# A small subroutine to help search support vectors
def search_sup_vectors(record,z1):
    judge=0
    for x in range(len(record)):
        if(int(z1)==int(record[x])):
            judge=1
    return judge

# Return fx based on the input coefficients (omega and omega_0) and instance
def getfx(coef,testInstance):
    sum=0
    for x in range(16):
        sum += coef[x]*testInstance[x]
    sum +=coef[16]
    return sum

# Return fx based on the input coefficients and instance. getfx is used only for 
# linear case while getfx2 is used for any available kernels
def getfx2(alphas,beta0,y_counter,trainingSet_X,testInstance,kernel_name):
    sum=0
    if(kernel_name=='linear'):
        for x in range(len(alphas)):
            sum+=alphas[x]*y_counter[x]*kernel_linear(testInstance,trainingSet_X[x])
        sum+=beta0
    if(kernel_name=='poly'):
        for x in range(len(alphas)):
            sum+=alphas[x]*y_counter[x]*kernel_poly(testInstance,trainingSet_X[x],2)
        sum+=beta0
    if(kernel_name=='rbf'):
        for x in range(len(alphas)):
            sum+=alphas[x]*y_counter[x]*kernel_rbf(testInstance,trainingSet_X[x],1)
        sum+=beta0
    return sum

# Linear kernel Calculation
def kernel_linear(instance1,instance2):
    sum=0
    for x in range(16):
        sum += instance1[x]*instance2[x]
    return sum

# Polynomial kernel Calculation
def kernel_poly(instance1,instance2,power):
    sum=0
    for x in range(16):
        sum += instance1[x]*instance2[x]
    sum=math.pow(sum+1,power)
    return sum

# Radial basis kernel Calculation
def kernel_rbf(instance1,instance2,gamma):
    sum=0
    for x in range(16):
        sum += (instance1[x]-instance2[x])*(instance1[x]-instance2[x])
    sum=math.exp(-gamma*sum)
    return sum

# Build up the kernel matrix K(x_i,x_j)
def kernel_matrix_create(x_training,kernel_name,power,gamma):
    L=len(x_training)
    kernel_matrix=np.zeros((L,L),float)
    for x in range(L):
        for y in range(L):
            if(kernel_name=='linear'):
                kernel_matrix[x][y]=kernel_linear(x_training[x],x_training[y])
            if(kernel_name=='poly'):
                kernel_matrix[x][y]=kernel_poly(x_training[x],x_training[y],power)
            if(kernel_name=='rbf'):
                kernel_matrix[x][y]=kernel_rbf(x_training[x],x_training[y],gamma)
    return kernel_matrix

# Get a predicted responses(classes) based on the quadratic program solver
# This subroutine solves the original svm optimization functions
def getPrediction1(trainingSet,testSet):
    predictions=[]
    X=[]
    Y=[]
    for x in range(len(trainingSet)):
        X.append(trainingSet[x][0:16])
        Y.append(trainingSet[x][-1])
    # define trade-off C
    C=1.0
    L=len(trainingSet)
    solutions=np.zeros((10,17),float)
    for counter in range(10):
        P=np.zeros((L+17,L+17),float)
        q=np.zeros((L+17,1),float)
        G=np.zeros((L*2,L+17),float)
        h=np.zeros((L*2,1),float)
        for z1 in range(16):
            P[z1][z1]=1
        for z2 in range(L):
            q[z2+17][0]=C
        for z3 in range(L):
            G[z3][z3+17]=-1
            G[z3+L][z3+17]=-1
        for z4 in range(L):
            if(trainingSet[z4][-1]==counter):
                y_counter=1
            else:
                y_counter=-1
            for z5 in range(17):
                if(z5==16):
                    G[z4][z5]=-y_counter
                else:
                    G[z4][z5]=-y_counter*trainingSet[z4][z5]
        for z6 in range(L):
            h[z6][0]=-1
        P=matrix(P)
        q=matrix(q)
        G=matrix(G)
        h=matrix(h)
        print counter
        sol=solvers.qp(P,q,G,h)
        for z1 in range(17):
            solutions[counter][z1]=sol['x'][z1]
        print solutions[counter]
        
    for x in range(len(testSet)):
        test1=getfx(solutions[0],testSet[x])
        label=0
        for z1 in range(9):
            test2=getfx(solutions[z1+1],testSet[x])
            if(test1<test2):
                test1=test2
                label=z1+1
        predictions.append(label)
    return predictions

# Get a predicted responses(classes) based on the quadratic program solver
# This subroutine solves the Lagrange dual expression of the original svm optimization functions
def getPrediction2(trainingSet,testSet,kernel_name):
    f_alpha=open('alpha solutions','w')
    predictions=[]
    X=[]
    Y=[]
    for x in range(len(trainingSet)):
        X.append(trainingSet[x][0:16])
        Y.append(trainingSet[x][-1])
    # define trade-off C
    C=1.0
    L=len(trainingSet)
    solutions=np.zeros((10,L+1),float)
    kernel_matrix=kernel_matrix_create(X,kernel_name,2,1)
    y_counter_tot=[]
    for counter in range(10):
        P=np.zeros((L,L),float)
        q=np.zeros((L,1),float)
        G=np.zeros((L*2,L),float)
        h=np.zeros((L*2,1),float)
        A=np.zeros((1,L),float)
        b=np.zeros((1,1),float)
        y_counter=[]
        for z0 in range(L):
            if(trainingSet[z0][-1]==counter):
                y_counter.append(1)
            else:
                y_counter.append(-1)
        y_counter_tot.append(y_counter)
        for z0 in range(L):
            for z1 in range(L):
                P[z0][z1]=y_counter[z0]*y_counter[z1]*kernel_matrix[z0][z1]
        for z2 in range(L):
            q[z2][0]=-1
        for z3 in range(L):
            G[z3][z3]=1
            G[z3+L][z3]=-1
        for z4 in range(L):
            h[z4][0]=C
        for z5 in range(L):
            A[0][z5]=y_counter[z5]
        b[0][0]=0
        P=matrix(P)
        q=matrix(q)
        G=matrix(G)
        h=matrix(h)
        A=matrix(A)
        b=matrix(b)
        print counter
        sol=solvers.qp(P,q,G,h,A,b)
        intercept_search=0
        for z1 in range(L):
            solutions[counter][z1]=sol['x'][z1]
            if((sol['x'][z1]>0.00000001)and(sol['x'][z1]<0.99999999)and(int(intercept_search)==0)):
                print 'We found nonzero alpha '+repr(sol['x'][z1])
                intercept_search=1
                solutions[counter][L]=1.00/y_counter[z1]
                for z2 in range(L):
                    solutions[counter][L]-=sol['x'][z2]*y_counter[z2]*kernel_matrix[z1][z2]
        if(int(intercept_search)==0):
            print 'We did not find nonzero alpha'
        print solutions[counter]
        f_alpha.write('This is '+repr(counter)+' vs the rest. Its alpha solutions are:\n')
        for z1 in range(len(solutions[counter])):
            f_alpha.write(repr(solutions[counter][z1])+'\n')
        f_alpha.write('\n\n\n')
        
    for x in range(len(testSet)):
        test1=getfx2(solutions[0][0:L],solutions[0][-1],y_counter_tot[0],X,testSet[x],kernel_name)
        label=0
        for z1 in range(9):
            test2=getfx2(solutions[z1+1][0:L],solutions[z1+1][-1],y_counter_tot[z1+1],X,testSet[x],kernel_name)
            if(test1<test2):
                test1=test2
                label=z1+1
        predictions.append(label)
    return predictions

    
# Get a predicted responses(classes) based on the quadratic program solver
# This subroutine solves the original svm optimization functions and also return positive support vectors
def getPrediction3(trainingSet,testSet,sup_vectors,epsilons):
    predictions=[]
    X=[]
    Y=[]
    for x in range(len(trainingSet)):
        X.append(trainingSet[x][0:16])
        Y.append(trainingSet[x][-1])
    # define trade-off C
    C=1.0
    L=len(trainingSet)
    solutions=np.zeros((10,17),float)
    for counter in range(10):
        P=np.zeros((L+17,L+17),float)
        q=np.zeros((L+17,1),float)
        G=np.zeros((L*2,L+17),float)
        h=np.zeros((L*2,1),float)
        for z1 in range(16):
            P[z1][z1]=1
        for z2 in range(L):
            q[z2+17][0]=C
        for z3 in range(L):
            G[z3][z3+17]=-1
            G[z3+L][z3+17]=-1
        for z4 in range(L):
            if(trainingSet[z4][-1]==counter):
                y_counter=1
            else:
                y_counter=-1
            for z5 in range(17):
                if(z5==16):
                    G[z4][z5]=-y_counter
                else:
                    G[z4][z5]=-y_counter*trainingSet[z4][z5]
        for z6 in range(L):
            h[z6][0]=-1
        P=matrix(P)
        q=matrix(q)
        G=matrix(G)
        h=matrix(h)
        print counter
        sol=solvers.qp(P,q,G,h)
        for z1 in range(17):
            solutions[counter][z1]=sol['x'][z1]
        print solutions[counter]
        sup_vec=[]
        epsilon=[]
        for z1 in range(L):
            if((sol['x'][z1+17]<0.000001)and(getfx(solutions[counter],X[z1])>0.99999)and(getfx(solutions[counter],X[z1])<1.00001)):
                sup_vec.append(trainingSet[z1])
                epsilon.append(sol['x'][z1+17])
        sup_vectors.append(sup_vec)        
        epsilons.append(epsilon)
        
    for x in range(len(testSet)):
        test1=getfx(solutions[0],testSet[x])
        label=0
        for z1 in range(9):
            test2=getfx(solutions[z1+1],testSet[x])
            if(test1<test2):
                test1=test2
                label=z1+1
        predictions.append(label)
    return predictions

# Get a predicted responses(classes) based on the quadratic program solver
# This subroutine solves the transfer svm problem basen on the original svm optimization functions
def getPrediction4(trainingSet,trainingSet2,testSet):
    predictions=[]
    X=[]
    Y=[]
    for x in range(len(trainingSet)):
        X.append(trainingSet[x][0:16])
        Y.append(trainingSet[x][-1])
    # define trade-off C
    C=1.0
    L=len(trainingSet)
    solutions=np.zeros((17),float)
    P=np.zeros((L+17,L+17),float)
    q=np.zeros((L+17,1),float)
    G=np.zeros((L*2,L+17),float)
    h=np.zeros((L*2,1),float)
    for z1 in range(16):
        P[z1][z1]=1
    for z2 in range(L):
        q[z2+17][0]=C
    for z3 in range(L):
        G[z3][z3+17]=-1
        G[z3+L][z3+17]=-1
    for z4 in range(L):
        if(trainingSet[z4][-1]==0):
            y_counter=1
        else:
            y_counter=-1
        for z5 in range(17):
            if(z5==16):
                G[z4][z5]=-y_counter
            else:
                G[z4][z5]=-y_counter*trainingSet[z4][z5]
    for z6 in range(L):
        h[z6][0]=-1
    P=matrix(P)
    q=matrix(q)
    G=matrix(G)
    h=matrix(h)
    sol=solvers.qp(P,q,G,h)
    for z1 in range(17):
        solutions[z1]=sol['x'][z1]
    print solutions
        
    L2=len(trainingSet2)
    solutions2=np.zeros((17),float)
    P2=np.zeros((L2+17,L2+17),float)
    q2=np.zeros((L2+17,1),float)
    G2=np.zeros((L2*2,L2+17),float)
    h2=np.zeros((L2*2,1),float)
    for z1 in range(16):
        P2[z1][z1]=1
    for z2 in range(L2):
        q2[z2+17][0]=C
    for z3 in range(L2):
        G2[z3][z3+17]=-1
        G2[z3+L2][z3+17]=-1
    for z4 in range(L2):
        if(trainingSet2[z4][-1]==0):
            y_counter=1
        else:
            y_counter=-1
        for z5 in range(17):
            if(z5==16):
                G2[z4][z5]=-y_counter
            else:
                G2[z4][z5]=-y_counter*trainingSet2[z4][z5]
    for z6 in range(L2):
        if(trainingSet2[z6][-1]==0):
            y_counter=1
        else:
            y_counter=-1
        fx_cal=getfx(solutions,trainingSet2[z6])
        h2[z6][0]=-1+y_counter*fx_cal
    P2=matrix(P2)
    q2=matrix(q2)
    G2=matrix(G2)
    h2=matrix(h2)
    sol2=solvers.qp(P2,q2,G2,h2)
    for z1 in range(17):
        solutions2[z1]=sol2['x'][z1]
    print solutions2
    
    for x in range(len(testSet)):
        test=getfx(solutions2,testSet[x])+getfx(solutions,testSet[x])
        if(test>0):
            label=0
        else:
            label=8
        predictions.append(label)
    return predictions

# Evaluate the accuracy of the model by calculating a ratio of the total correct
# predictions out of all predictions made, called the classification accuracy
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if (testSet[x][-1]==predictions[x]):
            correct +=1
    return (correct/float(len(testSet)))*100.0

# Define the main function: return 10 cross validation results (with confusion matrix)
def main1():
    f1=open('multiclass svm output','w')
    f2=open('multiclass svm confusion matrix','w')
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
        predictions=getPrediction2(trainingSet,testSet,'poly')
        for x in range(len(testSet)):
            print('> predicted '+repr(predictions[x])+',actual='+repr(testSet[x][-1]))
            f1.write('> predicted '+repr(predictions[x])+',actual='+repr(testSet[x][-1])+'\n')
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
        f2.write('   This is the '+repr(counter)+'th confusion matrix ! \n\n')
        for x in range(10):
            for y in range(10):
                confumatrix_tot[x][y]+=confumatrix[x][y]
                f2.write('   '+repr(confumatrix[x][y]))
            f2.write('\n\n')
    accuracy_overall=accuracy_overall/10.00
    print('Overall Accuracy: '+repr(accuracy_overall)+'%')
    f1.write('Overall Accuracy: '+repr(accuracy_overall)+'%'+'\n\n\n')
    f2.write('      This is the overall confusion matrix ! \n\n')
    print confumatrix_tot
    for x in range(10):
        for y in range(10):
            f2.write('      '+repr(confumatrix_tot[x][y]))
        f2.write('\n\n')

# Define the main function: get the predictions for the test data and also support vectors from traning data
def main2():
    f1=open('svm predict test data','w')
    f2=open('svm support vectors','w')
    f3=open('svm epsilons','w')
    trainingSet=[]
    testSet=[]
    sup_vectors=[]
    epsilons=[]
    loadDataset2('pendigits-train.csv' ,'pendigits-test-nolabels.csv' ,trainingSet ,testSet )
    print 'Train set:'+repr(len(trainingSet))
    print 'Test set:'+repr(len(testSet))
    f1.write('Train set:'+repr(len(trainingSet))+'\n')
    f1.write('Test set:'+repr(len(testSet))+'\n')
    # generate predictions
    predictions=[]
    predictions=getPrediction2(trainingSet,testSet,'poly')
    #predictions=getPrediction3(trainingSet,testSet,sup_vectors,epsilons)
    for x in range(len(testSet)):
        print(repr(predictions[x]))
        f1.write(repr(predictions[x])+'\n')
    f2.write('support vectors from training data \n')
    for x in range(len(sup_vectors)):
        f2.write('This is digit '+repr(x)+'. Its positive support vectors (epsilons=0) are: \n')
        for y in range(len(sup_vectors[x])):
            f2.write(repr(sup_vectors[x][y])+'\n')
        f2.write('there are total '+repr(len(sup_vectors[x]))+' support vectors for this digit\n')
        f2.write('\n\n\n\n')
    f3.write('epsilons from training data \n')
    for x in range(len(epsilons)):
        f3.write('This is digit '+repr(x)+'. Its epsilons are: \n')
        for y in range(len(epsilons[x])):
            f3.write(repr(epsilons[x][y])+'\n')
        f3.write('there are total '+repr(len(epsilons[x]))+' zero epsilons for this digit\n')
        f3.write('\n\n\n\n')
    
# Define the main function: return 10 fold cross validation results for the transfer target data set(with confusion matrix)
def main3():
    f1=open('transfer svm output','w')
    f2=open('transfer svm confusion matrix','w')
    confumatrix_tot=np.zeros((10,10),int)
    accuracy_overall=0
    for counter in range(10):
        trainingSet=[]
        trainingSet2=[]
        testSet=[]
        loadDataset3('0vs8Source.csv', '0vs8Target.csv' ,trainingSet ,trainingSet2 ,counter, testSet )
        print 'Counter='+repr(counter)
        print 'Train set in target file:'+repr(len(trainingSet2))
        print 'Test set:'+repr(len(testSet))
        f1.write('Counter='+repr(counter)+'\n')
        f1.write('Train set in target file:'+repr(len(trainingSet2))+'\n')
        f1.write('Test set:'+repr(len(testSet))+'\n')
        # generate predictions
        predictions=[]
        predictions=getPrediction4(trainingSet,trainingSet2,testSet)
        for x in range(len(testSet)):
            print('> predicted '+repr(predictions[x])+',actual='+repr(testSet[x][-1]))
            f1.write('> predicted '+repr(predictions[x])+',actual='+repr(testSet[x][-1])+'\n')
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
        f2.write('   This is the '+repr(counter)+'th confusion matrix ! \n\n')
        for x in range(10):
            for y in range(10):
                confumatrix_tot[x][y]+=confumatrix[x][y]
                f2.write('   '+repr(confumatrix[x][y]))
            f2.write('\n\n')
    accuracy_overall=accuracy_overall/10.00
    print('Overall Accuracy: '+repr(accuracy_overall)+'%')
    f1.write('Overall Accuracy: '+repr(accuracy_overall)+'%'+'\n\n\n')
    f2.write('      This is the overall confusion matrix ! \n\n')
    print confumatrix_tot
    for x in range(10):
        for y in range(10):
            f2.write('   '+repr(confumatrix_tot[x][y]))
        f2.write('\n\n')

# Define the main function: get the predictions for the test data using transfer svm
def main4():
    f1=open('transfer svm predict test data','w')
    trainingSet=[]
    trainingSet2=[]
    testSet=[]
    loadDataset4('0vs8Source.csv', '0vs8Target.csv','0vs8TestNoLabels.csv' ,trainingSet ,trainingSet2, testSet )
    print 'Source Train Set:'+repr(len(trainingSet))
    print 'Target Train Set:'+repr(len(trainingSet2))
    print 'Test Set:'+repr(len(testSet))
    f1.write('Source Train Set:'+repr(len(trainingSet))+'\n')
    f1.write('Target Train Set:'+repr(len(trainingSet2))+'\n')
    f1.write('Test Set:'+repr(len(testSet))+'\n')
    # generate predictions
    predictions=[]
    predictions=getPrediction4(trainingSet,trainingSet2,testSet)
    for x in range(len(testSet)):
        print(repr(predictions[x]))
        f1.write(repr(predictions[x])+'\n')

# I want to know the predictions of the test data using transfer svm, so I call main4()
main4()


