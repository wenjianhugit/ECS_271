import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import locally_linear_embedding
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D

def load_images():
    poses=[]
    poses.append('left')
    poses.append('right')
    poses.append('straight')
    poses.append('up')
    
    expressions=[]
    expressions.append('angry')
    expressions.append('happy')
    expressions.append('neutral')
    expressions.append('sad')
    
    sunglasses=[]
    sunglasses.append('open')
    sunglasses.append('sunglasses')
    
    names=[]
    names.append('an2i')
    names.append('at33')
    names.append('boland')
    names.append('bpm')
    names.append('ch4f')
    
    names.append('cheyer')
    names.append('choon')
    names.append('danieln')
    names.append('glickman')
    names.append('karyadi')
    
    names.append('kawamura')
    names.append('kk49')
    names.append('megak')
    names.append('mitchell')
    names.append('night')
    
    names.append('phoebe')
    names.append('saavik')
    names.append('steffi')
    names.append('sz24')
    names.append('tammo')
    
    images=[]
    colors=[]
    for x in range(len(names)):
        for y in range(len(poses)):
            for z in range(len(expressions)):
                for w in range(len(sunglasses)):
                    full_name=names[x]+'_'+poses[y]+'_'+expressions[z]+'_'+sunglasses[w]+'_2.pgm'
                    judge1=(full_name != 'choon_left_angry_sunglasses_2.pgm')
                    judge2=(full_name != 'danieln_straight_happy_sunglasses_2.pgm')
                    judge3=(full_name != 'glickman_right_sad_open_2.pgm')
                    judge4=(full_name != 'glickman_straight_angry_open_2.pgm')
                    judge5=(full_name != 'karyadi_right_happy_sunglasses_2.pgm')
                    judge6=(full_name != 'kawamura_up_happy_open_2.pgm')
                    judge7=(full_name != 'megak_straight_sad_sunglasses_2.pgm')
                    judge8=(full_name != 'megak_up_sad_sunglasses_2.pgm')
                    judge9=(full_name != 'mitchell_right_angry_sunglasses_2.pgm')
                    judge10=(full_name != 'mitchell_straight_angry_open_2.pgm')
                    judge11=(full_name != 'mitchell_up_happy_sunglasses_2.pgm')
                    judge12=(full_name != 'mitchell_up_neutral_sunglasses_2.pgm')
                    judge13=(full_name != 'phoebe_left_angry_open_2.pgm')
                    judge14=(full_name != 'sz24_left_neutral_sunglasses_2.pgm')
                    judge15=(full_name != 'tammo_right_happy_open_2.pgm')
                    judge16=(full_name != 'tammo_right_sad_open_2.pgm')
                    #                    judge17=(full_name != 'ch4f_left_sad_open_2.pgm')
                    judge_overall=judge1 and judge2 and judge3 and judge4 and judge5 and judge6 and judge7 and judge8 and judge9 and judge10 and judge11 and judge12 and judge13 and judge14 and judge15 and judge16
                    if (judge_overall):
                        face = misc.imread(full_name)
                        images.append(face)
                        temp_value=x*5
                        colors.append(temp_value)
    return images,colors

def matrix_build(images):
    L=len(images)
    matrix_images=np.zeros((L,3840),int)
    for x in range(L):
        for y in range(60):
            for z in range(64):
                matrix_images[x][y*64+z]=images[x][y][z]
    return matrix_images

def euclideanDistance(instance1, instance2):
    distance=0
    length=len(instance1)
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

def getNeighbors(dataSet, testID, k):
    distances=[]
    for x in range(len(dataSet)):
        dist=euclideanDistance(dataSet[testID], dataSet[x])
        distances.append((x,dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k+1):
        if(distances[x][0]!=testID):
            neighbors.append(distances[x][0])
    return neighbors

def getG_matrix(matrix_images,neighbors,x):
    L=len(matrix_images)
    kk=len(neighbors)
    Lx=len(matrix_images[0])
    G_matrix=np.zeros((kk,kk),float)
    for j in range(kk):
        for k in range(kk):
            if(j>=k):
                x_left=np.zeros(Lx,float)
                x_right=np.zeros(Lx,float)
                for count in range(Lx):
                    x_left[count]=matrix_images[x][count]-matrix_images[neighbors[j]][count]
                    x_right[count]=matrix_images[x][count]-matrix_images[neighbors[k]][count]
                for count in range(Lx):
                    G_matrix[j][k]+=x_left[count]*x_right[count]
    for j in range(kk):
        for k in range(kk):
            if(j<k):
                G_matrix[j][k]=G_matrix[k][j]
    return G_matrix

def main_PCA():
    f1=open('largest eigenvalues','w')
    f2=open('projection vectors 1','w')
    f3=open('projection vectors 2','w')
    f4=open('projection vectors 3','w')
    f5=open('scatter matrix in 3D','w')
    images=[]
    images,colors=load_images()
    matrix_images=matrix_build(images)
    cov_matrix=np.cov(matrix_images.T)
    v,w=np.linalg.eig(cov_matrix)
    v=v.real
    w=w.real
    ev_list = zip(v,w)
    ev_list.sort(key=lambda tup:tup[0], reverse=True)
    v,w= zip(*ev_list)
    for x in range(len(v)):
        if(x<3):
            f1.write(repr(v[x])+'\n')
    for x in range(len(w[0])):
        f2.write(repr(w[0][x])+'\n')
    for x in range(len(w[1])):
        f3.write(repr(w[1][x])+'\n')
    for x in range(len(w[2])):
        f4.write(repr(w[2][x])+'\n')
    pca = PCA(n_components=3)
    newmatrix=pca.fit_transform(matrix_images)
    xx=[]
    yy=[]
    zz=[]
    for x in range(len(newmatrix)):
        xx.append(newmatrix[x][0])
        yy.append(newmatrix[x][1])
        zz.append(newmatrix[x][2])
        f5.write(repr(newmatrix[x][0])+','+repr(newmatrix[x][1])+','+repr(newmatrix[x][2])+'\n')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xx,yy,zz,c=colors)
    plt.show()

def main_kernelPCA():
    f6=open('scatter matrix in 3D','w')
    images=[]
    images,colors=load_images()
    matrix_images=matrix_build(images)
    newmatrix=KernelPCA(n_components=3, kernel='rbf',degree=4,gamma=10.0).fit_transform(matrix_images)
    print 'len(newdata)='+repr(len(newmatrix))
    xx=[]
    yy=[]
    zz=[]
    for x in range(len(newmatrix)):
        xx.append(newmatrix[x][0])
        yy.append(newmatrix[x][1])
        zz.append(newmatrix[x][2])
        f6.write(repr(newmatrix[x][0])+','+repr(newmatrix[x][1])+','+repr(newmatrix[x][2])+'\n')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xx,yy,zz,c=colors)
    plt.show()

def main_LLE():
    f1=open('P smallest eigenvalues','w')
    f2=open('P bottom eigenvector','w')
    f3=open('P projection vectors 1','w')
    f4=open('P projection vectors 2','w')
    f5=open('P projection vectors 3','w')
    f6=open('M smallest eigenvalues','w')
    f7=open('M bottom eigenvector','w')
    f8=open('M projection vectors 1','w')
    f9=open('M projection vectors 2','w')
    f10=open('M projection vectors 3','w')
    k=40
    images=[]
    images,colors=load_images()
    matrix_images=matrix_build(images)
    L=len(matrix_images)
    W_matrix=np.zeros((L,L),float)
    for x in range(L):
        neighbors=getNeighbors(matrix_images,x,k)
        G_matrix=getG_matrix(matrix_images,neighbors,x)
        num_count=0
        for i in range(k):
            for j in range(k):
                if(G_matrix[i][j]>0.001 or G_matrix[i][j]<-0.001):
                    num_count+=1
        print 'num_count='+repr(num_count)
        print 'x='+repr(x)
        print 'det='+repr(np.linalg.det(G_matrix))
        G_matrix_inv=np.linalg.inv(G_matrix)
        G_inv_all_sum=0
        for i in range(k):
            for j in range(k):
                G_inv_all_sum+=G_matrix_inv[i][j]
        for i in range(k):
            for j in range(k):
                W_matrix[x][neighbors[i]]+=G_matrix_inv[i][j]
            W_matrix[x][neighbors[i]]=W_matrix[x][neighbors[i]]/(G_inv_all_sum*1.00)
    X_matrix=matrix_images.T
    I=np.identity(L)
    Eigen_matrix=np.dot(np.subtract(I,W_matrix),X_matrix.T)
    Eigen_matrix=np.dot((np.subtract(I,W_matrix)).T,Eigen_matrix)
    Eigen_matrix=np.dot(X_matrix,Eigen_matrix)
    XX=np.linalg.inv(np.dot(X_matrix,X_matrix.T))
    Eigen_matrix=np.dot(XX,Eigen_matrix)
    M_matrix=np.dot((np.subtract(I,W_matrix)).T,np.subtract(I,W_matrix))
    v,w=np.linalg.eig(Eigen_matrix)
    v=v.real
    w=w.real
    ev_list = zip(v,w)
    ev_list.sort(key=lambda tup:tup[0], reverse=False)
    v,w= zip(*ev_list)
    for x in range(len(v)):
        if(x<4):
            f1.write(repr(v[x])+'\n')
    for x in range(len(w[0])):
        f2.write(repr(w[0][x])+'\n')
    for x in range(len(w[1])):
        f3.write(repr(w[1][x])+'\n')
    for x in range(len(w[2])):
        f4.write(repr(w[2][x])+'\n')
    for x in range(len(w[3])):
        f5.write(repr(w[3][x])+'\n')
    v,w=np.linalg.eig(M_matrix)
    v=v.real
    w=w.real
    ev_list = zip(v,w)
    ev_list.sort(key=lambda tup:tup[0], reverse=False)
    v,w= zip(*ev_list)
    for x in range(len(v)):
        if(x<4):
            f6.write(repr(v[x])+'\n')
    for x in range(len(w[0])):
        f7.write(repr(w[0][x])+'\n')
    for x in range(len(w[1])):
        f8.write(repr(w[1][x])+'\n')
    for x in range(len(w[2])):
        f9.write(repr(w[2][x])+'\n')
    for x in range(len(w[3])):
        f10.write(repr(w[3][x])+'\n')
    xx=w[1]
    yy=w[2]
    zz=w[3]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xx,yy,zz,c=colors)
    plt.show()

def main_LLE2():
    f3=open('projection vectors 1','w')
    f4=open('projection vectors 2','w')
    f5=open('projection vectors 3','w')
    f6=open('scatter matrix in 3D','w')
    k=40
    images=[]
    images,colors=load_images()
    matrix_images=matrix_build(images)
    newmatrix,squared_error=locally_linear_embedding(matrix_images,k,3,eigen_solver='auto')
    print 'squared_error='+repr(squared_error)
    print newmatrix.shape
    M1=np.dot(newmatrix.T,matrix_images)
    M2=np.linalg.inv(np.dot(matrix_images.T,matrix_images))
    P_matrix=(np.dot(M1,M2)).T
    print P_matrix.shape
    for x in range(len(P_matrix)):
        f3.write(repr(P_matrix[x][0])+'\n')
        f4.write(repr(P_matrix[x][1])+'\n')
        f5.write(repr(P_matrix[x][2])+'\n')
    xx=[]
    yy=[]
    zz=[]
    for x in range(len(newmatrix)):
        xx.append(newmatrix[x][0])
        yy.append(newmatrix[x][1])
        zz.append(newmatrix[x][2])
        f6.write(repr(newmatrix[x][0])+','+repr(newmatrix[x][1])+','+repr(newmatrix[x][2])+'\n')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xx,yy,zz,c=colors)
    plt.show()

main_kernelPCA()

