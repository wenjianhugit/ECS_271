import csv
import random
import math
import operator
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy
import mpmath
from sympy import Matrix,pretty

# load data and store them in the trainingSet list.
def loadDataset1(filename, trainingSet):
    with open(filename, 'rU') as csvfile: 
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            if(x<100):
                gotoint=dataset[x][0].split("\t")
                gotoint2=[]
                for y in range(len(gotoint)):
                    gotoint2.append(int(gotoint[y]))
                trainingSet.append(gotoint2)

# Build up the original graph using trainingSet data
def Graph_build(trainingSet):
    G=nx.Graph()
    for x in range(len(trainingSet)):
        G.add_edge(trainingSet[x][0],trainingSet[x][1])
    return G

# Build up the graph for each connected component
def new_Graph_build(G,cluster):
    G_new=nx.Graph()
    for x in range(len(G.edges())):
        node1=G.edges()[x][0]
        node2=G.edges()[x][1]
        if ((node1 in cluster)and(node2 in cluster)):
            G_new.add_edge(node1,node2)
    return G_new

# Get graph information such as A, B, Vol(A), Vol(B), c, according to the initial partition (MQI)
def Size_Graph(G,parts):
    Vol0=0
    Vol1=0
    G0=[]
    G1=[]
    for x in range(len(G.nodes())):
        node_x=G.nodes()[x]
        G.node[node_x]['cluster']=parts[x]
        if (parts[x]==0):
            Vol0+=G.degree(node_x)
            G0.append(node_x)
        elif (parts[x]==1):
            Vol1+=G.degree(node_x)
            G1.append(node_x)
    if (Vol0>=Vol1):
        A=G1
        B=G0
        VolA=Vol1
        VolB=Vol0
    else:
        A=G0
        B=G1
        VolA=Vol0
        VolB=Vol1
    c=0
    for x in range(len(G.edges())):
        node0=G.edges()[x][0]
        node1=G.edges()[x][1]
        if(G.node[node0]['cluster']!=G.node[node1]['cluster']):
            c+=1
    return A,B,VolA,VolB,c

# Build up the direct graph following MQI algorithm (MQI)
def Build_Direct_Graph(G,A,B,VolA,VolB,c):
    H=nx.DiGraph()
    s=-1
    t=-2
    for x in range(len(G.edges())):
        node0=G.edges()[x][0]
        node1=G.edges()[x][1]
        if((node0 in A)and(node1 in A)):
            H.add_edge(node0,node1)
            H.add_edge(node1,node0)
            H.edge[node0][node1]['capacity']=VolA
            H.edge[node1][node0]['capacity']=VolA
        elif ((node0 in A)and(node1 in B)):
            H.add_edge(s,node0)
            if (len(H.edge[s][node0])==0):
                H.edge[s][node0]['capacity']=VolA
            elif (len(H.edge[s][node0])==1):
                H.edge[s][node0]['capacity']+=VolA
        elif ((node0 in B)and(node1 in A)):
            H.add_edge(s,node1)
            if (len(H.edge[s][node1])==0):
                H.edge[s][node1]['capacity']=VolA
            elif (len(H.edge[s][node1])==1):
                H.edge[s][node1]['capacity']+=VolA
    for x in range(len(A)):
        H.add_edge(A[x],t)
        H.edge[A[x]][t]['capacity']=c
    return H,s,t

# Get a new partition (A', bar{A'}) with smaller conductance (MQI)
def Build_A_new(HH,s,t,flow_dict):
    remove_edges=[]
    for x in range(len(HH.edges())):
        node0=HH.edges()[x][0]
        node1=HH.edges()[x][1]
        if (HH.edge[node0][node1]['capacity']==flow_dict[node0][node1]):
            remove_edges.append((node0,node1))
    for x in range(len(remove_edges)):
        HH.remove_edge(remove_edges[x][0],remove_edges[x][1])
    A_new=list(nx.dfs_preorder_nodes(HH,t))
    if (t in A_new):
        A_new.remove(t)
    if (s in A_new):
        A_new.remove(s)
    return A_new

# MQI core algorithm (MQI)
def MQI(G,A,B,VolA,VolB,c):
    H,s,t=Build_Direct_Graph(G,A,B,VolA,VolB,c)
    if (c!=0):
        HH=nx.DiGraph.reverse(H)
        flow_value,flow_dict=nx.maximum_flow(HH,t,s,capacity='capacity')
        print 'flow_value='+repr(flow_value)
        if (flow_value < c*len(A)):
            change=1
            A_new=Build_A_new(HH,s,t,flow_dict)
            B_new=[]
            VolA_new=0
            VolB_new=0
            for x in range(len(G.nodes())):
                if (G.nodes()[x] in A_new):
                    VolA_new+=G.degree(G.nodes()[x])
                else:
                    B_new.append(G.nodes()[x])
                    VolB_new+=G.degree(G.nodes()[x])
            c_new=0
            for x in range(len(G.edges())):
                node0=G.edges()[x][0]
                node1=G.edges()[x][1]
                if (((node0 in A_new)and(node1 in B_new))or((node1 in A_new)and(node0 in B_new))):
                    c_new+=1
        else:
            change=0
            A_new=A
            B_new=B
            VolA_new=VolA
            VolB_new=VolB
            c_new=c
    else:
        change=0
        A_new=A
        B_new=B
        VolA_new=VolA
        VolB_new=VolB
        c_new=c
    conductance_new=c_new/(VolA_new*1.00)
    return A_new,B_new,VolA_new,VolB_new,c_new,change,conductance_new

# Build up a quick search between node index and matrix index (Spectral Clustering)
def Node_num_and_matrix_num(G):
    largest_node_num=G.nodes()[0]
    for x in range(len(G.nodes())):
        if (largest_node_num<G.nodes()[x]):
            largest_node_num=G.nodes()[x]
    N_to_M=np.zeros((largest_node_num+1),int)
    M_to_N=[]
    for x in range(len(G.nodes())):
        N_to_M[G.nodes()[x]]=x
        M_to_N.append(G.nodes()[x])
    return N_to_M, M_to_N

# Build up the adjacency matrix (Spectral Clustering)
def similarity_cal(G,N_to_M):
    L=len(G.nodes())
    similarity_matrix=np.zeros((L,L),int)
    for x in range(len(G.edges())):
            node1=G.edges()[x][0]
            node2=G.edges()[x][1]
            M_num1=N_to_M[node1]
            M_num2=N_to_M[node2]
            similarity_matrix[M_num1][M_num2]=1
            similarity_matrix[M_num2][M_num1]=1
    return similarity_matrix            

# Build up the degree matrix (Spectral Clustering)
def degree_cal(similarity_matrix):
    L=len(similarity_matrix)
    degree_matrix=np.zeros((L,L),int)
    for x in range(L):
        sum=0
        for y in range(L):
            sum+=similarity_matrix[x][y]
        degree_matrix[x][x]=sum
    return degree_matrix

# Build up the Laplacian matrix (Spectral Clustering)
def L_cal(degree_matrix, similarity_matrix):
    L=len(similarity_matrix)
    L_matrix=np.zeros((L,L),int)
    for x in range(L):
        for y in range(L):
            L_matrix[x][y]=degree_matrix[x][y]-similarity_matrix[x][y]
    return L_matrix

def get_Y(w,L,k):
    Y=[]
    for x in range(L):
        YY=[]
        for y in range(k):
            YY.append(w[y][x])
        Y.append(YY)
    return Y

# Get eigenvalues of Laplacian matrix and store in v (Spectral Clustering)
def get_eigen(L_matrix):
    v,w=scipy.linalg.eigh(L_matrix)
    v=v.real
    w=w.real
    ev_list = zip(v,w)
    ev_list.sort(key=lambda tup:tup[0], reverse=False)
    v,w= zip(*ev_list)
    return v,w

# Get eigenvectors of Laplacian matrix and store in w_new (Spectral Clustering)
def get_new_eigenvectors(MM,w):
    w_new=w
    for x in range(len(MM)):
        if(MM[x][0]==0):
            for y in range(MM[x][1]):
                for z in range(len(MM[x][2][y])):
                    w_new[y][z]=MM[x][2][y][z]
                get_normed(w_new[y])
    return w_new

# Find out the number of connected components using the multiplicity k of the eigenvalue 0 of L (Spectral Clustering)
def get_k(MM):
    for x in range(len(MM)):
        if(MM[x][0]==0):
            k=MM[x][1]
    return k

# Normalize vectors (Spectral Clustering)
def get_normed(vector):
    sum=0
    for x in range(len(vector)):
        sum+=vector[x]**2
    sum_sqrt=math.sqrt(sum)
    for x in range(len(vector)):
        vector[x]=vector[x]/(sum_sqrt*1.00)

# Find out corresponding connected components (Spectral Clustering)
def clusters_cal(k,Y_pred,M_to_N):
    clusterList=[]
    for x in range(k):
        clu_list=[]
        for y in range(len(Y_pred)):
            if(Y_pred[y]==x):
                clu_list.append(M_to_N[y])
        clusterList.append(clu_list)
    return clusterList

# Main MQI code (MQI)
def random_mqi(G,x):
    fp1=open(repr(x)+'th random cluster size','w')
    fp2=open(repr(x)+'th random cut conductance','w')
    fp3=open(repr(x)+'th random best A','w')
    fp4=open(repr(x)+'th cluster members','w')
    print 'G_nodes='+repr(len(G.nodes()))
    fp4.write(repr(G.nodes())+'\n')
    conductance_list=np.ones(len(G.nodes())+1,float)
    A_best=[]
    for x in range(len(G.nodes())+1):
        A_best.append([])
    for x in range(len(conductance_list)):
        conductance_list[x]=-conductance_list[x]
    i=0
    while i<50000:
        i=i+1
        print repr(i)+'th'
        parts=[]
        for x in range(len(G.nodes())):
            parts.append(random.randint(0,1))
        if(sum(parts)<0.1):
            parts[0]=1
        if(sum(parts)>len(parts)-0.1):
            parts[0]=0
        A,B,VolA,VolB,c=Size_Graph(G,parts)
        conductance=c/(VolA*1.00)
        print 'conductance='+repr(conductance)
        if(((conductance_list[len(A)]>0)and(conductance_list[len(A)]>conductance))or(conductance_list[len(A)]<0)):
            conductance_list[len(A)]=conductance
            A_best[len(A)]=A
        change=1
        while change==1:
            A,B,VolA,VolB,c,change,conductance=MQI(G,A,B,VolA,VolB,c)
            print 'conductance='+repr(conductance)
            if(((conductance_list[len(A)]>0)and(conductance_list[len(A)]>conductance))or(conductance_list[len(A)]<0)):
                conductance_list[len(A)]=conductance
                A_best[len(A)]=A
    for x in range(len(conductance_list)):
        if(conductance_list[x]>-0.01):
            fp1.write(repr(x)+'\n')
            fp2.write(repr(conductance_list[x])+'\n')
            fp3.write('len(A)='+repr(x)+':\n'+repr(A_best[x])+'\n\n\n')

# Main code (Spectral clustering + MQI)
def main():
    fp=open('clusters','w') 
    fp2=open('eigenvalues and best cluster number k','w')
    fp3=open('cluster sizes','w')
    trainingSet=[]
    loadDataset1('roadNet-PA.csv',trainingSet)
    G=Graph_build(trainingSet)
    print len(G.nodes())
    N_to_M, M_to_N=Node_num_and_matrix_num(G)
    similarity_matrix=similarity_cal(G,N_to_M)
    degree_matrix=degree_cal(similarity_matrix)
    L_matrix=L_cal(degree_matrix,similarity_matrix)
    M=Matrix(L_matrix)
    print 'start eigenvectors'
    MM=M.eigenvects()
    print 'end eigenvectors'
    v,w=get_eigen(L_matrix)
    L=len(G.nodes())
    k=get_k(MM)
    w=get_new_eigenvectors(MM,w)
    for x in range(len(v)):
        fp2.write(repr(v[x])+'\n')
    Y=get_Y(w,L,k)
    Y_pred=KMeans(n_clusters=k, max_iter=600).fit_predict(Y)
    clusterList=clusters_cal(k,Y_pred,M_to_N)
    for x in range(len(clusterList)):
        fp.write('This is '+repr(x)+'th cluster and it has members:\n')
        fp.write(repr(clusterList[x])+'\n\n\n\n')
        fp3.write(repr(len(clusterList[x]))+'\n')
    for x in range(len(clusterList)):
        G_new=new_Graph_build(G,clusterList[x])
        random_mqi(G_new,x)

# call main()
main()