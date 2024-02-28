import numpy as np
import scipy as sp
import cmath as cm
import math as m
from scipy.sparse import csr_matrix 
from scipy.stats import unitary_group
from matplotlib import pyplot as plt

########### Generation of Gates (PPU), extraction of parameters, conversion to Gaussian tensor #########


# Generate a parity preserving unitary (PPU) from two matrices A and B, the matrices should be unitary for the result to be unitary

def constructPPU(A,B):
    U = np.array([[A[0][0],0,0,A[0][1]],[0,B[0,0],B[0,1],0],[0,B[1,0],B[1,1],0],[A[1,0],0,0,A[1,1]]])    
    return U


# These functions extract important parameters of PPUs, the entangling power EPower, the non0Gaussianity Gamma2 defined as the
# difference between the determinants of A and B, the 'normalization' N and the rescaled non-Guassianity Gamma

def getEPower(U):
    p00=cm.phase(U[0][0])
    p33=cm.phase(U[3][3])
    p11=cm.phase(U[1][1])
    p22=cm.phase(U[2][2])
    c=(p00+p33-(p11+p22))/4
    return m.sin(2*c)**2 

def getA(U):
    A=[[U[0,0],U[0,3]],[U[3,0],U[3,3]]]
    return A

def getB(U):
    B=[[U[1,1],U[1,2]],[U[2,1],U[2,2]]]
    return B

def getGamma2fromAB(A,B):
    return np.linalg.det(A)-np.linalg.det(B)

def getGamma2(U):
    A=getA(U)
    B=getB(U)
    return getGamma2fromAB(A,B)

def getN(U):
    return U[0,0]

def getGamma(U):
    return getGamma2(U)/getN(U)**2

# Generating the triple (N,A,gamma,gamma2) from a PPU

def getAmat(a,b,N):
    A=[[0,a[0][1]/N,b[0][1]/N,b[1][1]/N],[0,0,b[0][0]/N,b[1][0]/N],[0,0,0,a[1][0]/N],[0,0,0,0]]
    A=np.transpose(A)-A
    return np.array(A)    

def getNAGamma(U):
    a=getA(U)
    b=getB(U)
    N=getN(U)
    A=np.array(getAmat(a,b,N))
    gamma=getGamma2fromAB(a,b)/N**2
    gamma2=getGamma2fromAB(a,b)
    return [N,A,gamma,gamma2]

# Generate Haar random MG
def genRMG():
    A = unitary_group.rvs(2)
    A = A/np.sqrt(np.linalg.det(A))
    B = unitary_group.rvs(2)
    B = B/np.sqrt(np.linalg.det(B))
    return constructPPU(A,B)

########## Lattice ##########

# tuple to position
def TTP(t):
    return t[0]*4+t[1]

# all connections sorted by left right even odd rows
def ConRightEven(d,l):
    list = []
    for r in range(0,d):
        for c in range(0,l-1):
            x=(2*l-1)*r+c
            nonzeroentry = [[x,2],[x+l,0]]
            list.append(nonzeroentry)
    return list

def ConRightOdd(d,l):
    list = []
    for r in range(0,d):
        for c in range(0,l-1):
            x=l+(2*l-1)*r+c
            nonzeroentry = [[x,2],[x+l,0]]
            list.append(nonzeroentry)
    return list

def ConLeftEven(d,l):
    list = []
    for r in range(0,d):
        for c in range(1,l):
            x=(2*l-1)*r+c
            nonzeroentry = [[x,3],[x+l-1,1]]
            list.append(nonzeroentry)
    return list

def ConLeftOdd(d,l):
    list = []
    for r in range(0,d):
        for c in range(0,l-1):
            x=l+(2*l-1)*r+c
            nonzeroentry = [[x,3],[x+l-1,1]]
            list.append(nonzeroentry)
    return list

#list of all connections, human readable
def GetConnections(d,l):
    return ConRightEven(d,l)+ConRightOdd(d,l)+ConLeftEven(d,l)+ConLeftOdd(d,l)


#lists of non-zero rows and columns to be feeded into sparse matrix format
def GetNonZeroRows(d,l):
    return list(map(TTP,list(map(lambda x: x[0],GetConnections(d,l)))))

def GetNonZeroColumns(d,l):
    return list(map(TTP,list(map(lambda x: x[1],GetConnections(d,l)))))

print(GetConnections(2,2))      
print(GetNonZeroRows(2,2))  
print(GetNonZeroColumns(2,2))  



# row = np.array([0, 0, 1, 1, 2, 1]) 
# col = np.array([0, 1, 2, 0, 2, 2]) 
  
# # taking data 
# data = np.array([1, 4, 5, 8, 9, 6]) 
  
# # creating sparse matrix 
# sparseMatrix = csr_matrix((data, (row, col)), shape = (3, 3)).toarray() 
  
# # print the sparse matrix 
# print(sparseMatrix)
















########## Statistics ##########

# This function returns a list of the parameters above for haar random PPUs.

def GetUStatitics(samples):
    list=[]
    for x in range(samples):
        A = unitary_group.rvs(2)
        B = unitary_group.rvs(2)
        U = constructPPU(A,B)
        entry = [getEPower(U),getGamma2(U),getN(U),getGamma(U)]
        list.append(entry)
    return(list)    





########## Some evaluation ##########

# The functions below were used for some preliminary statistcal analysis of haar random PPUs
# s=100000
# testlist=GetUStatitics(s)
# e = list(map(lambda x: x[0],testlist))
# g = list(map(lambda x: x[1],testlist))
# print(np.average(e))
# print(data)
# fixed bin size
# testbins = np.arange(0, 2, 0.01) # fixed bin size
# print(sum(i < 0.5 for i in g))
# print(sum(i > 0.5 for i in g))

# plt.hist2d(g,e, bins=(200, 200), cmap=plt.cm.Reds)
# plt.colorbar()
# plt.show()
# plt.hist(g, bins=testbins, alpha=0.5, log=False)
# plt.show()

# plt.xlim([min(data), max(data)])
# plt.title('Data (fixed bin size)')
# plt.xlabel('variable X')
# plt.ylabel('count')

# plt.show()
    

