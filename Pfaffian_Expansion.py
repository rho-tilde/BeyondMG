# TN Contraction via Pfaffian sums - 25.4.2025, C. Wille

# Using https://pypi.org/project/pfapack/


import numpy as np
import torch
import copy
import sys
import scipy as sp
import cmath as cm
import math as m
import time
import pandas as pd
import random
import os.path
import multiprocessing
import functools




from pfapack.ctypes import pfaffian as cpf
from itertools import combinations
from functools import reduce
from scipy.sparse import kron, csr_matrix, csc_matrix 
from scipy.sparse import coo_matrix
from scipy.stats import unitary_group
from matplotlib import pyplot as plt
from scipy.stats import sem
from scipy.sparse.linalg import splu
from scipy import linalg
from scipy.special import binom, comb



########## PRE FUNCTIONS ##########

## product of a list
def listproduct(list):
    return reduce(lambda x, y: x * y, list)

# Gerenate k-tuples outputs data format tuples
def generate_k_tuples(lst, k):
    return combinations(lst, k)

########## Functions to convert PPUs U=G(A,B) ##########


def getA(U):
    U=np.array(U)
    # U=np.array(U.todense())
    Aa=[[U[0,0],U[0,3]],[U[3,0],U[3,3]]]
    return Aa

def getB(U):
    U=np.array(U)
    Bb=[[U[1,1],U[1,2]],[U[2,1],U[2,2]]]
    return Bb

def getUtensor(U):
    a=getA(U)
    b=getB(U)

    utens=torch.zeros((2,2,2,2),dtype=torch.cdouble)

    utens[0,0,0,0]=a[0][0]
    utens[0,0,1,1]=a[0][1]
    utens[1,1,0,0]=a[1][0]
    utens[1,1,1,1]=a[1][1]

    utens[0,1,0,1]=b[0][0]
    utens[1,0,0,1]=b[1][0]
    utens[0,1,1,0]=b[0][1]
    utens[1,0,1,0]=b[1][1]
    return utens

def getGamma2fromAB(A,B):
    diff = np.linalg.det(A)-np.linalg.det(B)   
    if np.abs(diff)<1.0e-12:
        diff=0.0   
    return diff

def getGamma2(U):
    Aa=getA(U)
    Bb=getB(U)
    return getGamma2fromAB(Aa,Bb)

def getN(U):
    U=np.array(U)
    return U[0,0]

def getGamma(U):
    return getGamma2(U)/getN(U)**2


# Generating the triple (N,A,gamma,gamma2) from a PPU

def getAmat(a,b,N):
    Aa=[[0,a[0][1]/N,b[0][1]/N,b[1][1]/N],[0,0,b[0][0]/N,b[1][0]/N],[0,0,0,a[1][0]/N],[0,0,0,0]]
    Aa=Aa-np.transpose(Aa)
    return np.array(Aa)    

def getNAGamma(U):
    a=getA(U)
    b=getB(U)
    Nn=getN(U)
    Aa=np.array(getAmat(a,b,Nn))
    gamma=getGamma2fromAB(a,b)/(Nn**2)
    gamma2=getGamma2fromAB(a,b)
    return [Nn,Aa,gamma,gamma2]


########## Special Gates and random gates ##########

Id=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

def CPhase(phi):
    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,np.exp(phi*1.0j)]]

def agate(phi):
    return [[np.exp(-phi*1.0j/4),0,0,0],[0,np.exp(phi*1.0j/4),0,0],[0,0,np.exp(phi*1.0j/4),0],[0,0,0,np.exp(3*phi*1.0j/4)]]

def bgate(phi):
    return [[np.exp(-phi*1.0j/4),0,0,0],[0,-np.exp(phi*1.0j/4),0,0],[0,0,-np.exp(phi*1.0j/4),0],[0,0,0,np.exp(3*phi*1.0j/4)]]

def FSim(theta,phi):
    return [[1,0,0,0],[0,np.cos(theta),-1.0j*np.sin(theta),0],[0,-1.0j*np.sin(theta),np.cos(theta),0],[0,0,0,np.exp(-1j*phi)]]

def constructPPU(A,B):
    U = np.array([[A[0][0],0,0,A[0][1]],[0,B[0,0],B[0,1],0],[0,B[1,0],B[1,1],0],[A[1,0],0,0,A[1,1]]])    
    return U

def MakeUnitDet(A):
    det = np.linalg.det(A)
    n = 1/np.sqrt(det)
    return n*A

def RandomMG():
    A = unitary_group.rvs(2)
    A = MakeUnitDet(A)
    B = unitary_group.rvs(2)
    B = MakeUnitDet(B)
    return(constructPPU(A,B))

def RandomPPU():
    A = unitary_group.rvs(2)
    B = unitary_group.rvs(2)
    U = constructPPU(A,B)
    return U

########## Lattice ##########

# tuple to position
def TTP(t):
    return t[0]*4+t[1]

# all connections sorted by left right even odd rows and additional boundary connections
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

def ConBoundary(d,l):
    list = []
    for r in range(0,d):
        x=r*(2*l-1)
        y=(r+1)*(2*l-1)
        nonzeroentryleft = [[x,3],[y,0]]
        list.append(nonzeroentryleft)
        x=r*(2*l-1)+l-1
        y=(r+1)*(2*l-1)+l-1
        nonzeroentryright = [[x,2],[y,1]]
        list.append(nonzeroentryright)
    return list

# dimension of the matrix for number of qubits=2l, number of layers= 2d+1  
# this corresponds to l pairs of qubits and d (even/odd)-row pairs plus one even row to finish off the grid
# we choose the parameters this way, because it simplifies the generation of the connectivity matrix 
def matrixdim(d,l):
    return 4*(d*(2*l-1)+l)

#list of all connections, human readable
def GetConnections(d,l):
    return ConRightEven(d,l)+ConRightOdd(d,l)+ConLeftEven(d,l)+ConLeftOdd(d,l)+ConBoundary(d,l)

#lists of non-zero rows and columns to be feeded into sparse matrix format
def GetNonZeroRows(connections):
    return list(map(TTP,list(map(lambda x: x[0],connections))))

def GetNonZeroColumns(connections):
    return list(map(TTP,list(map(lambda x: x[1],connections))))

# generate sparse connectity matrix (anti-symmetric)
def GenCMatrix(d,l):
    dim = matrixdim(d,l)
    connections= GetConnections(d,l)
    rows=GetNonZeroRows(connections)
    numberofconnections = len(rows)
    valsU = np.ones(numberofconnections)
    valsL = -1*valsU
    allvals = np.concatenate((valsU, valsL))

    cols=GetNonZeroColumns(connections)
    allrows=np.concatenate((rows, cols))
    allcols=np.concatenate((cols, rows))
    return csr_matrix((allvals,(allrows,allcols)), shape = (dim, dim))

# Generate list of the 6 entries of the A matrix (needed for sparse format)
def GetAs(A):
    return [A[0][1],A[0][2],A[0][3],A[1][2],A[1][3],A[2][3]]

# Generate A-matrix in anti-symmetric format
def GenAMatrix(Alist):
    vallist=[]
    rowlist=[]
    collist=[]
    dim=len(Alist)
    for i in range(dim):
        vals=GetAs(Alist[i])
       # row=4*i+[0,0,0,1,1,2]
        row=list(map(lambda x: 4*i+x,[0,0,0,1,1,2]))
       # column=4*i+[1,2,3,2,3,3]
        column=list(map(lambda x: 4*i+x,[1,2,3,2,3,3]))
        vallist=np.concatenate((vallist,vals))
        rowlist=np.concatenate((rowlist,row))
        collist=np.concatenate((collist,column))  
    vallistminus = -1*vallist 
    allrows=np.concatenate((rowlist, collist))
    allcols=np.concatenate((collist, rowlist))
    allvals = np.concatenate((vallist, vallistminus))   
    return csr_matrix((allvals,(allrows,allcols)), shape = (4*dim, 4*dim))

# Generate full matrix 
def GenFullMatrix(Alist,d,l):
    return GenAMatrix(Alist)+GenCMatrix(d,l)


########## Boundary conditions ##########

# function to remove rows and columns

def DeleteRowColFULL(mat,list):
    mat=np.delete(mat,list,0)
    mat=np.transpose(mat)
    mat=np.delete(mat,list,0)
    mat=np.transpose(mat)
    return mat

# some boundary conditions

# consecutive interaval from a to b not occupied (CURRENTLY NOT SUPPORTED)
def AtoB_empty(a,b,shift):
    list=[]
    for i in range (a,b):
        list.append(4*i+shift)
        list.append(4*i+1+shift)
    return list   

# first x sites not occupied
def First_x_empty(d,l,x,top_bottom):
    if top_bottom=="top":
        start=(2*l-1)*d
        shift=2
        list=AtoB_empty(start,start+x,shift)
    if top_bottom=="bottom":
        shift=0
        list=AtoB_empty(0,x,shift)    
    return list

# everysecond site occupied  
def All_Even_Odd_emtpy(d,l,even_odd,top_bottom):
    if top_bottom=="top":
        start=4*(2*l-1)*d+2
        shift=0
    if top_bottom=="bottom":
        start=0
        shift=0
    list=[]
    if even_odd=="even" and top_bottom=="bottom":
        shift=shift+1
    else:
        shift=0
    for i in range(0,l): 
        list.append(start+shift+4*i)
    return list           

# function to select BD condition from key word        

def all_down(d,l,top_bottom):
    list=[]
    dim=matrixdim(d,l)
    if top_bottom=="top":
        for i in range (0,l):
            list.append(4*i)
            list.append(4*i+1)
    if top_bottom=="bottom":
        for i in range (0,l):
            list.append(dim-1-(4*i))
            list.append(dim-1-(4*i+1))
    return list        
        

def select_BD(d,l,key_word,top_bottom):
    if key_word=="even":
        return(All_Even_Odd_emtpy(d,l,"even",top_bottom))
    if key_word=="odd":
        return(All_Even_Odd_emtpy(d,l,"odd",top_bottom))    
    if key_word=="firsthalf":
        return(First_x_empty(d,l,m.floor(l/2),top_bottom))
    if key_word=="alldown":
        return(all_down(d,l,top_bottom))

# implementing BD condition
def Boundary_Top_Bottom(mat,top_BD,bottom_BD,d,l): 
    toplist=select_BD(d,l,top_BD,"top")
    bottomlist=select_BD(d,l,bottom_BD,"bottom")
    return DeleteRowColFULL(mat.toarray(),toplist+bottomlist)  




########## Converting circuit list to ready data ##########

# Generate circuit-matrix, norms and Gammas (plus positions) from list of PPUs (FULL FORMAT BUT SPARSE)
def MatFromUs_Boundary(Ulist,d,l,BD1,BD2):
    lenn=len(Ulist)
    Alist=[]
    Nlist=[]
    Gammlist=[]
    for i in range(lenn):
        u=Ulist[i]
        NAGam=getNAGamma(u)
        Alist.append(NAGam[1])
        Nlist.append(NAGam[0])
        if not(NAGam[2]==0):
            Gammlist.append([i,NAGam[2]])
    return [np.array(Nlist),Boundary_Top_Bottom(GenFullMatrix(Alist,d,l),BD1,BD2,d,l),Gammlist]



########## ALTERNATIVE DECOMPOSITION ##########


#  Fsim(theta), Cphase(phi)

# starting point: for MG background with Fsim(theta,phi) gates (uniform!), generate the zeroth order ulist+positions of nonMG gates
def initial_ulist_and_positions(us,phi,theta):
    gate=np.matmul(FSim(theta,0),agate(phi))    
    positions=[]
    for i in range(len(us)):
        u=us[i]
        NAGam=getNAGamma(u)
        if not(NAGam[2]==0):
            positions.append(i)
            us[i]=gate
    return [us,positions]  
# updating the matrix fomr agate to bgate at all positions 
def update_atob_mat(matrix,l,positions,phi,theta,BD):
    mat=copy.deepcopy(matrix)

    if BD=="half":
        x=1
    if BD=="zero":
        x=2    

    for pos in positions:
        p=int(4*pos-x*l)
        entry1=-np.exp(1j*phi/2)*np.cos(theta)
        entry2=1j*np.exp(1j*phi/2)*np.sin(theta)
        mat[p+0][p+3]=entry1
        mat[p+3][p+0]=-entry1
        mat[p+1][p+2]=entry1
        mat[p+2][p+1]=-entry1

        mat[p+0][p+2]=entry2
        mat[p+2][p+0]=-entry2
        mat[p+1][p+3]=entry2
        mat[p+3][p+1]=-entry2
    return mat    


# pf sum ar order k
def pf_sum_order_k_mat(init_mat,positions,k,d,l,phi,theta,Bd):
    tuples=combinations(positions, k)
    pfs=0
    for tuple in tuples:
        tuple=list(tuple)
        pfs+=cpf(update_atob_mat(init_mat,l,tuple,phi,theta,Bd))
    return pfs
# result at order k 
def res_order_k_mat(init_mat,positions,k,d,l,phi,theta,Bd):
    m=len(positions)
    pf=pf_sum_order_k_mat(init_mat,positions,k,d,l,phi,theta,Bd)
    factor=np.cos(phi/4)**(m-k)*(1j*np.sin(phi/4))**k
    return factor*pf


# compute the result with ortho decomp
def compute_alt_decomp_mat(ulist_in,phi,theta,d,l,BD1,BD2,name,circ):
    namestring=name
    if circ==1:
        circ_res=CirfromUlistManual_BD(ulist_in,d,l,BD1,BD2)
    else: 
        circ_res=0
    ulist=copy.deepcopy(ulist_in)
    init_us_pos=initial_ulist_and_positions(ulist,phi,theta)
    init_us=init_us_pos[0]
    positions=init_us_pos[1]

    dat=MatFromUs_Boundary(init_us,d,l,BD1,BD2)
    init_mat=dat[1]
    nres=listproduct(dat[0])
    m=len(positions)
    
    k=0
    res0=np.cos(phi/4)**m*nres*cpf(init_mat)
    reslist=[[k,res0]]

    if BD1=="alldown":
        Bd="zero"
    else:
        Bd="half"   

    while k<m:
        k+=1
        res=nres*res_order_k_mat(init_mat,positions,k,d,l,phi,theta,Bd)
        reslist.append([k,res])

    result=0
    for r in reslist:
        result+=r[1]
    print("circ result="+str(complex(circ_res))+" pf result="+str(result))

    return reslist    


### slower version based on ulist of the funciotns above
# pfaffian from ulist
def npf(ulist,d,l,BD1,BD2):
    dat=MatFromUs_Boundary(ulist,d,l,BD1,BD2)
    nres=listproduct(dat[0])
    mat=dat[1]
    t1=time.time()
    res=nres*cpf(mat)
    t2=time.time()
    t=t2-t1
    # print("pf time="+str(t))
    return res

def update_atob(ulist,positions,phi,theta):
    gate=np.matmul(FSim(theta,0),bgate(phi))
    newlist=copy.deepcopy(ulist)
    for i in positions:
        newlist[i]=gate
    return newlist    
def pf_sum_order_k(init_us,positions,k,d,l,phi,theta,BD1,BD2):
    res=0
    tuples=combinations(positions, k)
    timelist=[]
    for t in tuples:
        tuple=list(t)
        t1=time.time()
        ulist=update_atob(init_us,tuple,phi,theta)
        res+=npf(ulist,d,l,BD1,BD2)
        t2=time.time()
        t=t2-t1
        timelist.append(t)
    # print(np.mean(timelist))    
    return res
def res_order_k(init_us,positions,k,d,l,phi,theta,BD1,BD2):
    m=len(positions)
    pf=pf_sum_order_k(init_us,positions,k,d,l,phi,theta,BD1,BD2)
    factor=np.cos(phi/4)**(m-k)*(1j*np.sin(phi/4))**k
    # print("at order "+str(k)+" the pf sum is "+str(pf)+ "and the prefactor is "+str(factor))
    return factor*pf
def compute_alt_decomp(ulist,phi,theta,d,l,BD1,BD2,name):
    namestring=name
    circ_res=CirfromUlistManual_BD(ulist,d,l,BD1,BD2)

    init_us_pos=initial_ulist_and_positions(ulist,phi,theta)
    init_us=init_us_pos[0]
    positions=init_us_pos[1]
    m=len(positions)
    
    k=0
    res0=np.cos(phi/4)**m*npf(init_us,d,l,BD1,BD2)
    reslist=[[k,res0]]

    while k<m:
        k+=1
        res=res_order_k(init_us,positions,k,d,l,phi,theta,BD1,BD2)
        reslist.append([k,res])
        np.savetxt("TEMP"+namestring+".txt", reslist)
        np.save("TEMP"+namestring, np.array(reslist))

    result=0
    for r in reslist:
        result+=r[1]
    print("circ result="+str(complex(circ_res))+" pf result="+str(result))

    return reslist    


########## HOLE DECOMPOSITION ##########




# used for normal boundary conditions (0000 to 0000) 2l dimensions removed or with half filling (initial and final), l dimensions removed
def GenReducedMatBatch(mat,positions,l,BD):
    if BD=="half":
        x=1
    if BD=="zero":
        x=2    
    rows=[]
    for pos in positions:
        start=int(4*pos-x*l)
        end=start+4
        list=range(start,end)
        rows.append(list)
    return DeleteRowColFULL(mat,rows)    

# compute the Pfaffian sum at order k
def CompPfaffianSum(mat,NMGlist,k,l,BD):
    tuples=generate_k_tuples(NMGlist,k)
    pfs=0
    c=0
    for tuple in tuples:
        tuple=list(tuple)
        positions=[elem[0] for elem in tuple]
        t1=time.time()
        redmat=GenReducedMatBatch(mat,positions,l,BD)
        t2=time.time()
        pfs+=cpf(redmat)
        t3=time.time()
        c+=1
    return [pfs]

# spit out list [k,result(k)]
def MyMethod(us,d,l,BD1,BD2,name,circ):
    namestring=name
    if circ==1:
        circ_res=CirfromUlistManual_BD(us,d,l,BD1,BD2)
    else: 
        circ_res=0

    data=MatFromUs_Boundary(us,d,l,BD1,BD2)
    ns=data[0]
    nres=listproduct(ns)
    mat=data[1]
    gammas=data[2]
    m=len(gammas)
    # print(m)
    reslist=[]

    k=0
    res0=nres*cpf(mat)
    reslist.append([k,res0])

    if BD1=="alldown":
        Bd="zero"
    else:
        Bd="half"   

    while k<m:
        k+=1
        resss=CompPfaffianSum(mat,gammas,k,l,Bd)
        pfsum=resss[0]
        reslist.append([k,gammas[0][1]**k*nres*pfsum])
        np.savetxt("TEMP"+namestring+".txt", reslist)
        np.save("TEMP"+namestring, np.array(reslist))
        if pfsum==0:
            break


    result=0
    for r in reslist:
        result+=r[1]
    print("circ result="+str(circ_res)+ "my method result="+str(result))
    return reslist    



            
########## SETTINGS and their generation ##########


########## RANDOM ##########

# random MG background
def backgroundMG(d,l):
    MGlist=[]
    size=(2*l-1)*d+l
    for i in range(0,size):
        MGlist.append(RandomMG())
    return MGlist    

###  functions to place non MG gates

# generate list to choose from (exclude boundaries and no double entries)
def generate_unique_random_integers(m, start, end):
    # # Make sure the range is large enough to accommodate m unique integers
    if end - start < m:
        raise ValueError("Range not large enough to generate m unique integers")
    # Generate m unique random integers in the given range
    return random.sample(range(start, end), m)

# choose set m
def PickRandomPositions(m,d,l):
    size=(2*l-1)*d+l
    return generate_unique_random_integers(m, l+1, size-l-1)    

# place specific nonMG gate
def placeNonMG(ulist,gate,position):
    ulistnew=ulist.copy()
    ulistnew[position]=gate
    return ulistnew

# place all nonMGs, uniform gate
def placeAll_NonMG_fixedGate(ulist,gate,m,d,l):
    positions = PickRandomPositions(m,d,l)
    for p in positions:
        ulist=placeNonMG(ulist,gate,p)
    return ulist




########## RANDOM CPHASE ##########

# Generate setting with 
def Random_Cphase(d,l,m,phi):
    MGlist=backgroundMG(d,l)
    us=placeAll_NonMG_fixedGate(MGlist,CPhase(phi),m,d,l)
    return us

########## HUBBARD ##########

# generate trotter circuit: spits out ulist
def HubbardTimeTrotter(d,l,A_hop,A_int):
    Ulist=[]
    Hop=FSim(A_hop,0)
    IntHop=FSim(A_hop,A_int)
    # even hop, odd: hop+inter
    for i in range(0,d):
        for c_e in range(0,l):
            Ulist.append(Hop)
        for c_o in range(0,l-1):
            Ulist.append(IntHop)                      
    for c_e in range(0,l):
        Ulist.append(Id)
    return Ulist




    
 
########## CIRCUIT DIRECT (TN contraction) ##########

# tensor: index reshuffling function
def new_order(n,l):
    old=tuple(range(0,n))
    inset=tuple(range(2*l-2,2*l))
    rest=tuple(range(n,2*l-2))
    order=old+inset+rest
    return order

# circuit boundary conditions
def all_down_cir(l):
    return [0 for _ in range(m.floor(2*l))]    

def firsthalf_cir(l):
    first = [0 for _ in range(m.floor(l))]
    second = [1 for _ in range(m.floor(l))]
    return first+second

def even_cir(l):
    return ([1,0]*l)

def odd_cir(l):
    return ([0,1]*l)

def select_BD_Circ(l,BD):
    if BD=="firsthalf":
        return firsthalf_cir(l)
    if BD=="even":
        return even_cir(l)
    if BD=="odd":
        return odd_cir(l)
    if BD=="alldown":
        return all_down_cir(l)
    
# Circuit with boundary conditions

def CirfromUlistManual_BD(ulist,d,l,BD1,BD2):
    size=2*l

    initial_list=select_BD_Circ(l,BD1)
    final_list=select_BD_Circ(l,BD2)

    x=torch.zeros((2,) * size,dtype=torch.cdouble)
    x[tuple(initial_list)]=1
    # iterate through layers
    for i in range(0,d): 
       # go through the even row
        r1=(2*l-1)*i
        r2=(2*l-1)*i+l
        n=0
        for u in ulist[r1:r2]: # go through the even row
            utens=getUtensor(u) # convert 4x4 matrix to tensor
            torch.tensordot(x,utens,dims=([n,n+1],[2,3]),out=x) #multiply u to qubits n,n+1
            x=torch.permute(x, new_order(n,l))
            n+=2      
       # go through the odd row               
        r1=(2*l-1)*i+l
        r2=(2*l-1)*i+2*l-1
        n=1
        for u in ulist[r1:r2]:
            utens=getUtensor(u) # convert 4x4 matrix to tensor
            torch.tensordot(x,utens,dims=([n,n+1],[2,3]),out=x) #multiply u to qubits n,n+1
            x=torch.permute(x, new_order(n,l))
            n+=2
    # final layer        
    r1=(2*l-1)*d
    r2=(2*l-1)*d+l
    n=0
    for u in ulist[r1:r2]: # go through the last row
        utens=getUtensor(u) # convert 4x4 matrix to tensor
        torch.tensordot(x,utens,dims=([n,n+1],[2,3]),out=x) #multiply u to qubits n,n+1
        x=torch.permute(x, new_order(n,l))        
        # print(x)
        n+=2     
    return  x[tuple(final_list)]

def CirfromUlistManual(ulist,d,l):
    return CirfromUlistManual_BD(ulist, d,l,"down","down")




#################### EVALUATION ####################

def plot_shape(tuple_list,label):
    plt.plot(np.transpose(tuple_list)[0],np.transpose(tuple_list)[1],label=str(label))
    plt.legend()


def load_k_dat_rand(d,l,m,phi,BD1,sample,machine,folder,setting,method):
    load_path=path(machine,folder)
    namestring=name_fun(d,l,m,phi,BD1,method,setting)
    dat=np.load(os.path.join(load_path, namestring+str(sample)+".npy"))
    return dat


def result_from_k(klist,k):
    if k<len(klist):
        res=0
        for i in range(0,k):
            res+=klist[1]
        return res           


def result_full(klist):
    return result_from_k(klist,len(klist))       


