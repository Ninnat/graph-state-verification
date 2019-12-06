# Module for verification of stabilizer/graph states

import numpy as np
from scipy.optimize import linprog
import cvxpy as cp
from math import log, ceil, floor
from fractions import Fraction

from graphLib import graphState # relative import
import networkx as nx
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------

# Adjacency matrix of the n-qubit GHZ state
def GHZ(n):
    A = np.ones([n,n])
    A[-(n-1):,-(n-1):] = 0
    return(A) 

# Adjacency matrix of the n-qubit linear cluster state
def cluster1D(n):
    return (np.diag(np.ones(n-1),1) + np.diag(np.ones(n-1),-1))

# Adjacency matrix of the n-qubit ring state (linear cluster state with periodic boundary) 
def ring(n):
    A = cluster1D(n)
    A[0,-1] += 1
    A[-1,0] += 1
    return(A)

#-----------------------------------------------------------------------------------------------------------

# Output binary symplectic vectors + phase bits of all stabilizers of a graph state defined by the adjacency matrix A
def stabTable(A):
    n = A.shape[0]
    pcheck = np.concatenate((np.diag(np.ones(n)),A),axis=1) # parity-check matrix
    v = [[] for j in range(2**n)]
    j = 0
    while j < 2**n:
        v[j] = np.zeros(2*n) # each binary symplectic vector
        q = bin(j)
        p = q[2:].zfill(n)
        # add a stabilizer generator whenever the corresponding value in the binary j index is 1
        genlist = [j for j, x in enumerate(p) if x == '1']
        for gen in genlist:
            v[j] = np.remainder((v[j] + pcheck[-1-gen,:]),2)
        j += 1
    return(v)

# Enumerate binary symplectic vectors: https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-in-any-base-to-a-string
def intToSymplec(j,n,XZ = None):
    if j == 0:
        return ([0]*n + [1]*n) 
    vec = [0]*n + [1]*n # symplectic vector (X,Z)
    k = 0
    if XZ == 1:
        while j:
            if int(j % 2) == 0:
                vec[-1-n-k],vec[-1-k] = 0,1
            if int(j % 2) == 1:
                vec[-1-n-k],vec[-1-k] = 1,0
            j //= 2
            k += 1
    else:
        while j:
            if int(j % 3) == 0:
                vec[-1-n-k],vec[-1-k] = 0,1
            if int(j % 3) == 1:
                vec[-1-n-k],vec[-1-k] = 1,0
            if int(j % 3) == 2:
                vec[-1-n-k],vec[-1-k] = 1,1
            j //= 3
            k += 1
    return(vec)

def checkLocalMeasurable(u,v,n):
    j = 0
    while j < n:
        if (u[j] == v[j] and u[j+n] == v[j+n]) or (u[j] == 0 and u[j+n] == 0) or (v[j] == 0 and v[j+n] == 0):
            j += 1
        else: j = n+1
    if j == n:
        return('Yes')
    else: return('No')

# Output all locally measurable stabilizer codes of a graph state defined by the adjacency matrix A
def localMeasurableCode(A,XZ = None):
    stat = stabTable(A)
    n = A.shape[0]
    j = 0 
    lst = [] 
    if XZ == 1:
        m = 2**n
    else:
        m = 3**n
        
    while j < m: #loop over Pauli strings
        k = 1 
        code = []
        while k < 2**n: #loop over the stabilizers
            if checkLocalMeasurable(intToSymplec(j,n,XZ),stat[k],n) == 'Yes':
                code.append(k)
            k += 1
        code.append(j) # Keep the Pauli string at the end of the subgroup
        lst.append(code)
        j += 1
        
    filt = [x for x in lst if not any(set(x[:-1]) <= set(y[:-1]) for y in lst if x is not y)] # https://stackoverflow.com/a/1319083
    
    for x in filt:
        vec = intToSymplec(x[-1],n,XZ)
        tervec = []
        l = 0
        while l < n:
            tervec.append(int(2*vec[l] + vec[l+n]))
            l += 1
        x[-1] = strToPauli(''.join(str(x) for x in tervec))
    return(sorted(filt, key=len, reverse=True))

def strToPauli(s): # https://codereview.stackexchange.com/a/183660
    pauli = {'0':'I','1':'Z','2':'X','3':'Y'}
    paulistr = [pauli[char] for char in s]
    return ''.join(paulistr)   


#-----------------------------------------------------------------------------------------------------------

# Group theory

# Character table
def charTable(n):
    T = np.zeros(shape=(2**n-1,2**n))
    j = 1
    while j < 2**n:
        k = 0
        while k < 2**n:
            row,col = np.unpackbits(np.array([j], dtype=np.uint32).view(np.uint8)[::-1]), np.unpackbits(np.array([k], dtype=np.uint32).view(np.uint8)[::-1]) # 32 qubits maximum
            T[j-1,k] += (-1)**(np.dot(row,col))
            k += 1
        j += 1
    return(T)

# 0-1 character table
def binCharTable(n):
    T = np.zeros(shape=(2**n-1,2**n))
    j = 1
    while j < 2**n:
        k = 0
        while k < 2**n:
            row, col = np.unpackbits(np.array([j], dtype=np.uint32).view(np.uint8)[::-1]), np.unpackbits(np.array([k], dtype=np.uint32).view(np.uint8)[::-1]) # 32 qubits maximum
            T[j-1,k] += 1 - np.remainder(np.dot(row,col),2)
            k += 1
        j += 1
    return(T)

# This is the (2^n)-1 x m (locally measurable subgroups) matrix to be use in the linear program
def subCharTable(T,subgrp):
    m = len(subgrp) # number of locally measurable subgroups
    n = T.shape[0] # 2^n-1
    A = np.ones((n,m)) # The 1's is for the identity in the sum
    k = 0
    while k < m:
        for l in subgrp[k][:-1]:
            A[:,k] += T[:,l]
        A[:,k] = A[:,k]/(len(subgrp[k][:-1]) + 1) # normalize by the subgroup's order 
        k += 1
    return(A) 

def eigenTable(A,XZ = None):
    local = localMeasurableCode(A,XZ)
    return(local,subCharTable(charTable(A.shape[0]),local))
        
#-----------------------------------------------------------------------------------------------------------

# Optimization

def kToN(k):
    if k <= 3:
        n = k+1
    elif k == 4:
        n = 4
    elif k < 9:
        n = 5
    elif k < 20:
        n = 6
    else: n = 7
    return(n)


def optimal(A): # A = eigentable
    n,m = len(A[0]),len(A[1]) # n = number of local subgroups, m = (2**N)-1 
    x = cp.Variable()
    p = cp.Variable(shape=n,nonneg = True)
    constraints = [cp.sum(p) == 1] # Probabilities add up to 1
    j = 0
    while j < m:  
        constraints += [
            (A[1]*p)[j] - x <= 0,
            -(A[1]*p)[j] - x <= 0
        ]
        j += 1
    obj = cp.Minimize(x)
    prob = cp.Problem(obj,constraints)
    prob.solve(solver = cp.GLPK, verbose = False)
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", p.value)

def minimalSettings(A,gap): # A = eigentable
    # https://www.cvxpy.org/examples/applications/sparse_solution.html#iterative-log-heuristic

    n = len(A[0]) # len(A[0]) = number of local subgroups
    # The threshold value below which we consider an element to be zero.
    delta = 1e-8
    # Do 10 iterations, allocate variable to hold number of non-zeros
    # (cardinality of p) for each run.
    NUM_RUNS = 30
    nnzs_log = np.array(())

    W = cp.Parameter(shape=n, nonneg=True);
    p = cp.Variable(shape=n, nonneg=True)

    # Initial weights.
    W.value = np.ones(n);

    obj = cp.Minimize( W.T*cp.abs(p) )
    constraints = [cp.sum(p) == 1, A*p <= 1-gap]
    prob = cp.Problem(obj, constraints)

    for k in range(1, NUM_RUNS+1):
        # Solve problem.
        # The ECOS solver has known numerical issues with this problem
        # so force a different solver.
        prob.solve(solver=cp.GLPK)

        # Check for error.
        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")

        # Display new number of nonzeros in the solution vector.
        nnz = (np.absolute(p.value) > delta).sum()
        nnzs_log = np.append(nnzs_log, nnz);
        # print('Iteration {}: Found a feasible p in R^{}'
        #      ' with {} nonzeros...'.format(k, n, nnz))

        # Adjust the weights elementwise and re-iterate
        W.value = np.ones(n)/(delta*np.ones(n) + np.absolute(p.value))
    #print('Iteration {}: Found a feasible p in R^{}'
     #         ' with {} nonzeros...'.format(k, n, nnz))
    return(p.value,nnz)

def autoGraph(k,gap,XZ = None): # k = graph label
    
    n = kToN(k)
        
    graph = graphState(k)
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    [B,C] = eigenTable(A,XZ)
    
    print(k)
    plt.figure(k) # produce a seperate plot for each graph
    nx.draw(graph,node_size=1000,node_color='xkcd:salmon',with_labels = True)
    plt.show()
    
    Min = minimalSettings(C,gap)[0]
    print(Min)
    nonzeroPos = [j for j, y in enumerate(Min) if y > 1e-10] # only display probabilities significantly away from zero
    for j in nonzeroPos:
        print('{} of {} rank {} measuring {}'.format( Fraction(Min[j]).limit_denominator() , [bin(x)[2:].zfill(n) for x in B[j][:-1]] , int( (2**n)/(len(B[j][:-1])+1)), B[j][-1] ))
    return()

def autoDisplay(k,gap,XZ = None): # k = graph label
    
    n = kToN(k)
        
    graph = graphState(k)
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    [B,C] = eigenTable(A,XZ)
    
    plt.figure(k) # produce a seperate plot for each graph
    nx.draw(graph,node_size=1000,node_color='xkcd:salmon',with_labels = True)
    plt.show()
    
    [Min,nnz] = minimalSettings(C,gap)
    print('\\multirow{{{0}}}{{*}}{{{1}}}'.format(nnz,k))
    nonzeroPos = [j for j, y in enumerate(Min) if y > 1e-10] # only display eigenvalues significantly far away from zero 
    for j in nonzeroPos:
        print('& {} & {} & {} & & & \\\\'.format(B[j][-1] , Fraction(Min[j]).limit_denominator() , int( (2**n)/(len(B[j][:-1])+1)))) 
    print('\\cline{2-4}')
    return()

def isHomogeneous(A): # A = eigentable
    n,m = len(A[0]),len(A[1]) # n = number of local subgroups, m = (2**N)-1 
    x = cp.Variable()
    p = cp.Variable(shape=n, nonneg = True)
    constraints = [cp.sum(p) == 1,(A[1]*p)[0] - x <= 0, -(A[1]*p)[0] - x <= 0]
    j = 1
    while j < m:
        constraints += [
            (A[1]*p)[j] == (A[1]*p)[j-1]
        ]   
        j += 1
    obj = cp.Minimize(x)
    prob = cp.Problem(obj,constraints)
    prob.solve(solver = cp.ECOS, verbose = False)
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", p.value) 

def random(k,M,iteration,XZ = None): # k = graph label, M = number of settings  

    n = kToN(k)   
        
    graph = graphState(k)
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    [B,C] = eigenTable(A,XZ)
    m = len(B)    
    l = 0
    while l < iteration:
        ran = np.random.choice(m,size=M,replace=False)
        beta = np.amax(np.mean(C[:,ran],axis=1))
        if beta < 1:
            print(1 - beta)
            for j in ran:
                print("{} measures {} generators".format(B[j][-1],int(  log(len(B[j][:-1])+1 , 2) ) )) 
            l = iteration
        else:
            l += 1
    return() 

def randomDisplay(k,M,iteration,XZ = None): # k = graph label, M = number of settings  

    n = kToN(k)   
        
    graph = graphState(k)
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    [B,C] = eigenTable(A,1)
    m = len(B)    
    l = 0
    while l < iteration:
        ran = np.random.choice(m,size=M,replace=False)
        beta = np.amax(np.mean(C[:,ran],axis=1))
        if beta < 1:
            print(1 - beta)
            print('\\multirow{{3}}{{*}}{{{0}}} & \\multirow{{3}}{{*}}{{{1}}} & \\multirow{{3}}{{*}}{{3}} & \\multirow{{3}}{{*}}{{1/3}}'.format(k,n))
            # label &  number of qubits & chromatic number & spectral gap
            for j in ran:
                print("& & & & {} & {} & {} \\\\".format(B[j][-1],int( (2**n)/(len(B[j][:-1])+1)),int(  log(len(B[j][:-1])+1 , 2) ) )) 
            l = iteration
        else:
            l += 1
    print('\\hline')
    return() 

def allPairs(k,XZ = None):
    
    n = kToN(k)
    graph = graphState(k)
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    [B,C] = eigenTable(A,XZ)
    m = len(B)
    p = 0
    counter = 0
    while p < m:
        q = m-1
        while p < q:
            beta = np.amax((C[:,p] + C[:,q])/2)
            if beta < 1:
                print("{} from {} and {} of rank {} and {}".format(1 - beta, B[p][-1], B[q][-1], int( (2**n)/(len(B[p][:-1])+1)), int( (2**n)/(len(B[q][:-1])+1))))
                q = p
                p = m
            else:
                q -= 1
                counter += 1
        p += 1
        counter += 1
    return(counter)             
    
#-----------------------------------------------------------------------------------------------------------

def rankScaling(k):
    
    n = kToN(k)
    graph = graphState(k)
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    [B,C] = eigenTable(A)
    totalp = 0
    for col in B:
        print("{} bin {}".format(col[-1],[bin(x)[2:].zfill(n) for x in col[:-1]]))
        rank = int((2**n)/(len(col[:-1])+1))
        p = 2/(3*rank)
        j = B.index(col)
        C[:,j] = C[:,j]*p
        totalp += p
    print(totalp)
    return(np.amax(np.sum(C,axis=1)))

#-----------------------------------------------------------------------------------------------------------

# Cluster state heuristic

# Generate all n bit strings that contain no consecutive 0's

def fibonacci(n):
    # base case
    if n==1:
        return ['0','1']
    elif n==2:
        return ['10','01','11']
    else:
        return [x + '1' for x in fibonacci(n-1)] + [y + '10' for y in fibonacci(n-2)]
    
# Erase sequences with the greatest number of consecutive 1's >= n and <= m (Python can't remove a list element while iterating through a list)

def deleteFibonacci(F,n,m):
    a = []
    for x in F:
        if n <= max(map(len,x.split('0'))) <= m:
            a += [x]
    return( list(set(F) - set(a)) )

def subgrpFromFibonacci(F):
    
    a = [[] for j in range(len(F))] # list of empty lists
    n = len(F[0])
    
    for x in F:
        
        #a[F.index(x)] += x.split('0')
        zeroSet = [j for j, e in enumerate(x) if e == '0']
        if not zeroSet:
            a[F.index(x)].append(x)
        #rank = x[1:-1].count('0') # Rank of subgroup = Number of generators
        else:
            if zeroSet[0] != 0:
                a[F.index(x)].append(('1'*zeroSet[0]).zfill(n)[::-1]) # Add 1's if the string doesn't start with 0
            j = 0
            while j < len(zeroSet) - 1:
                a[F.index(x)].append(('1'*(zeroSet[j+1] - zeroSet[j] - 1) + '0'*(zeroSet[j] + 1)).zfill(n)[::-1])
                j += 1
            if zeroSet[-1] != n-1:
                a[F.index(x)].append(('1'*(n - zeroSet[-1] - 1)).zfill(n))
    return(sorted(a, key=len, reverse=True))

def eigenFibonacci(n,lower = None,upper = None): # k-qubit cluster states, with no number of consecutive 1's in the Fibonacci string between (and including) lower and upper
    if upper is None:
        upper = lower
    if lower is None:
        A = subgrpFromFibonacci(fibonacci(n))
        subgrp = [[int(gen,2) for gen in subgroup] for subgroup in A]
    else:
        A = subgrpFromFibonacci(deleteFibonacci(fibonacci(n),lower,upper))
        subgrp = [[int(gen,2) for gen in subgroup] for subgroup in A]
    m = len(subgrp)
    T = binCharTable(n)
    B = np.ones((2**n-1,m))
    j = 0
    while j < m:
        for k in subgrp[j]:
            B[:,j] = np.multiply(B[:,j],T[:,k]) 
        j += 1
    return(A,B)    
 
def eigenDepthk(n,k): # k-qubit cluster states, with no number of consecutive 1's in the Fibonacci string between (and including) lower and upper
    if k == 1:
        subgrp = [[int(gen,2) for gen in subgroup] for subgroup in subgrpFromFibonacci(deleteFibonacci(fibonacci(n),2,n))]
    elif k == n:
        subgrp = [[int(gen,2) for gen in subgroup] for subgroup in subgrpFromFibonacci(deleteFibonacci(fibonacci(n),1,n-1))]
    else:
        subgrp = [[int(gen,2) for gen in subgroup] for subgroup in subgrpFromFibonacci( deleteFibonacci( deleteFibonacci(fibonacci(n),1,k-1) ,k+1,n) )]
        print(subgrpFromFibonacci( deleteFibonacci( deleteFibonacci(fibonacci(n),1,k-1) ,k+1,n) ))
    m = len(subgrp)
    T = binCharTable(n)
    A = np.ones((2**n-1,m))
    j = 0
    while j < m:
        for k in subgrp[j]:
            A[:,j] = np.multiply(A[:,j],T[:,k]) 
        j += 1
    return(A) 

def rank(n):
    lst = []
    for x in subgrpFromFibonacci(fibonacci(n)):
        lst += [len(x)]
    return(list(set(lst)))

#def eigenDepth(n):
#    A = np.reshape(np.sum(eigenDepthk(n,1)/len(eigenDepthk(n,1)[0]),axis=1),(2**n-1,1))
#    k = 2
#    while k <= n:
#        A = np.append(A,np.reshape(np.sum(eigenDepthk(n,k)/len(eigenDepthk(n,k)[0]),axis=1),(2**n-1,1)),axis=1)
#        k += 1
#    return(A)

def eigenDepth(n):
    A = np.reshape(np.sum(eigenDepthk(n,1)/len(eigenDepthk(n,1)[0]),axis=1),(2**n-1,1))
    A = np.append(A,np.reshape(np.sum(eigenDepthk(n,n)/len(eigenDepthk(n,n)[0]),axis=1),(2**n-1,1)),axis=1)
    #A = np.reshape((np.sum(eigenDepthk(n,1)/2,axis=1) + np.sum(eigenDepthk(n,n),axis=1) ),(2**n-1,1))
    A = np.append(A,eigenDepthk(n,2),axis=1)
    return(A)

# For cluster states with with 1/6 depth n Fibonacci measurement
def cluster16(n,depth3 = None):
    
    Dn = eigenDepthk(n,n)
    D1 = np.sum(eigenDepthk(n,1)/len(eigenDepthk(n,1)[0]),axis=1) 
    A = np.append(Dn,np.reshape(D1,(2**n-1,1)),axis=1)
    A = np.append(A,eigenDepthk(n,2),axis=1)
    if depth3 == True:
        A = np.append(A,eigenDepthk(n,3),axis=1)
    m = len(A[0]) # len(A[0]) = number of local subgroups
    delta = 1e-8
    NUM_RUNS = 30
    nnzs_log = np.array(())

    W = cp.Parameter(shape=m, nonneg=True);
    p = cp.Variable(shape=m, nonneg=True)

    W.value = np.ones(m);

    obj = cp.Minimize( W.T*cp.abs(p) )
    constraints = [cp.sum(p) == 1, A*p <= 1/3, p[0] == 1/6]
    prob = cp.Problem(obj, constraints)

    for k in range(1, NUM_RUNS+1):
        prob.solve(solver=cp.GLPK)
        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        nnz = (np.absolute(p.value) > delta).sum()
        nnzs_log = np.append(nnzs_log, nnz);
        W.value = np.ones(m)/(delta*np.ones(m) + np.absolute(p.value))
    print('Iteration {}: Found a feasible p in R^{}'
              ' with {} nonzeros...'.format(k, m, nnz))
    return(p.value)

# For cluster states with with 1/6 depth n Fibonacci measurement, and 1/(3*2^(floor(n/2)-2))
def cluster32(n):
    
    Dn = eigenDepthk(n,n)
    D1 = np.sum(eigenDepthk(n,1)/len(eigenDepthk(n,1)[0]),axis=1) 
    A = np.append(Dn,np.reshape(D1,(2**n-1,1)),axis=1)
    A = np.append(A,eigenDepthk(n,2),axis=1)

    m = len(A[0]) # len(A[0]) = number of local subgroups
    # The threshold value below which we consider an element to be zero.
    delta = 1e-8
    # Do 10 iterations, allocate variable to hold number of non-zeros
    # (cardinality of p) for each run.
    NUM_RUNS = 30
    nnzs_log = np.array(())

    W = cp.Parameter(shape=m, nonneg=True);
    p = cp.Variable(shape=m, nonneg=True)

    # Initial weights.
    W.value = np.ones(m);

    obj = cp.Minimize( W.T*cp.abs(p) )
    constraints = [cp.sum(p) == 1, A*p <= 1/3, p[0] == 1/6, p[1] == 1/(3*(2**(floor(n/2)-2)))]
    prob = cp.Problem(obj, constraints)

    for k in range(1, NUM_RUNS+1):
        # Solve problem.
        # The ECOS solver has known numerical issues with this problem
        # so force a different solver.
        prob.solve(solver=cp.GLPK)

        # Check for error.
        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")

        # Display new number of nonzeros in the solution vector.
        nnz = (np.absolute(p.value) > delta).sum()
        nnzs_log = np.append(nnzs_log, nnz);
        # print('Iteration {}: Found a feasible p in R^{}'
        #      ' with {} nonzeros...'.format(k, n, nnz))

        # Adjust the weights elementwise and re-iterate
        W.value = np.ones(m)/(delta*np.ones(m) + np.absolute(p.value))
    print('Iteration {}: Found a feasible p in R^{}'
              ' with {} nonzeros...'.format(k, m, nnz))
    return(p.value)