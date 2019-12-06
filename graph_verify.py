# Module for verification of stabilizer/graph states

import numpy as np
from scipy.optimize import linprog
import cvxpy as cp
from math import log, ceil, floor
from fractions import Fraction

from graph_library import graphState # relative import
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
def stab_table(A):
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
def int_to_symplec(j,n,XZ = None):
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

def check_local(u,v,n):
    j = 0
    while j < n:
        if (u[j] == v[j] and u[j+n] == v[j+n]) or (u[j] == 0 and u[j+n] == 0) or (v[j] == 0 and v[j+n] == 0):
            j += 1
        else: j = n+1
    if j == n:
        return('Yes')
    else: return('No')

# Output all locally measurable stabilizer codes of a graph state defined by the adjacency matrix A
def local_subgroups(A,XZ = None):
    stat = stab_table(A)
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
            if check_local(int_to_symplec(j,n,XZ),stat[k],n) == 'Yes':
                code.append(k)
            k += 1
        code.append(j) # Keep the Pauli string at the end of the subgroup
        lst.append(code)
        j += 1

    filt = [x for x in lst if not any(set(x[:-1]) <= set(y[:-1]) for y in lst if x is not y)] # https://stackoverflow.com/a/1319083

    for x in filt:
        vec = int_to_symplec(x[-1],n,XZ)
        tervec = []
        l = 0
        while l < n:
            tervec.append(int(2*vec[l] + vec[l+n]))
            l += 1
        x[-1] = str_to_pauli(''.join(str(x) for x in tervec))
    return(sorted(filt, key=len, reverse=True))

def str_to_pauli(s): # https://codereview.stackexchange.com/a/183660
    pauli = {'0':'I','1':'Z','2':'X','3':'Y'}
    paulistr = [pauli[char] for char in s]
    return ''.join(paulistr)


#-----------------------------------------------------------------------------------------------------------

# Group theory

# Character table
def char_table(n):
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
def bin_char_table(n):
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
def sub_char_table(T,subgrp):
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

def eigen_table(A,XZ = None):
    local = local_subgroups(A,XZ)
    return(local,sub_char_table(char_table(A.shape[0]),local))

#-----------------------------------------------------------------------------------------------------------

# Optimization

def k_to_n(k):
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

def min_settings(A,gap): # A = eigentable
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
        prob.solve(solver=cp.GLPK)

        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")

        # Display new number of nonzeros in the solution vector.
        nnz = (np.absolute(p.value) > delta).sum()
        nnzs_log = np.append(nnzs_log, nnz);
        # Adjust the weights elementwise and re-iterate
        W.value = np.ones(n)/(delta*np.ones(n) + np.absolute(p.value))
    return(p.value,nnz)

def auto_graph(k,gap,XZ = None): # k = graph label

    n = k_to_n(k)

    graph = graphState(k)
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    [B,C] = eigen_table(A,XZ)

    print(k)
    plt.figure(k) # produce a seperate plot for each graph
    nx.draw(graph,node_size=1000,node_color='xkcd:salmon',with_labels = True)
    plt.show()

    Min = min_settings(C,gap)[0]
    print(Min)
    nonzeroPos = [j for j, y in enumerate(Min) if y > 1e-10] # only display probabilities significantly away from zero
    for j in nonzeroPos:
        print('{} of {} rank {} measuring {}'.format( Fraction(Min[j]).limit_denominator() , [bin(x)[2:].zfill(n) for x in B[j][:-1]] , int( (2**n)/(len(B[j][:-1])+1)), B[j][-1] ))
    return()

def is_homogeneous(A): # A = eigentable
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

    n = k_to_n(k)

    graph = graphState(k)
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    [B,C] = eigen_table(A,XZ)
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

def all_pairs(k,XZ = None):

    n = k_to_n(k)
    graph = graphState(k)
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    [B,C] = eigen_table(A,XZ)
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
