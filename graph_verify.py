# Module for verification of graph states

import numpy as np
import cvxpy as cp
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
from math import log, ceil, floor
from fractions import Fraction

from graph_library import graphState # relative import


#-----------------------------------------------------------------------------------------------------------

# Adjacency matrix of the n-qubit GHZ state
def GHZ(n):
    A = np.ones([n,n])
    A[-(n-1):,-(n-1):] = 0
    A[0,0] = 0
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

def intToSymplec(j, n):
    vec = [0]*n + [1]*n # symplectic vector (X,Z)
    k = 0
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


def intToPauli(j, n):
    pauli = ['Z']*n
    k = 0
    while j:
        if int(j % 3) == 0:
            pauli[-1-k] = 'Z'
        if int(j % 3) == 1:
            pauli[-1-k] = 'X'
        if int(j % 3) == 2:
            pauli[-1-k] = 'Y'
        j //= 3
        k += 1
    return(''.join(pauli))

#-----------------------------------------------------------------------------------------------------------  
# Test projectors

def localCommutator(A):
    n = A.shape[0]
    symplec = np.array([intToSymplec(j,n) for j in range(3**n)])
    
    Z = np.zeros((*symplec[:,n:].shape, symplec[:,n:].shape[-1]), symplec[:,n:].dtype)
    np.einsum('...jj->...j', Z)[...] = symplec[:,n:] # einsum magic https://stackoverflow.com/questions/48627163/construct-n1-dimensional-diagonal-matrix-from-values-in-n-dimensional-array
    X = np.multiply(symplec[:,:n][:,:,None],A)
    
    return(Z + X)

def unpackbits(x,n): # np.unpackbits but for an arbitrary number of bits https://stackoverflow.com/a/51509307/13049108
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(n)[::-1].reshape([1, n])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [n])

def rspan(local):
    n = local[0].shape[1]
    i = np.array([j for j in range(2**n)]) 
    # bin_matrix = np.unpackbits(i, axis=1)[:,-n:] # up to 8 qubits
    bin_matrix = unpackbits(i, n)
    return(np.matmul(bin_matrix,local)%2)


def testMat(adm):
    n = len(adm[0][1])
    m = len(adm)
    C = [i[0] for i in adm]
    testmat = np.array([[0]*(2**n)]*m)
    j = 0
    while j < m:
        for k in C[j]:
            testmat[j][k] = 1
        j += 1
    return(np.transpose(testmat))


def admissible(A): # A = adjacency matrix
    n = A.shape[0]
    B = localCommutator(A)   
    paulis = [intToPauli(j, n) for j in range(3**n)]
    
    # eliminate trivial tests
    condition = [np.linalg.det(x)%2 < 1e-10 or (2-np.linalg.det(x))%2 < 1e-10 for x in B]
    rank_def = list(it.compress(B,condition))
    rank_def_paulis = list(it.compress(paulis,condition))
    
    # convert row span to position of 1's in test vector
    bit_converter = 2**np.arange(n-1,-1,-1)
    testvec = (rspan(rank_def).dot(bit_converter)).astype(int)
    testvec2 = [set(x) for x in testvec]
    testsort, paulisort = zip(*sorted(zip(testvec2,rank_def_paulis),key=lambda x: len(x[0]))) # https://stackoverflow.com/questions/38240236/python-sorting-a-zip-based-on-length-and-weight

    # filter admissible projectors by rank
    index = [index for index, (item,group) in enumerate(it.groupby(testsort,len))]
    test = [[j for j in group] for index, (item, group) in enumerate(it.groupby(testsort,len))]
    j = 1
    while j <= max(index):
        test[j] = [set(i) for i in set(frozenset(i) for i in test[j])]
        j += 1
    j = 0
    while j < max(index):
        k = j+1
        while k <= max(index):
            test[k] = [x for x in test[k] if not any(x >= y for y in test[j])]
            k += 1
        j += 1 
    adm = test[0]
    j = 1
    while j <= max(index):
        adm += test[j]
        j += 1
    adm_pair = [x for x in list(zip(testsort,paulisort)) if x[0] in adm]
    return(adm_pair,testMat(adm_pair)[1:])

        
#-----------------------------------------------------------------------------------------------------------
# Optimization


def optimal(test): # test matrix (of 0-1 eigenvalues)
    N,m = test.shape[0], test.shape[1]
    x = cp.Variable()
    p = cp.Variable(shape = m, nonneg = True)
    constraints = [cp.sum(p) == 1]
    j = 0
    while j < N:  
        constraints += [
            (test*p)[j] - x <= 0,
            -(test*p)[j] - x <= 0
        ]
        j += 1
    obj = cp.Minimize(x)
    prob = cp.Problem(obj,constraints)
    prob.solve(solver = cp.GLPK)
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", p.value)

def minSet(test,gap): # A = eigentable
    
    m = test.shape[1]
    
    # https://www.cvxpy.org/examples/applications/sparse_solution.html#iterative-log-heuristic
    delta = 1e-8 # threshold for 0
    NUM_RUNS = 30
    nnzs_log = np.array(()) # (cardinality of p) for each run

    W = cp.Parameter(shape = m, nonneg=True);
    p = cp.Variable(shape = m, nonneg=True)

    W.value = np.ones(m);  # Initial weights

    obj = cp.Minimize( W.T*cp.abs(p) )
    constraints = [cp.sum(p) == 1, test*p <= 1-gap]
    prob = cp.Problem(obj, constraints)

    for k in range(1, NUM_RUNS+1):
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
    return(p.value,nnz)
    
def equiprob(adm, test, k, notAllow = None): # iterate over all verifications with k settings with equal probabilities
    # you can put notAllow = 'I' if all Pauli settings are allowed
    cols = [row for row in np.transpose(test)]
    condition = [notAllow not in x[1] for x in adm]
    cols2 = list(it.compress(cols, condition)) # list is okay here
    adm2 = list(it.compress(adm, condition))
    index = it.combinations(np.arange(len(cols2)),k)
    combi = it.combinations(cols2,k)
    verify = next((i for i,x in zip(index,combi) if np.amax(np.sum(x,axis=0)/k) < 1 ), None)
    if verify != None:
        for j in verify:
            print(adm2[j])
        return()
    else:
        return()
    
def localCover2(graph): # the minimum number of settings when each party only have two choices of Paulis
    n = graph.number_of_nodes()
    A = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
    adm, test = admissible(A)
    cols = [row for row in np.transpose(test)]
    k = 2
    while k <= n:
        j = 0
        while j < 3**n:
            s = intToPauli(j,n)
            condition = [not any(c1 == c2 for c1,c2 in zip(s, x[1])) for x in adm]
            cols2 = list(it.compress(cols,condition))
            combi = list(it.combinations(cols2,k))
            if next((x for x in combi if np.amax(np.sum(x,axis=0)/k) < 1),None) != None:
                print('tilde_chi_2 = {}; No {}'.format(k, s))
                return(combi)
            else:
                j += 1
        k += 1
    return(print('Something is wrong')) # local cover number with 2 settings cannot be greater than the chromatic number

def colorProt(graph0, strategy = None): # coloring protocol
    n = graph0.number_of_nodes()
    graph = nx.convert_node_labels_to_integers(graph0, first_label = 1)
     
    # sometimes the default strategy cannot come up with a minimum coloring 
    # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.coloring.greedy_color.html
    if strategy != None: 
        min_coloring = nx.greedy_color(graph, strategy)
    else:
        min_coloring = nx.greedy_color(graph)
    colors = [min_coloring.get(color) for color in min_coloring]
    
    plt.figure(1)
    nx.draw(graph, node_size=1000, font_size=20, with_labels = True, node_color = [min_coloring.get(node) for node in graph.nodes()], cmap = plt.cm.Set1, vmax = 8)
    plt.show()
    #plt.savefig("./img_coloring/graph-"+str(k)+".png", format="PNG", transparent = True)
    
    A = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
    adm = admissible(A)[0]
    
    test = np.zeros(2**n)
    for color in set(colors):
        pos = [k for (k, v) in min_coloring.items() if v == color]
        mu = np.sum([3**(n-x) for x in pos]) # these are the settings we want to construct the test vectors from
        # graph node has to start at 1; otherwise replace x by x+1
        pauli = intToPauli(mu, n)
        
        meas = rspan(localCommutator(A))[mu]
        bit_converter = 2**np.arange(n-1,-1,-1)
        testvec = (meas.dot(bit_converter)).astype(int)
        testvec2 = list(set(testvec))
        rank = len(testvec2)
        gen = int(log((2**n)/rank, 2))
        print('{}: rank = {}, # subgroup gen = {}, admissible? {} {}'.format(pauli, rank, gen, pauli in [x[1] for x in adm], [x[1] for x in adm if set(x[0]).issubset(testvec2) == True]))  
        
        testmat = [0]*(2**n)
        for x in testvec2:
            testmat[x] = 1
        test += testmat 
    print(test[1:])
    print('nu = {}'.format(1 - Fraction(np.amax(test[1:])/len(set(colors))).limit_denominator()))
    return()