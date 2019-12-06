# graph-state-verification
 Package for verification of quantum graph states

`graphVerify.py` is a Python module for finding an efficient verification scheme for a quantum graph state. In principle, the graph state can be arbitrary, but currently the code only works up to 32 qubits because of the `numpy.unpackbits` command (line 138 and 151).

# Usage and examples

The main input to the module is an adjacency matrix of the desired graph state. For an arbitrary number of qubits $n$, the adjacency matrices of some well known families of graph states are provided:
- `GHZ(n)`: $n$-qubit GHZ states
- `cluster1D(n)`, $n$-qubit linear cluster states
- `ring(n)`, $n$-qubit ring states (linear cluster states with periodic boundary condition)

`graphLib.py` contains native `networkx` descriptions of all graph states up to seven qubit graph states. (There are [45 local-unitary-equivalent classes](https://arxiv.org/abs/quant-ph/0307130) of them in total.) The graphs are not yet in the matrix format appropriate for the input and need to be turn into ones via `networkx.to_numpy_array`.

1. Finding a sample-optimal verification scheme

```
import graph_verify as gv
import networkx as nx

n = gv.k_to_n(k)

graph = graphState(k)
A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
[B,C] = gv.eigen_table(A)
```
