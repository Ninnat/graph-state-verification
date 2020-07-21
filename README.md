# graph verify
 Package for verification of quantum graph states

`graph_verify.py` is a Python module for finding measurement schemes that efficiently verify a given quantum graph state of arbitrary dimension.

Accompanying paper: [https://arxiv.org/abs/2007.09713](https://arxiv.org/abs/2007.09713)

# Usage and example

The input to the module is the adjacency matrix of a graph state |G⟩ as a `numpy` array. The adjacency matrices of some well known families of graph states are provided:
- `GHZ(n)`: n-qubit GHZ states
- `cluster1D(n)`: n-qubit linear cluster states
- `ring(n)`: n-qubit ring cluster states

Up to seven qubits, there are in total [45 local-unitary-equivalent classes](https://arxiv.org/abs/quant-ph/0307130) of genuinely entangled graph states. Their graphs are provided as `graphState(k)` for k=1 to 45 in `graph_library.py` in the native `networkx` format, which can be turned into adjacency matrices via `networkx.to_numpy_array`. (See the code snippet below.)

The main routine `admissible` outputs Pauli strings for all m admissible Pauli measurement settings for |G⟩ (see Section IV of the [paper](https://arxiv.org/abs/2007.09713)) and their associated (2^n-1)-by-m test matrix. The (j,k) matrix element of the test matrix is the eigenvalue in the (j+1)th graph-basis state (excluding the first one which is always 1) of the kth admissible measurement setting.
```
import graph_verify as gv
import networkx as nx

graph = graphState(k)
A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
[adm, test] = gv.admissible(A)
```

`gv.optimal(test)` outputs the optimal spectral gap and an associated (non-unique) optimal measurement scheme. The number of settings may be reduced somewhat by using `gv.minSet(test,ν)` given a fixed spectral gap ν. (Minimizing the number of settings is equivalent to minimizing an l_0 norm which is not convex, so one has to rely on a [heuristic](https://www.cvxpy.org/examples/applications/sparse_solution.html#iterative-log-heuristic) which does not gaurantee that the minimum found is global.)

The subroutine `equiprob(adm, test, k, 'I')` outputs a valid  verification scheme consisting of $k$ Pauli measurement settings (if it exists) with equal probabilities. Replace `I` by `X`, `Y`, or `Z` to exclude the respective Pauli measurement.

The minimum number of measurement settings required to verify a graph state $|G\rangle$, can be found by iterating `equiprob` over k up to the minimum chromatic number of G under local Clifford transformations. (See ection VII of the [paper](https://arxiv.org/abs/2007.09713).) If the choice of Pauli measurements for each party is limited to two, use `localCover2(G)` instead.

(In principle, k is at most the chromatic number of the graph, but I use [SageMath](https://doc.sagemath.org/html/en/reference//graphs/index.html) to find the chromatic number and did not include the algorithm in the module.)
