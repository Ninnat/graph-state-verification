# graph-state-verification
 Package for verification of quantum graph states

`graph_verify.py` is a Python module for finding a measurement scheme to efficiently verify a quantum graph state of arbitrary dimension.

# Usage and example

The input to the module is the adjacency matrix $A$ of a graph state $|G\rangle$ as a `numpy` array. The adjacency matrices of some well known families of graph states are provided:
- `GHZ(n)`: $n$-qubit GHZ states
- `cluster1D(n)`, $n$-qubit linear cluster states
- `ring(n)`, $n$-qubit ring cluster states

Up to seven qubits, there are in total [45 local-unitary-equivalent classes](https://arxiv.org/abs/quant-ph/0307130) of genuinely entangled graph states. Their graphs are provided as `graphState(k)` for $k=1$ to 45 in `graph_library.py` in the native `networkx` format, which can be turned into adjacency matrices via `networkx.to_numpy_array`.

The main routine `admissible` outputs Pauli strings for all $m$ admissible Pauli measurement settings for $|G\rangle$ (see the description in the paper) and their associated $(2^n-1)$-by-$m$ test matrix $\mathcal{A}$. The matrix element $\mathcal{A}_{jk}$ is the eigenvalue in the $(j+1)$th graph-basis state (excluding the first one which is always 1) of the $k$th admissible measurement setting.
```
import graph_verify as gv
import networkx as nx

graph = graphState(k)
A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
[adm, test] = gv.admissible(A)
```

`gv.optimal(test)` outputs the optimal spectral gap and an associated (non-unique) optimal measurement scheme. The number of settings may be reduced somewhat by using `gv.minSet(test,nu)` given a fixed spectral gap $\nu$. (Minimizing the number of settings is equivalent to minimizing an $l_0$ norm which is not convex, so one has to rely on a [heuristic](https://www.cvxpy.org/examples/applications/sparse_solution.html#iterative-log-heuristic) which does not gaurantee that the minimum found is global.)

The subroutine `equiprob(adm, test, k, 'I')` outputs a valid  verification scheme consisting of $k$ Pauli measurement settings (if it exists) with equal probabilities. Replace `I` by `X`, `Y`, or `Z` to exclude the respective Pauli measurement.

The minimum number of measurement settings required to verify a graph state $|G\rangle$, can be found by iterating `equiprob` over all $2 \le k \le \chi_{\mathrm{LC}}(G) \le n$, where $\chi_{\mathrm{LC}}(G)$ is the minimum chromatic number of $G$ under local Clifford transformations. (See the paper.) If the choice of Pauli measurements for each party is limited to two, use `localCover2(G)` instead.

(In principle, $k$ is at most the chromatic number of the graph, but I use [SageMath](https://doc.sagemath.org/html/en/reference//graphs/index.html) to find the chromatic number and the fractional chromatic number, and did not include the algorithms in `graph_verify.py`.)
