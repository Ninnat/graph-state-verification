import networkx as nx

lib = [[] for k in range(46)] # list of empty lists
l = 0
while l <= 45:
    lib[l] = nx.Graph()
    l += 1
lib[0].add_node(1) # 1 qubit
lib[1].add_edges_from([(1,2)]) # 2 qubits
lib[2].add_edges_from([(1,2),(1,3)]) # 3 qubits GHZ state = cluster state
lib[3].add_edges_from([(1,2),(1,3),(1,4)]) # 4 qubit GHZ state 
lib[4].add_edges_from([(1,2),(2,3),(3,4)]) # cluster state

# 5 qubits

lib[5].add_edges_from([(1,2),(1,3),(1,4),(1,5)]) # GHZ state
lib[6].add_edges_from([(1,2),(2,3),(3,4),(2,5)])
lib[7].add_edges_from([(1,2),(2,3),(3,4),(4,5)]) # cluster state
lib[8].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,1)]) # ring state

# 6 qubits

lib[9].add_edges_from([(1,2),(1,3),(1,4),(1,5),(1,6)]) # GHZ state
lib[10].add_edges_from([(1,2),(2,3),(3,4),(2,5),(2,6)])
lib[11].add_edges_from([(1,2),(2,3),(3,4),(2,5),(3,6)])
lib[12].add_edges_from([(1,2),(2,3),(3,4),(4,5),(2,6)])
lib[13].add_edges_from([(1,2),(2,3),(3,4),(4,5),(3,6)])
lib[14].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6)]) # cluster state
lib[15].add_edges_from([(1,2),(2,3),(3,4),(4,5),(2,6),(4,6)])
lib[16].add_edges_from([(1,2),(2,3),(3,1),(1,4),(2,5),(3,6)])
lib[17].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(5,1)])
lib[18].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(6,1)]) # ring state
lib[19].add_edges_from([(1,2),(2,3),(3,6),(6,5),(5,4),(4,1),(1,3),(4,6),(2,5)])

# 7 qubits

lib[20].add_edges_from([(1,2),(1,3),(1,4),(1,5),(1,6),(1,7)]) # GHZ state
lib[21].add_edges_from([(1,7),(7,6),(6,5),(7,2),(7,3),(7,4)]) # line-4 skeleton
lib[22].add_edges_from([(1,7),(7,6),(6,5),(7,2),(7,3),(6,4)])
lib[23].add_edges_from([(1,7),(7,6),(6,5),(7,2),(7,3),(5,4)])
lib[24].add_edges_from([(1,7),(7,6),(6,5),(5,4),(7,2),(5,3)])
lib[25].add_edges_from([(2,1),(1,7),(7,6),(6,5),(7,3),(7,4)])
lib[26].add_edges_from([(1,7),(7,6),(6,5),(5,4),(7,2),(6,3)])
lib[27].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(2,7)])
lib[28].add_edges_from([(1,2),(2,3),(3,4),(5,6),(6,7),(3,5)])
lib[29].add_edges_from([(1,2),(2,3),(3,4),(4,5),(3,6),(6,7)])
lib[30].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]) # cluster state
lib[31].add_edges_from([(1,2),(2,3),(3,4),(4,1),(1,5),(1,6),(3,7)]) #
lib[32].add_edges_from([(1,7),(7,6),(6,5),(5,4),(5,7),(7,2),(6,3)])
lib[33].add_edges_from([(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(3,7)])
lib[34].add_edges_from([(1,4),(2,3),(3,4),(4,5),(5,6),(6,7),(3,6)])
lib[35].add_edges_from([(1,6),(2,3),(3,4),(4,5),(5,6),(6,7),(3,7)])
lib[36].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(3,5),(4,7)])
lib[37].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(2,6)])
lib[38].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(1,6)])
lib[39].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(1,5)])
lib[40].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,1)]) # ring state
lib[41].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(1,5),(1,6)])
lib[42].add_edges_from([(2,3),(3,4),(4,5),(5,6),(6,7),(7,1),(1,3),(2,6)])
lib[43].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(7,1),(1,4),(7,5),(3,6)])
lib[44].add_edges_from([(2,3),(3,4),(4,5),(5,6),(6,7),(7,1),(7,2),(1,4),(3,5)]) #
lib[45].add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,2),(2,5),(3,7),(4,6)])

def graphState(j):
    return(lib[j])