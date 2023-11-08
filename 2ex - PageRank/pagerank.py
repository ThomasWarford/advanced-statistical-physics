import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

RAD = 0.3
N_ITERATIONS = 15
DAMPING_FACTOR = .85
connection_style = f'arc3,rad={RAD}'

def offset(d, pos, dist = RAD/2, loop_shift = .2):
    # from stackoverflow
    for (u,v),obj in d.items():
        if u!=v:
            par = dist*(pos[v] - pos[u])
            dx,dy = par[1],-par[0]
            x,y = obj.get_position()
            obj.set_position((x+dx,y+dy))
        else:
            x,y = obj.get_position()
            obj.set_position((x,y+loop_shift))

# Define a stochastic matrix (example matrix)
stochastic_matrix = np.array([
    [0, 0, 0, 0, 0],
    [0.25, 0, 0, 0, 1/3],
    [0.25, 0, 0, 0.5, 1/3],
    [0.25, 1, 1, 0, 1/3],
    [0.25, 0, 0, 0.5, 0],
])
num_nodes = len(stochastic_matrix)

# Run PageRank
probability_vector = np.ones(num_nodes) / num_nodes

for _ in range(N_ITERATIONS):
    probability_vector = (1-DAMPING_FACTOR)/num_nodes + DAMPING_FACTOR * stochastic_matrix @ probability_vector

node_labels = {i: f"{i}\n{p:.2f}" for i, p in enumerate(probability_vector)}

# Create the graph visualization    
## Create a directed graph
G = nx.DiGraph()

## Add nodes to the graph
G.add_nodes_from(range(num_nodes))

## Add edges to the graph based on the stochastic matrix
for i in range(num_nodes):
    for j in range(num_nodes):
        i_to_j_probability = stochastic_matrix[j][i]
        if i_to_j_probability > 0:
            G.add_edge(i, j, weight=i_to_j_probability)

## Define layout for the network
pos = nx.spring_layout(G)

## Draw the nodes and edges
nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=probability_vector*4000, connectionstyle=connection_style, node_color='lightblue', font_size=10, font_color='black', font_weight='bold')
edge_labels = {(i, j): f'{G[i][j]["weight"]:.2f}' for i, j in G.edges()}
label_dict=nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

## Offset Edge Labels so they can all be seen
offset(label_dict, pos)

# Show the network
plt.title("PageRank - d={DAMPING_FACTOR}")
plt.show()