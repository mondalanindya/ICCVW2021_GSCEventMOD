import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy import sparse

NEIGHBORS = 100
TOTAL = 656

NC = []

to_be_ignored = [182, 183, 293, 325, 562, 591, 592, 593]
for i in range(TOTAL):
    if to_be_ignored.count(i) > 0:
        NC.append(0)
        continue
    xx = '0000000000'
    yy = str(i)
    file_name = xx[:len(xx) - len(yy)] + yy
    cluster = np.load("results/656/clusters/meanshift/"+file_name+".npy")
    NC.append(max(cluster) + 1)

df = pd.DataFrame (NC)
filepath = 'NC_meanshift.xlsx'
df.to_excel(filepath, index=False)


'''
def generate_graph_laplacian(df, nn):
    """Generate graph Laplacian from data."""
    # Adjacency Matrix.
    connectivity = kneighbors_graph(X=df, n_neighbors=nn, mode='connectivity')
    adjacency_matrix_s = (1/2)*(connectivity + connectivity.T)
    # Graph Laplacian.
    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=adjacency_matrix_s, normed=False)
    graph_laplacian = graph_laplacian_s.toarray()
    return graph_laplacian

for i in range(19, 30):
    xx = '0000000000'
    yy = str(i)
    file_name = xx[:len(xx) - len(yy)] + yy
    selected_events = np.load("results/656/selected_events/"+file_name+".npy")
    print(file_name, len(selected_events))
    L = generate_graph_laplacian(df=selected_events, nn=NEIGHBORS)

    vals, vecs = np.linalg.eigh(L)
    # real eval and evec
    sorted_vals = np.sort(vals)

    plt.plot(sorted_vals[:10])
    plt.grid()
    plt.savefig("results/"+str(0)+file_name+".png")
    plt.close()
'''

