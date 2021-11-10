import os
import matplotlib.pyplot as plt
import random
import h5py
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
print('reading Hands_sequence...')
file_name = "Object Motion Data (mat files)/Cars_sequence.mat"
f = h5py.File(file_name, "r")
davis = f['davis']
dvs = davis['dvs']
pol = dvs['p'][0]
ts = dvs['t'][0]
x = dvs['x'][0]
y = dvs['y'][0]

ALL = len(pol)
NEIGHBORS = 30
print(str(ALL)+' events in dataset...')
seg = 256
while seg >= 256:
    print('dividing the sequence into ' + str(seg) + ' segments...')
    X = ALL // seg
    print('each segment has ' + str(X) + ' events, out of which ' + str(X // 4) + ' events will be selected...')
    for sl_no in range(seg):
        print('segment no: ' + str(sl_no + 1))
        selected_events = []
        for i in range(0, ALL)[sl_no * X:sl_no * X + X:4]:
            selected_events.append([y[i], x[i], ts[i] * 0.0001, pol[i] * 0])
        selected_events = np.asarray(selected_events)
        adMat = kneighbors_graph(selected_events, n_neighbors=NEIGHBORS)
        print('clustering...')
        score = []
        for k in range(2, 10):
            clustering = SpectralClustering(n_clusters=k, random_state=0,
                                        affinity='precomputed_nearest_neighbors',
                                        n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                        n_jobs=-1,
                                        verbose=True).fit_predict(adMat)
            score.append(silhouette_score(selected_events, clustering))


    break
    seg = seg // 2
