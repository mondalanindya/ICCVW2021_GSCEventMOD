import os
import matplotlib.pyplot as plt
import random
import h5py
import numpy as np
import warnings
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
print('reading Cars_sequence...')
file_name = "Object Motion Data (mat files)/Cars_sequence.mat"
f = h5py.File(file_name, "r")
davis = f['davis']
dvs = davis['dvs']
pol = dvs['p'][0]
ts = dvs['t'][0]
x = dvs['x'][0]
y = dvs['y'][0]
warnings.filterwarnings('ignore', '.*Graph is not fully connected*')
ALL = len(pol)
NEIGHBORS = 30
print(str(ALL)+' events in dataset...')
seg = 656
while seg >= 656:
    print('dividing the sequence into '+str(seg)+' segments...')
    X = ALL//seg
    print('each segment has '+str(X)+' events, out of which '+str(X//4)+' events will be selected...')
    for sl_no in range(seg):

        print('segment no: '+str(sl_no+1))
        selected_events = []
        for i in range(0,ALL)[sl_no*X:sl_no*X+X]:
            selected_events.append([y[i], x[i], ts[i]*0.0001, pol[i]*0])
        selected_events = np.asarray(selected_events)

        print('removing noise...')
        cleaned_events = IsolationForest(random_state=0, n_jobs=-1, contamination=0.05).fit(selected_events)
        unwanted_events = cleaned_events.predict(selected_events)
        selected_events_cleaned = selected_events[np.where(unwanted_events == 1, True, False)]

        print('constructing graph...')
        adMat_cleaned = kneighbors_graph(selected_events_cleaned, n_neighbors=NEIGHBORS)

        print('finding optimal number of clusters...')
        max_score = -20
        opt_clusters = 2
        for CLUSTERS in range(2, 10):
            clustering = SpectralClustering(n_clusters=CLUSTERS, random_state=0,
                                            affinity='precomputed_nearest_neighbors',
                                            n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                            n_jobs=-1).fit_predict(adMat_cleaned)
            curr_score = silhouette_score(selected_events_cleaned, clustering)
            if curr_score > max_score:
                max_score = curr_score
                opt_clusters = CLUSTERS

        print('clustering...')
        clustering_opt = SpectralClustering(n_clusters=opt_clusters, random_state=0,
                                            affinity='precomputed_nearest_neighbors',
                                            n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                            n_jobs=-1).fit_predict(adMat_cleaned)

        print('saving files...')
        xx = '0000000000'
        yy = str(sl_no)
        file_name = xx[:len(xx) - len(yy)] + yy
        np.save(os.path.join('results/cars/'+str(seg)+'/selected_events', file_name + '.npy'), selected_events_cleaned)
        np.save(os.path.join('results/cars/'+str(seg)+'/clusters', file_name + '.npy'), clustering_opt)
    seg = seg // 2
    break
print('done')


