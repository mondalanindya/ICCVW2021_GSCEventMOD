import os
import matplotlib.pyplot as plt
import random
import h5py
import numpy as np
import warnings
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore', '.*Graph is not fully connected*')
print('reading Hands_sequence...')
file_name = "Object Motion Data (mat files)/Hands_sequence.mat"
f = h5py.File(file_name, "r")
davis = f['davis']
dvs = davis['dvs']
pol = dvs['p'][0]
ts = dvs['t'][0]
x = dvs['x'][0]
y = dvs['y'][0]
aps_ts = np.load("hands_img_ts.npy")
dvs_ts = np.load("hands_all_ts.npy")
#for i in dvs_ts:
#    print(i)
#exit()
n = len(dvs_ts)
last = 0
ALL = len(pol)
NEIGHBORS = 30

for i in [66, 70, 87, 26, 101]:

    xx = '0000000000'
    yy = str(i)
    file_name = xx[:len(xx) - len(yy)] + yy

    print('img : ', i)
    selected_events = []
    last = dvs_ts[i-1] + 1 if i>0 else 0
    idx  = dvs_ts[i]
    #for i in range(0, ALL)[last:idx]:
    #    selected_events.append([y[i], x[i], ts[i] * 0.0001, pol[i] * 0])
    #    if len(selected_events)>=116000:
    #        break
    selected_events = np.load("results/190/selected_events/" + file_name + ".npy")
    #selected_events = np.asarray(selected_events)

    print('removing noise...')
    #cleaned_events = IsolationForest(random_state=0, n_jobs=-1, contamination=0.05).fit(selected_events)
    #unwanted_events = cleaned_events.predict(selected_events)
    #selected_events = selected_events[np.where(unwanted_events == 1, True, False)]

    print('graph construction...')
    adMat = kneighbors_graph(selected_events, n_neighbors=NEIGHBORS)
    max_score = -20
    opt_clusters = 2
    scores = []
    all_clusters = []

    print('predicting number of clusters...')
    for CLUSTERS in range(2, 6):
        clustering = SpectralClustering(n_clusters=CLUSTERS, random_state=0,
                                        affinity='precomputed_nearest_neighbors',
                                        n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                        n_jobs=-1).fit_predict(adMat)
        all_clusters.append(clustering)
        curr_score = silhouette_score(selected_events, clustering)
        scores.append(curr_score)
        if curr_score > max_score:
            max_score = curr_score
            opt_clusters = CLUSTERS


    print('clustering...')
    #clustering = SpectralClustering(n_clusters=opt_clusters, random_state=0,
    #                                affinity='precomputed_nearest_neighbors',
    #                                n_neighbors=NEIGHBORS, assign_labels='kmeans',
    #                                n_jobs=-1).fit_predict(adMat)

    #clustering_kmeans = KMeans(n_clusters=opt_clusters, random_state=0).fit_predict(selected_events)

    #BW = estimate_bandwidth(selected_events)

    #clustering_meanshift = MeanShift(bandwidth=BW).fit_predict(selected_events)

    #clustering_dbscan = DBSCAN(eps=10, min_samples=NEIGHBORS).fit_predict(selected_events)

    #clustering_aggc = AgglomerativeClustering(n_clusters=opt_clusters, linkage='ward', connectivity=adMat).fit_predict(selected_events)

    #clustering_gmm = GaussianMixture(n_components=opt_clusters, random_state=0).fit_predict(selected_events)
    print('saving results...')
    #np.save(os.path.join('results/190/predict_k',
    #                     file_name + '.npy'),
    #        np.asarray(scores))
    #np.save(os.path.join('results/190/selected_events',
    #                     file_name + '.npy'),
    #        selected_events)
    #np.save(os.path.join('results/190/clusters/spectral',
    #                     file_name + '.npy'),
    #        clustering)
    np.save(os.path.join('results/190/for/clusters',
                         file_name+'.npy'),
            all_clusters)
    #np.save(os.path.join('results/190/clusters/kmeans',
    #                     file_name + '.npy'),
    #        clustering_kmeans)
    #np.save(os.path.join('results/190/clusters/meanshift',
    #                     file_name + '.npy'),
    #        clustering_meanshift)
    #np.save(os.path.join('results/190/clusters/dbscan',
    #                     file_name + '.npy'),
    #        clustering_dbscan)
    #np.save(os.path.join('results/190/clusters/aggc',
    #                     file_name + '.npy'),
    #        clustering_aggc)
    #np.save(os.path.join('results/190/clusters/gmm',
    #                     file_name + '.npy'),
    #        clustering_gmm)
print('done')