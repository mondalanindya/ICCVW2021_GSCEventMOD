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
warnings.filterwarnings('ignore', '.*number of connected components of the connectivity*')
print('reading Cars_sequence...')
file_name = "Object Motion Data (mat files)/Cars_sequence.mat"
f = h5py.File(file_name, "r")
davis = f['davis']
dvs = davis['dvs']
pol = dvs['p'][0]
ts = dvs['t'][0]
x = dvs['x'][0]
y = dvs['y'][0]
aps_ts = np.load("cars_img_ts.npy")
dvs_ts = np.load("cars_all_ts.npy")
to_be_ignored = [182, 183, 293, 325, 562, 591, 592, 593]
n = len(dvs_ts)
last = 0
ALL = len(pol)
NEIGHBORS = 100

for i in range(1, len(dvs_ts)):
    if to_be_ignored.count(i) > 0:
        continue
    xx = '0000000000'
    yy = str(i)
    file_name = xx[:len(xx) - len(yy)] + yy

    print('img : ', i)
    selected_events = []
    last = dvs_ts[i-1] + 1 if i>0 else 0
    idx  = dvs_ts[i]
    #for i in range(0, ALL)[last:idx]:
    #    selected_events.append([y[i], x[i], ts[i] * 0.0001, pol[i] * 0])
    selected_events = np.load("results/200/selected_events/" + file_name + ".npy")
    print(selected_events.dtype)
    print(len(selected_events))
    #selected_events = np.asarray(selected_events)

    print('removing noise...')
    #cleaned_events = IsolationForest(random_state=0, n_jobs=-1, contamination=0.05).fit(selected_events)
    #unwanted_events = cleaned_events.predict(selected_events)
    #selected_events = selected_events[np.where(unwanted_events == 1, True, False)]

    print('graph construction...')
    adMat = kneighbors_graph(selected_events, n_neighbors=NEIGHBORS)

    max_score_sc = -20
    #max_score_km = -20
    max_score_gmm = -20
    #max_score_agc = -20

    opt_clusters_sc = 2
    #opt_clusters_km = 2
    opt_clusters_gmm = 2
    #opt_clusters_agc = 2

    scores_sc = []
    #scores_km = []
    scores_gmm = []
    #scores_agc = []


    print('predicting number of clusters...')
    for CLUSTERS in range(2, 7):
        clustering_sc = SpectralClustering(n_clusters=CLUSTERS, random_state=0,
                                        affinity='precomputed_nearest_neighbors',
                                        n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                        n_jobs=1).fit_predict(adMat)

        #clustering_km = KMeans(n_clusters=CLUSTERS, random_state=0).fit_predict(selected_events)

        #clustering_gmm = GaussianMixture(n_components=CLUSTERS,
        #                                 random_state=0).fit_predict(selected_events)

        #clustering_agc = AgglomerativeClustering(n_clusters=CLUSTERS, linkage='ward',
        #                                          connectivity=adMat).fit_predict(selected_events)
        #print(silhouette_score(selected_events, clustering_sc * 0))
        #exit()
        scores_sc.append(silhouette_score(selected_events, clustering_sc))

        #scores_km.append(silhouette_score(selected_events, clustering_km))
        #scores_gmm.append(silhouette_score(selected_events, clustering_gmm))
        #scores_agc.append(silhouette_score(selected_events, clustering_agc))

        if scores_sc[-1] > max_score_sc:
            max_score_sc = scores_sc[-1]
            opt_clusters_sc = CLUSTERS
        #if scores_km[-1] > max_score_km:
        #    max_score_km = scores_km[-1]
        #    opt_clusters_km = CLUSTERS
        #if scores_gmm[-1] > max_score_gmm:
        #    max_score_gmm = scores_gmm[-1]
        #    opt_clusters_gmm = CLUSTERS
        #if scores_agc[-1] > max_score_agc:
        #    max_score_agc = scores_agc[-1]
        #    opt_clusters_agc = CLUSTERS
    print(max_score_sc)
    print('clustering...')
    clustering_sc = SpectralClustering(n_clusters=opt_clusters_sc, random_state=0,
                                       affinity='precomputed_nearest_neighbors',
                                       n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                       n_jobs=-1).fit_predict(adMat)

    #clustering_km = KMeans(n_clusters=opt_clusters_km, random_state=0).fit_predict(selected_events)
    '''
    clustering_gmm = GaussianMixture(n_components=opt_clusters_gmm,
                                     random_state=0).fit_predict(selected_events)

    #clustering_agc = AgglomerativeClustering(n_clusters=opt_clusters_agc, linkage='ward',
    #                                         connectivity=adMat).fit_predict(selected_events)

    clustering_db = DBSCAN(eps=10, min_samples=NEIGHBORS).fit_predict(selected_events)

    BW = estimate_bandwidth(selected_events)
    clustering_ms = MeanShift(bandwidth=BW).fit_predict(selected_events)
    '''
    print('saving results...')
    '''
    np.save(os.path.join('results/656/predict_k/spectral',
                         file_name + '.npy'),
            np.asarray(scores_sc))
    #np.save(os.path.join('results/656/predict_k/kmeans',
    #                     file_name + '.npy'),
    #        np.asarray(scores_km))
    np.save(os.path.join('results/656/predict_k/gmm',
                         file_name + '.npy'),
            np.asarray(scores_gmm))
    #np.save(os.path.join('results/656/predict_k/agc',
    #                     file_name + '.npy'),
    #        np.asarray(scores_agc))

    #np.save(os.path.join('results/656/selected_events',
    #                     file_name + '.npy'),
    #        selected_events)

    np.save(os.path.join('results/656/clusters/spectral',
                         file_name + '.npy'),
            clustering_sc)
    #np.save(os.path.join('results/656/clusters/kmeans',
    #                     file_name + '.npy'),
    #        clustering_km)
    np.save(os.path.join('results/656/clusters/meanshift',
                         file_name + '.npy'),
            clustering_ms)
    np.save(os.path.join('results/656/clusters/dbscan',
                         file_name + '.npy'),
            clustering_db)
    #np.save(os.path.join('results/656/clusters/agc',
    #                     file_name + '.npy'),
    #        clustering_agc)
    np.save(os.path.join('results/656/clusters/gmm',
                         file_name + '.npy'),
            clustering_gmm)
    '''
print('done')