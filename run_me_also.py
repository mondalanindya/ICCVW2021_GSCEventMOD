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
print('reading Street_sequence...')
file_name = "Object Motion Data (mat files)/Street_sequence.h5"
f = h5py.File(file_name, "r")
#events = f['events']
#frame  = f['frame']
frame_idx = f['frame_idx']
frame_ts = f['frame_ts']
#x = events[:,1]
x = np.load("street_x.npy").astype(int)
#y = events[:,2]
y = np.load("street_y.npy").astype(int)
#ts= events[:,0]
ts = np.load("street_ts.npy")
ts = ts*0.000001
print(ts)
#pol=events[:,3]
pol = np.load("street_pol.npy").astype(int)
#np.save("street_img_ts.npy", np.asarray(frame_ts)*0.000001)
dvs_ts = np.asarray(frame_idx, dtype=np.int32)

n = len(dvs_ts)
last = 0
ALL = len(pol)
NEIGHBORS = 25
j = len(dvs_ts)
for i in range(1, len(dvs_ts)):

    xx = '0000000000'
    yy = str(i)
    file_name = xx[:len(xx) - len(yy)] + yy

    print('img : ', i)
    selected_events = []
    last = dvs_ts[i-1] if i>0 else 0
    idx  = dvs_ts[i]
    #print('Total : ', idx-last+1)
    jump = (idx-last+1)//3500 if (idx-last+1) > 3500 else 1
    #for i in range(0, ALL)[last:idx:jump]:
    #    selected_events.append([y[i], x[i], ts[i] * 0.0001, pol[i] * 0])
    selected_events = np.load("results/"+str(j)+"/selected_events_/" + file_name + ".npy")
    #selected_events = np.asarray(selected_events)
    #print('removing noise...')
    #cleaned_events = IsolationForest(random_state=0, n_jobs=-1, contamination=0.05).fit(selected_events)
    #unwanted_events = cleaned_events.predict(selected_events)
    #selected_events = selected_events[np.where(unwanted_events == 1, True, False)]
    #if len(selected_events) >= 3500:
    #    selected_events = selected_events[:3500]

    #print('Selected : ', len(selected_events))
    #print('graph construction...')
    adMat = kneighbors_graph(selected_events, n_neighbors=NEIGHBORS)

    #max_score_sc = -20
    max_score_gmm = -20

    #opt_clusters_sc = 2
    opt_clusters_gmm = 2

    scores_gmm = list(np.load("results/"+str(j)+"/predict_k/gmm/"+file_name+".npy"))
    opt_clusters_sc = scores_gmm.index(max(scores_gmm))+2
    #scores_gmm = []


    #print('predicting number of clusters...')

    #for CLUSTERS in range(2, 11):
        #clustering_sc = SpectralClustering(n_clusters=CLUSTERS, random_state=0,
        #                                affinity='precomputed_nearest_neighbors',
        #                                n_neighbors=NEIGHBORS, assign_labels='kmeans',
        #                                n_jobs=1).fit_predict(adMat)


        #clustering_gmm = GaussianMixture(n_components=CLUSTERS,
        #                                 random_state=0).fit_predict(selected_events)

        #ssc=silhouette_score(selected_events, clustering_sc)
        #scores_sc.append(ssc)
        #scores_gmm.append(silhouette_score(selected_events, clustering_gmm))

        #if scores_sc[-1] > max_score_sc:
        #    max_score_sc = scores_sc[-1]
        #    opt_clusters_sc = CLUSTERS
        #if scores_gmm[-1] > max_score_gmm:
        #    max_score_gmm = scores_gmm[-1]
        #    opt_clusters_gmm = CLUSTERS

    print('clustering...')
    #clustering_sc = SpectralClustering(n_clusters=opt_clusters_sc, random_state=0,
    #                                   affinity='precomputed_nearest_neighbors',
    #                                   n_neighbors=NEIGHBORS, assign_labels='kmeans',
    #                                   n_jobs=-1).fit_predict(adMat)

    #print(max(scores_sc))
    clustering_gmm = GaussianMixture(n_components=opt_clusters_gmm,
                                     random_state=0).fit_predict(selected_events)


    #clustering_db = DBSCAN(eps=5, min_samples=10).fit_predict(selected_events)

    #BW = estimate_bandwidth(selected_events)
    #clustering_ms = MeanShift(bandwidth=BW).fit_predict(selected_events)

    #print('saving results...')

    #np.save(os.path.join('results/200/predict_k/spectral',
    #                     file_name + '.npy'),
    #        np.asarray(scores_sc))
    #np.save(os.path.join('results/200/predict_k/gmm',
    #                     file_name + '.npy'),
    #        np.asarray(scores_gmm))

    #np.save(os.path.join('results/200/selected_events',
    #                     file_name + '.npy'),
    #        selected_events)

    #np.save(os.path.join('results/200/clusters/spectral',
    #                     file_name + '.npy'),
    #        clustering_sc)
    #np.save(os.path.join('results/200/clusters/meanshift',
    #                     file_name + '.npy'),
    #        clustering_ms)
    #np.save(os.path.join('results/200/clusters/dbscan',
    #                     file_name + '.npy'),
    #        clustering_db)
    np.save(os.path.join('results/200/clusters/gmm',
                         file_name + '.npy'),
            clustering_gmm)

print('done')