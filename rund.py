import os
import matplotlib.pyplot as plt
import random
import h5py
import numpy as np
import warnings
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore', '.*Graph is not fully connected*')
print('reading Cars_sequence...')
file_name = "Object Motion Data (mat files)/Cars_sequence.mat"
f = h5py.File(file_name, "r")
davis = f['davis']
dvs = davis['dvs']
pol = dvs['p'][0]
#ts = dvs['t'][0]
ts = np.load("street_ts.npy")
ts = ts*0.000001
#x = dvs['x'][0]
#y = dvs['y'][0]
aps_ts = np.load("street_img_ts.npy")
#dvs_ts = np.load("cars_all_ts.npy")
print(len(ts), len(aps_ts))

'''
# events frequency distribution

y_eve = []
i = 0
ctr = 0
j = 0
while i<len(ts):
    if ts[i] < aps_ts[j]:
        ctr += 1
    else:
        y_eve.append(ctr)
        ctr = 1
        j += 1
        if j==len(aps_ts):
            break
    i += 1
np.save("event_dist_street.npy", np.asarray(y_eve))
'''


# plot frequency distribution

y_eve = np.load("event_dist_street.npy")
print(y_eve)
print(len(y_eve))
fig = plt.figure()
plt.bar(range(200), y_eve, color='r')
plt.xlabel("Segments")
plt.ylabel("No. of events")
plt.title("Frequency of events in different segments")
plt.show()
print(sum(y_eve))


'''
#without cleaning

n = len(dvs_ts)
last = 0
ALL = len(pol)
NEIGHBORS = 100
ctr = -1
for idx in dvs_ts:

    ctr+=1

    xx = '0000000000'
    yy = str(ctr)
    file_name = xx[:len(xx) - len(yy)] + yy

    print(last)
    selected_events = []
    for i in range(0, ALL)[last:idx]:
        selected_events.append([y[i], x[i], ts[i] * 0.0001, pol[i] * 0])
        if len(selected_events)==6000:
            break
    last = idx

    selected_events = np.asarray(selected_events)
    cleaned_events = IsolationForest(random_state=0, n_jobs=-1, contamination=0.05).fit(selected_events)
    unwanted_events = cleaned_events.predict(selected_events)
    selected_events = selected_events[np.where(unwanted_events == 1, True, False)]


    adMat = kneighbors_graph(selected_events, n_neighbors=NEIGHBORS)
    max_score = -20
    opt_clusters = 2
    scores = []
    print('predicting number of clusters...')
    for CLUSTERS in range(2, 10):
        clustering = SpectralClustering(n_clusters=CLUSTERS, random_state=0,
                                        affinity='precomputed_nearest_neighbors',
                                        n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                        n_jobs=-1).fit_predict(adMat)
        curr_score = silhouette_score(selected_events, clustering)
        scores.append(curr_score)
        if curr_score > max_score:
            max_score = curr_score
            opt_clusters = CLUSTERS

    np.save(os.path.join('results/656/predict_k',
                         file_name + '.npy'),
            np.asarray(scores))
    clustering = SpectralClustering(n_clusters=opt_clusters, random_state=0, affinity='precomputed_nearest_neighbors',
                                            n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                            n_jobs=-1).fit_predict(adMat)


    np.save(os.path.join('results/656/selected_events',
                         file_name + '.npy'),
            selected_events)
    np.save(os.path.join('results/656/clusters',
                         file_name + '.npy'),
            clustering)
print('done')
'''


'''
# indices of nearest timestamps

event_idx = []
for t in aps_ts:
    idx_t = (np.abs(ts - t)).argmin()
    print(t)
    event_idx.append(idx_t)
event_idx = np.asarray(event_idx)
np.save("cars_all_ts.npy", event_idx)
print(len(event_idx))
'''

'''
#with cleaning and cluster prediction

ALL = len(pol)
NEIGHBORS = 30
print(str(ALL)+' events in dataset...')
seg = 64
while seg >= 64:
    print('dividing the sequence into '+str(seg)+' segments...')
    X = ALL//seg
    print('each segment has '+str(X)+' events, out of which '+str(X//4)+' events will be selected...')
    for sl_no in range(seg):
        print('segment no: '+str(sl_no+1))
        selected_events = []
        for i in range(0,ALL)[sl_no*X:sl_no*X+X:4]:
            selected_events.append([y[i], x[i], ts[i]*0.0001, pol[i]*0])
        selected_events = np.asarray(selected_events)
        cleaned_events = IsolationForest(random_state=0, n_jobs=-1, contamination=0.1).fit(selected_events)
        unwanted_events = cleaned_events.predict(selected_events)
        selected_events_cleaned = selected_events[np.where(unwanted_events == 1, True, False)]

        adMat_cleaned = kneighbors_graph(selected_events_cleaned, n_neighbors=NEIGHBORS)
        print('clustering...')
        clustering_cleaned = SpectralClustering(n_clusters=2, random_state=0, affinity='precomputed_nearest_neighbors',
                                        n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                        n_jobs=-1).fit_predict(adMat_cleaned)

        xx = '0000000000'
        yy = str(sl_no)
        file_name = xx[:len(xx) - len(yy)] + yy
        np.save(os.path.join('results/clean/64/selected_events',
                             file_name+'.npy'),
                selected_events_cleaned)
        np.save(os.path.join('results/clean/64/clusters',
                             file_name + '.npy'),
                clustering_cleaned)
    seg = seg // 2
    break
print('done')
'''

