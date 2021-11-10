import os
import sys
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import gridspec
import matplotlib.image as im
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
dbs = []
sil = []
chi = []
j = 255
to_be_ignored = [182, 183, 293, 325, 562, 591, 592, 593]
for i in range(j):
    print(i)
    if to_be_ignored.count(i) > 0:
        continue
    xx = '0000000000'
    yy = str(i)
    file_name = xx[:len(xx) - len(yy)] + yy

    selected_events = np.load("results/656/selected_events/"+file_name+".npy")

    cluster_kmeans = np.load("results/656/clusters/kmeans/"+file_name+".npy")
    cluster_meanshift = np.load("results/656/clusters/meanshift/"+file_name+".npy")
    cluster_dbscan = np.load("results/656/clusters/dbscan/"+file_name+".npy")
    cluster_spectral = np.load("results/656/clusters/spectral/"+file_name+".npy")
    cluster_aggc = np.load("results/656/clusters/aggc/" + file_name + ".npy")
    cluster_gmm = np.load("results/656/clusters/gmm/" + file_name + ".npy")
    if max(cluster_meanshift)==0:
        cluster_meanshift[0] = 1
    #print('davies_bouldin_score')
    #print('Kmeans : ', davies_bouldin_score(selected_events, cluster_kmeans))
    #print('MeanShift : ', davies_bouldin_score(selected_events, cluster_meanshift))
    #print('DBSCAN : ', davies_bouldin_score(selected_events, cluster_dbscan))
    #print('Spectral : ', davies_bouldin_score(selected_events, cluster_spectral))

    dbs.append([davies_bouldin_score(selected_events, cluster_kmeans),
                davies_bouldin_score(selected_events, cluster_meanshift),
                davies_bouldin_score(selected_events, cluster_dbscan),
                davies_bouldin_score(selected_events, cluster_spectral),
                davies_bouldin_score(selected_events, cluster_aggc),
                davies_bouldin_score(selected_events, cluster_gmm)])

    #print('silhouette_score')
    #print('Kmeans : ', silhouette_score(selected_events, cluster_kmeans))
    #print('MeanShift : ', silhouette_score(selected_events, cluster_meanshift))
    #print('DBSCAN : ', silhouette_score(selected_events, cluster_dbscan))
    #print('Spectral : ', silhouette_score(selected_events, cluster_spectral))

    #print('calinski_harabasz_score')
    #print('Kmeans : ', calinski_harabasz_score(selected_events, cluster_kmeans))
    #print('MeanShift : ', calinski_harabasz_score(selected_events, cluster_meanshift))
    #print('DBSCAN : ', calinski_harabasz_score(selected_events, cluster_dbscan))
    #print('Spectral : ', calinski_harabasz_score(selected_events, cluster_spectral))

    chi.append([calinski_harabasz_score(selected_events, cluster_kmeans),
                calinski_harabasz_score(selected_events, cluster_meanshift),
                calinski_harabasz_score(selected_events, cluster_dbscan),
                calinski_harabasz_score(selected_events, cluster_spectral),
                calinski_harabasz_score(selected_events, cluster_aggc),
                calinski_harabasz_score(selected_events, cluster_gmm)])
    continue
    x = np.array(selected_events[:, 0])
    y = np.array(selected_events[:, 1])
    z = np.array(selected_events[:, 2])

    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=4, nrows=1)
    ax1 = fig.add_subplot(spec[0], aspect=1)
    ax2 = fig.add_subplot(spec[1], aspect=1)
    ax3 = fig.add_subplot(spec[2], aspect=1)
    ax4 = fig.add_subplot(spec[3], aspect=1)

    ax1.scatter(x[::], y[::], marker=".", c=cluster_kmeans[::], s=0.1, cmap='rainbow')
    ax2.scatter(x[::], y[::], marker=".", c=cluster_meanshift[::], s=0.1, cmap='rainbow')
    ax3.scatter(x[::], y[::], marker=".", c=cluster_dbscan[::], s=0.1, cmap='rainbow')
    ax4.scatter(x[::], y[::], marker=".", c=cluster_spectral[::], s=0.1, cmap='rainbow')

    ax1.set_xlim([0, 260])
    ax1.set_ylim([0, 346])
    ax2.set_xlim([0, 260])
    ax2.set_ylim([0, 346])
    ax3.set_xlim([0, 260])
    ax3.set_ylim([0, 346])
    ax4.set_xlim([0, 260])
    ax4.set_ylim([0, 346])

    plt.show()
    #fig.savefig("results/190/comp/"+file_name+".png", bbox_inches="tight")
    plt.close()
dbs = np.asarray(dbs)
chi = np.asarray(chi)
print('mean davies_bouldin_score of Kmeans : ', mean(dbs[:, 0]))
print('mean davies_bouldin_score of MeanShift : ', mean(dbs[:, 1]))
print('mean davies_bouldin_score of DBSCAN : ', mean(dbs[:, 2]))
print('mean davies_bouldin_score of Agglomerative : ', mean(dbs[:, 4]))
print('mean davies_bouldin_score of GMM : ', mean(dbs[:, 5]))
print('mean davies_bouldin_score of Spectral : ', mean(dbs[:, 3]))


print('mean calinski_harabasz_score of Kmeans : ', mean(chi[:, 0]))
print('mean calinski_harabasz_score of MeanShift : ', mean(chi[:, 1]))
print('mean calinski_harabasz_score of DBSCAN : ', mean(chi[:, 2]))
print('mean calinski_harabasz_score of Agglomerative : ', mean(chi[:, 4]))
print('mean calinski_harabasz_score of GMM : ', mean(chi[:, 5]))
print('mean calinski_harabasz_score of Spectral : ', mean(chi[:, 3]))