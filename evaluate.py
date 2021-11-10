import numpy as np
import sys
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
i = 0
j = 256
for i in range(5, j):
    print('img number: ', i + 1)
    selected_events = np.load("results/"+str(j)+"/selected_events/selected_no_"+str(j)+"_" + str(i) + ".npy", )
    print(sys.getsizeof(selected_events))
    selected_events = selected_events.astype('int32')
    print(sys.getsizeof(selected_events))
    clustering = np.load("results/"+str(j)+"/clusters/cluster_no_"+str(j)+"_" + str(i) + ".npy")
    #score = silhouette_score(adMat, clustering)

    #print(silhouette_score(adMat, clustering))
    print(silhouette_score(selected_events, clustering))
    break

