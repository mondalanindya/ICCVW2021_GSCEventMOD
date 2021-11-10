import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from PIL import Image
from matplotlib import gridspec
import matplotlib.image as im
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import networkx as nx

'''
aps_ts = np.load("hands_img_ts.npy")
ts = np.load("hands_all_ts.npy")
print(len(aps_ts))
print(len(ts))
'''

j = 190
#to_be_ignored = [182, 183, 293, 325, 562, 591, 592, 593]
for i in [66, 70, 87, 26, 101]:
    #if to_be_ignored.count(i) > 0:
    #    continue
    print('img number: ', i + 1)
    xx = '0000000000'
    yy = str(i)
    file_name = xx[:len(xx) - len(yy)] + yy
    #selected_events = selected_events_all[last:event_idx[i]]
    #I = Image.open("raw_img/Cars_sequence/"+file_name+".jpeg")
    #I = I.rotate(-90, Image.NEAREST)
    #I.save("raw_img/Cars_sequence/"+file_name+".jpeg")
    #I.close()
    #continue

    #img = im.imread("raw_img/Street_sequence/"+file_name+".jpeg")
    #print(img.shape)
    selected_events = np.load("results/"+str(j)+"/selected_events/"+file_name+".npy")
    #G = nx.k_nearest_neighbors(selected_events)
    #G = kneighbors_graph(selected_events[:20], n_neighbors=3).toarray()
    #print(G.toarray())
    #G = nx.from_numpy_matrix(np.array(G))
    #nx.draw(G)
    #nx.draw(G, with_labels=True)
    #print(selected_events.dtype)
    #print(len(selected_events))
    #print(selected_events.shape)
    #selected_events = np.load("results/64/selected_events/selected_no_64_0.npy")
    #clustering = clustering_all[last:event_idx[i]]
    #last = event_idx[i]
    clustering = np.load("results/"+str(j)+"/for/clusters/"+file_name+".npy")

    for num in range(4):

        #clustering = clustering*0
        #clustering = np.load("results/64/clusters/cluster_no_64_0.npy")
        #clustering = np.ones(clustering.shape)
        #print(clustering)
        x = np.array(selected_events[:, 0])
        #x = 260-x
        y = np.array(selected_events[:, 1])
        #y = 346-y
        z = np.array(selected_events[:, 2])
        #points = [x[np.where(clustering == 1, True, False)], y[np.where(clustering == 1, True, False)]]
        #points = np.asarray(points)
        #print(points.T, points.T.shape)
        #points = np.random.default_rng().random((30,2))
        #print(points, points.shape)
        #hull0 = ConvexHull(points.T)
        #print(hull0)
        #x1 = np.array(selected_clean_events[:, 0])
        #y1 = np.array(selected_clean_events[:, 1])
        #z1 = np.array(selected_clean_events[:, 2])
        #all_ts.append(np.mean(z)*10000)
        #print(len(x), len(x1))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #ax0 = fig.add_subplot(131)
        #spec = gridspec.GridSpec(ncols=2, nrows=1)
        #ax1 = fig.add_subplot(spec[0])
        #ax2 = fig.add_subplot(spec[1], aspect=1)
        #ax0.plot()
        #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1, 1]))
        if num==0:
            col = 'PiYG'
        elif num==1:
            col = 'rainbow'
        elif num==2:
            col = 'rainbow'
        elif num==3:
            col = 'tab10_r'
        ax.scatter(y[::], x[::], z[::], marker="o", c=1-clustering[num][::], s=0.5,
                   cmap=col)
        #print(clustering)
        #plt.xlabel("x")
        #plt.ylabel("y")
        #ax.imshow(img, extent=[0, 780, 0, 466])
        #ax2.scatter(x, y, marker='.', c=clustering, s=0.1, cmap='gray')

        ax.view_init(90, 0)
        #ax.tick_params(axis='both', which='major', labelsize=5)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0,0)
        #ax2.view_init(90, 0)

        #ax.set_xlim([0, 260])
        #ax.set_ylim([0, 346])
        #plt.axis('off')
        #plt.plot(points[:,1], points[:,0])
        #plt.plot(selected_events[hull0.vertices,0], selected_events[hull0.vertices,1],'g--',lw=2)

        #ax2.set_xlim([0, 260])
        #ax2.set_ylim([0, 346])

        #plt.show()
        fig.savefig("results/"+str(j)+"/for/images/"+file_name+'_'+str(num)+".png", transparent=True)
        plt.close()


"""
import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import kneighbors_graph


def generate_random_3Dgraph(n_nodes, radius, seed=None):
    if seed is not None:
        random.seed(seed)

    # Generate a dict of positions
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}
    print(pos)
    pot = np.load("results/190/selected_events/0000000002.npy")
    pot = pot[:,[0,1,2]]
    pos = {i: (pot[i][0], pot[i][1], pot[i][2]) for i in range(len(pot[:50]))}
    print(pos)
    # Create random 3D network
    #G = nx.random_k_out_graph(len(pot[:50]), 1, 0.7, self_loops=True, seed=None)
    G = nx.random_geometric_graph(len(pot[:50]), radius, pos=pos)

    return G


def network_plot_3D(G, angle, save=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure()
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, s=20 + 20 * G.degree(0), edgecolors='k', alpha=0.5)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    if save is not False:
        plt.show()
        #plt.savefig("C:\scratch\\data\"+str(angle).zfill(3)+".png")
        plt.close('all')
    else:
        plt.show()

    return

n=100
G = generate_random_3Dgraph(n_nodes=n, radius=15, seed=1)
#selected_events = np.load("results/190/selected_events/0000000002.npy")
#selected_events = selected_events[:,[0,1,2]]
#G = kneighbors_graph(selected_events, n_neighbors=100)

network_plot_3D(G,0, save=False)
"""