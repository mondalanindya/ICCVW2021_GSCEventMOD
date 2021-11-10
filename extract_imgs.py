import os
import numpy as np
import h5py
from PIL import Image as im
file_name = "Object Motion Data (mat files)/Cars_sequence.mat"
f = h5py.File(file_name, "r")
#events = f['events']
#frame  = f['frame']
#frame_idx = f['frame_idx']
#frame_ts = f['frame_ts']
'''
x = events[:,1]
y = events[:,2]
ts= events[:,0]
pol=events[:,3]
'''
davis = f['davis']
aps = davis['aps']
for key in aps.keys():
    print(key)
frame = aps['frame']
size = aps['size']
t = aps['t']
'''
#ts = []
#for x in t:
#    ts.append(x)
#ts = np.asarray(ts)
#print(len(ts), ts)
#np.save("cars_img_ts.npy", ts)
ts = np.load("cars_img_ts.npy")
print(ts)
print(len(ts))
for curr_t in ts:
    idx_t = min(range(len(t)), key=lambda i: abs(t[i]-curr_t))
    to_save.append(idx_t)
    #print(curr_t, t[idx_t])
'''

for i in range(len(frame)):
    print("img no : ", i)
    I = im.fromarray(frame[i])
    I = I.rotate(-90, im.NEAREST, expand=1)
    I = I.convert('RGB')
    x = '0000000000'
    y = str(i)
    file_name = x[:len(x) - len(y)] + y
    I.save("raw_img/Cars_sequence/"+file_name+".jpeg")
