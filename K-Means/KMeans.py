import numpy as np
import matplotlib.pyplot as plt
import copy

#2-dim data range in (0,100)
K=4
random_data=5000*np.random.rand(500,2)
init_center=5000*np.random.rand(K,2)
COLOR_PANEL=['#CD853F','#FF69B4','#00CED1','#00BFFF','#008B8B','#FF8C00','#2F4F4F','#FFFACD','#00FF00','#FFE4E1']
def BelongGroup(data,centers):
    mindis=float('inf')
    group=0
    for index in range(0,centers.shape[0]):
        if np.sqrt(np.square(centers[index][0]-data[0])+np.square(centers[index][1]-data[1]))<mindis:
            mindis=np.sqrt(np.square(centers[index][0]-data[0])+np.square(centers[index][1]-data[1]))
            group=index
    return group
def KMeans(dataset,K,init_centers,iterations=10):
    centers=copy.deepcopy(init_centers)
    group=[]
    for i in range(0,K):
        group.append([])
    for i in range(0,iterations):
        group=[]
        for i in range(0,K):
            group.append([])
        for data_index in range(dataset.shape[0]):
            belongs=BelongGroup(dataset[data_index],centers)
            group[belongs].append(dataset[data_index])
            centers[belongs]=centers[belongs]+1/(len(group[belongs])+1)*(dataset[data_index]-centers[belongs])
    return centers,group
centers,group=KMeans(random_data,K,init_center,100)

fig = plt.figure()
ax = fig.add_subplot(1,3,1)
ax.scatter(random_data[:,0], random_data[:,1])
bx = fig.add_subplot(1,3,2)
for i in range(0,len(group)):
    points=np.array(group[i])
    bx.scatter(points[:,0],points[:,1],c=COLOR_PANEL[i])
    bx.scatter(centers[i,0],centers[i,1],c=COLOR_PANEL[i],marker='x')
cx=fig.add_subplot(1,3,3)
for i in range(0,init_center.shape[0]):
    l1=cx.scatter(init_center[i,0],init_center[i,1],c=COLOR_PANEL[i],marker='o')
    l2=cx.scatter(centers[i,0],centers[i,1],c=COLOR_PANEL[i],marker='x')
    cx.plot([init_center[i,0],centers[i,0]],[init_center[i,1],centers[i,1]],c=COLOR_PANEL[i])
plt.xlim((0, 5000))
plt.ylim((0, 5000))
plt.legend(handles=[l1,l2],labels=['start_center','end_center'])
plt.show()






