import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])

class Mean_Shift:
    def __init__(self,radius=4):
        self.radius = radius
        
    def fit(self,data):
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]
          
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)
                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))
            uniques = sorted(list(set(new_centroids)))
            optimized = True
            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            
            for i in centroids:
                if not np.array_equal(centroids[i],prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            if optimized:
                break
        self.centroids = centroids     

ms = Mean_Shift()
ms.fit(X)
centroids = ms.centroids
print(centroids)
for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],c='k',marker='x')

plt.scatter(X[:,0],X[:,1])
plt.show()    
    
            
                