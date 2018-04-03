import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

X = np.array([[1, 2],[1.5, 1.8],[5, 8],[8, 8],[1, 0.6],[9, 11],[1,3],[8,9],[0,3],[5,4],[6,4]])

class K_Means:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    
    def fit(self,data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        for i in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
                
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classified = distances.index(min(distances))
                self.classifications[classified].append(featureset)
                
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
                
            optimized = True
            
            for centroid in self.centroids:
                pct_change = np.sum((-prev_centroids[centroid]+self.centroids[centroid])*100/prev_centroids[centroid])
                if pct_change > self.tol:
                    print(pct_change)
                    optimized = False
            if optimized:
               break                
    
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classified = distances.index(min(distances))
        return classified

clf = K_Means()
clf.fit(X)
colors = ['b','g']
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],c='k',marker='x')
     
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
           plt.scatter(featureset[0],featureset[1],c=color)
#unknowns = np.array([[1,3],[8,9],[0,3],[5,4],[6,4],])
#for unknown in unknowns:
#    classification = clf.predict(unknown)
#    plt.scatter(unknown[0],unknown[1],c=colors[classification],marker='*')
           
plt.show()     
