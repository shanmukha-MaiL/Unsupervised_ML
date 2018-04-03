import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
"""
style.use('ggplot')
centers = [[1,1,1],[5,5,5],[3,10,10]]
X,y = make_blobs(n_samples=100,centers=centers)

mf = MeanShift()
mf.fit(X)
labels = mf.labels_
cluster_centres = mf.cluster_centers_
print(cluster_centres)
n_clusters = len(np.unique(labels))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
colors = ['r','b','g']
for i in range(len(X)):
    ax.scatter(X[i][0],X[i][1],X[i][2],c=colors[labels[i]])
ax.scatter(cluster_centres[:,0],cluster_centres[:,1],cluster_centres[:,2],c='k',marker='x',s=150,linewidth=5)
plt.show()    
"""
style.use('ggplot')
df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['name','body','ticket','home.dest'],1,inplace=True)
df.fillna(0,inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
             return text_digit_vals[val]
        if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:
            elements = df[column].values.tolist()
            unique_elements = set(elements)
            x = 0
            for u in unique_elements:
                if u not in text_digit_vals:
                    text_digit_vals[u] = x
                    x+=1
            df[column] = list(map(convert_to_int,df[column]))
    return df
            
df = handle_non_numerical_data(df)

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['survived'])

mf = MeanShift()
mf.fit(X)
cluster_centers = mf.cluster_centers_
labels = mf.labels_

original_df['cluster_group'] = np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]
    
n_clusters = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters):
    temp_df = original_df[original_df['cluster_group'] == float(i)]
    survival_cluster = temp_df[temp_df['survived'] == 1]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate
print(survival_rates)    