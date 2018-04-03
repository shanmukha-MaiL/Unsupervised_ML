import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing

style.use('ggplot')
df = pd.read_excel('titanic.xls')
df.drop(['name','body'],1,inplace=True)
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
clf = KMeans(n_clusters = 2)
clf.fit(X,Y)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == Y[i]:
        correct+=1
print(correct/len(X))        

"""
style.use('ggplot')

X = np.array([[1, 2],[1.5, 1.8],[5, 8],[8, 8],[1, 0.6],[9, 11]])
clf = KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
colors = ['g','b']
for i in range(len(X)):
    plt.scatter(X[i][0],X[i][1],c=colors[labels[i]])
plt.scatter(centroids[:,0],centroids[:,1],marker='x')
plt.show() """
    