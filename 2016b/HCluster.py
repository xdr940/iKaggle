import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


np.random.seed(123)

#variables = ['X', 'Y', 'Z']
#labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

'''
variables = []
for i in range(10000):
    variables.append(str(i))
labels =[]
for i in range(100):
        labels.append( 'ID_'+str(i))
'''

#
variables = []
labels =[]
vecs = np.load('vecs.npy')
h,w = vecs.shape

for i in range(w):
    variables.append(str(i))
labels =[]
for i in range(h):
        labels.append(str(i))

#

#X = np.random.random_sample([100, 10000]) * 10
X=vecs



# 层次聚类树
df = pd.DataFrame(X, columns=variables, index=labels)
#print (df)
# 计算距离关联矩阵，两两样本间的欧式距离
# row_dist = pd.DataFrame(squareform(pdist(df,metric='euclidean')),columns=labels,index=labels)
# print (row_dist)
# print (help(linkage))
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')  # 使用抽秘籍距离矩阵
# row_clusters = linkage(df.values,method='complete',metric='euclidean')
print (pd.DataFrame(row_clusters, columns=['row label1', 'row label2', 'distance', 'no. of items in clust.'],
                    index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]))
# 层次聚类树
row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()
# 层次聚类热度图
fig = plt.figure(figsize=(8, 8))
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='right')
df_rowclust = df.ix[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

# 凝聚层次聚类，应用对层次聚类树剪枝
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(X)
print ('cluster labels:%s' % labels)