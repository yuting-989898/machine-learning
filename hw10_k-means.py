#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[2]:


#設5個質心=強制分成5群
blob_centers = np.array(
[[0.2, 2.3],
[-1.5, 2.3],
[-2.8, 1.8],
[-2.8, 2.8],
[-2.8, 1.3]])

#5群之間的標準差
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])


# In[3]:


#2000個data point

X, y =make_blobs(n_samples=2000, centers=blob_centers,
                cluster_std=blob_std, random_state=7)


# In[4]:


#defind 畫圖
# X[:,0]:index=0的column, every row

def plot_clusters(X,y=None):
    plt.scatter(X[:,0],X[:,1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14,rotation=0)
    


# In[6]:


plt.figure(figsize=(8,4))
plot_clusters(X)
plt.show()


# ### k-means

# In[7]:


from sklearn.cluster import KMeans


# In[11]:


#分5群
#unsupervise 沒有 y
#y為cluster 的index
kmeans=KMeans(n_clusters=5, random_state=6)
y_pred = kmeans.fit_predict(X)


# In[12]:


y_pred


# In[13]:


y_pred is kmeans.labels_


# In[15]:


#驗算 倒推質心

kmeans.cluster_centers_


# In[16]:


#倒推index

kmeans.labels_


# 分完群後，每個cluster之間直徑有很大的不同-->結果不好

# In[19]:


#分群可以predict分到哪群

X_new = np.array([[0,2], [3,2], [-3,3], [-3,2.5]])
kmeans.predict(X_new)


# In[20]:


#soft clustering:算出每個point的距離質心的分數

kmeans.transform(X_new )


# In[22]:


#計算每個point的跟質心的距離平方(ssd)

kmeans.inertia_


# In[24]:


X_dist = kmeans.transform(X)
np.sum(X_dist[np.arange(len(X_dist)), kmeans.labels_]**2)


# In[25]:


#找到k質最佳解

kmeans_per_k = [KMeans(n_clusters=k, random_state=6).fit(X)
               for k in range(1, 10)]
inertias =[model.inertia_ for model in kmeans_per_k]


# In[26]:


#inertia隨著k越多，會越小
#變化激烈的地方是最中庸的解

plt.figure(figsize=(8,3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("inertia", fontsize=14)
plt.annotate("Elbow",
            xy=(4, inertias[3]),
            xytext=(0.55, 0.55),
            textcoords='figure fraction',
            fontsize=16,
            arrowprops=dict(facecolor='black',shrink=0.1))
plt.axis([1, 9, 0, 1500])
plt.show()


# ### DBSCAN

# 找出高密度的區域
# 幫每一個data point找出有多少point在指定範圍(epsilon)裏面

# In[27]:


from sklearn.datasets import make_moons


# In[28]:


X ,y =make_moons(n_samples=1000,noise=0.05,random_state=8)


# In[29]:


from sklearn.cluster import DBSCAN


# In[30]:


dbscan= DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)


# In[31]:


#找出相對應的CLUSTER INDEX
dbscan.labels_[:20]


# In[32]:


#找出所有core data point
len(dbscan.core_sample_indices_)


# In[33]:


#各自索引有哪些
dbscan.core_sample_indices_[:10]


# In[34]:


#取得core data point
dbscan.components_[:3]


# In[35]:


#鄰居範圍增大
dbscan2= DBSCAN(eps=2)
dbscan2.fit(X)


# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


knn =KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])


# In[38]:


X_new = np.array([[-0.5, 0], [0, 0.5],[1,-0.1],[2,1]])
knn.predict(X_new)


# ### Gaussian Mixtures

# 用來做異常偵測
# 適用生產線

# In[39]:


from sklearn.mixture import GaussianMixture


# In[40]:


#
gm =GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)


# In[41]:


gm.weights_


# In[42]:


gm.means_


# In[43]:


gm.covariances_


# In[44]:


gm.converged_


# In[45]:


gm.n_iter_


# In[46]:


#把每個DATA POINT分到適合的群
gm.predict(X)


# In[47]:


#可以生成新資料
X_new,y_new =gm.sample(6)
X_new


# In[48]:


y_new


# In[49]:


#估計不同位置的密度(機率密度函數的對數,分數越高密度越高)
gm.score_samples(X)


# In[50]:


#異常檢測，找出outlier
#threshold 設6%
#density低於6%劃進異常質
density = gm.score_samples(X)
desity_threshold = np.percentile(density, 6)
anomaly = X[density < desity_threshold]


# In[51]:


anomaly


# In[ ]:




