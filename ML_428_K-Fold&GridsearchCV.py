#!/usr/bin/env python
# coding: utf-8

# ### K fold

# In[22]:


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


iris = load_iris()

X =iris.data
y = iris.target


# In[3]:


#10-fold cross-validation with k=5 for knn
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10,scoring="accuracy")
print(scores)


# In[4]:


print(scores.mean())


# In[5]:


k_range= list(range(1,31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10,scoring="accuracy")
    k_scores.append(scores.mean())
print(k_scores)


# In[6]:


plt.plot(k_range,k_scores)
plt.xlabel("value of k for knn")
plt.ylabel("cross-validation accuracy")


# #### auto parameter tuning

# ### 調K

# In[7]:


from sklearn.model_selection import GridSearchCV


# In[8]:


k_range= list(range(1,31))
print(k_range)


# In[10]:


pram_grid = dict(n_neighbors=k_range)
print(pram_grid)


# In[12]:


grid = GridSearchCV(knn,pram_grid, cv=10, scoring="accuracy")


# In[13]:


grid.fit(X,y)


# In[15]:


import pandas as pd  
pd.DataFrame(grid.cv_results_)[["mean_test_score","std_test_score","params"]]


# In[16]:


print(grid.best_score_)
print(grid.best_params_)


# ### 調weights

# In[17]:


k_range= list(range(1,31))
weight_options =["uniform","distance"]


# In[18]:


pram_grid = dict(n_neighbors=k_range,weights=weight_options)
print(pram_grid)


# In[19]:


grid = GridSearchCV(knn,pram_grid, cv=10, scoring="accuracy")
grid.fit(X,y)


# In[20]:


pd.DataFrame(grid.cv_results_)[["mean_test_score","std_test_score","params"]]


# In[21]:


print(grid.best_score_)
print(grid.best_params_)


# In[ ]:




