#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd  
phone =pd.read_csv('phone.csv') #load data


# In[42]:


print(phone.head)


# In[43]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


X = phone.drop(columns=['price_range'])
y = phone.price_range


# In[45]:


from sklearn.model_selection import GridSearchCV


# In[46]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[47]:


k_range= list(range(1,31))
weight_options =["uniform","distance"]


# In[48]:


pram_grid = dict(n_neighbors=k_range,weights=weight_options)
print(pram_grid)


# In[49]:


grid = GridSearchCV(knn,pram_grid, cv=10, scoring="accuracy")
grid.fit(X,y)


# In[50]:


pd.DataFrame(grid.cv_results_)[["mean_test_score","std_test_score","params"]]


# In[51]:


print(grid.best_score_)
print(grid.best_params_)

