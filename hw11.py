#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd  
import numpy as np
data =pd.read_csv('mnist_test.csv') #load data
data.head()


# In[3]:


data.shape


# In[10]:


X = data.drop('label',axis=1)
y = data['label']
print(x.head(2))
print(y.head(2))


# In[11]:


X.shape


# In[12]:


y.shape


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(y_test.shape)


# In[14]:


from sklearn.neural_network import MLPClassifier
from sklearn import set_config


# In[17]:


set_config(print_changed_only=False)
nn=MLPClassifier(max_iter=500, activation='relu')
nn


# In[18]:


nn.fit(X_train,y_train)


# In[19]:


y_pred = nn.predict(X_test)
y_pred


# In[20]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# In[ ]:




