#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
heart =pd.read_csv('heart.csv') #load data


# In[2]:


heart.head(10)


# In[3]:


from sklearn.model_selection import train_test_split  #載入train_test_split


# In[4]:


X = heart.drop(columns=['target'])
y = heart.target


# In[5]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# #### KNN

# In[6]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
print(knn)


# In[7]:


#step 3: Fit/Learn the model with data
knn.fit(X_train,y_train) #沒有分train, test


# In[16]:


#compute classification accuracy for KNN model
#training accuracy - testing the model on the same model we used to predict 
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[17]:


from sklearn.metrics import confusion_matrix, plot_confusion_matrix #載入confusion_matrix以及plot_confusion_matrix


# In[18]:


print(confusion_matrix(y_test,y_pred))

print(knn.score(X_test,y_test))
plot_confusion_matrix(knn,X_test,y_test,cmap='gray_r')


# In[19]:


a = confusion_matrix(y_test,y_pred)
TN =a[0][0]
print(a[0][0])
FP =a[0][1]
print(a[0][1])
FN =a[1][0]
print(a[1][0])
TP =a[1][1]
print(a[1][1])


# In[20]:


#Accuracy精確度 表示模型預估正確的機率
#所有人中有病、沒病被正確預測出來的比例
Accuracy =(TP+TN)/(TP+TN+FP+FN)
print(Accuracy)

#Precision 精確度  
# 預測出來生有病的人，有多少比例真的有病
Precision = (TP/(TP+FP))
print(Precision)

#Recall #真的有病的人，有多少比例的人預測出來有病
Recall =(TP/(TP+FN))
print(Recall)

# F score #運用Precision and recall 的總和評比價值
F = 2*((Precision*Recall)/(Precision+Recall))
print(F)


# In[21]:


#true_positive_rate 
true_positive_rate = TP/(TP+FN)
#fake_negitive_rate 
fake_negative_rate = FP/(FP+TN)
print(true_positive_rate)
print(fake_negative_rate)


# In[63]:


disp = plot_roc_curve(knn,X_test,y_test)


# #### LogisticRegression

# In[23]:


from sklearn.linear_model import LogisticRegression #載入LogisticRegression
from sklearn.model_selection import train_test_split  #載入train_test_split
from sklearn.metrics import plot_roc_curve


# In[24]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# In[25]:


lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[26]:


y_pred_lo = lr.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred_lo )) 


# In[27]:


print(confusion_matrix(y_test,y_pred_lo))

print(lr.score(X_test,y_test))
plot_confusion_matrix(lr,X_test,y_test,cmap='gray_r')


# In[28]:


a = confusion_matrix(y_test,y_pred_lo)
TN =a[0][0]
print(a[0][0])
FP =a[0][1]
print(a[0][1])
FN =a[1][0]
print(a[1][0])
TP =a[1][1]
print(a[1][1])


# In[29]:


#Accuracy精確度 表示模型預估正確的機率
#所有人中有病、沒病被正確預測出來的比例
Accuracy =(TP+TN)/(TP+TN+FP+FN)
print(Accuracy)

#Precision 精確度  
#預測出來生有病的人，有多少比例真的有病
Precision = (TP/(TP+FP))
print(Precision)

#Recall #真的有病的人，有多少比例的人預測出來有病
Recall =(TP/(TP+FN))
print(Recall)

# F score #運用Precision and recall 的總和評比價值
F = 2*((Precision*Recall)/(Precision+Recall))
print(F)


# In[30]:


#true_positive_rate 
true_positive_rate = TP/(TP+FN)
#fake_negitive_rate 
fake_negative_rate = FP/(FP+TN)
print(true_positive_rate)
print(fake_negative_rate)


# In[31]:


disp = plot_roc_curve(lr,X_test,y_test)


# ###### svm

# In[64]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# In[65]:


X_SVM = heart.drop(columns=['target'])
y_SVM= heart.target


# In[66]:


X_train,X_test,y_train,y_test = train_test_split(X_SVM,y_SVM,random_state=0)


# In[67]:


clf = SVC()


# In[68]:


clf.fit(X_train,y_train)


# In[69]:


y_pred_SVM=clf.predict(X_test)


# In[70]:


print(clf.score(X_test, y_test))


# In[71]:


print(confusion_matrix(y_test,y_pred_SVM))

print(clf.score(X_test,y_test))
plot_confusion_matrix(clf,X_test,y_test,cmap='gray_r')


# In[72]:


a = confusion_matrix(y_test,y_pred_SVM)
TN =a[0][0]
print(a[0][0])
FP =a[0][1]
print(a[0][1])
FN =a[1][0]
print(a[1][0])
TP =a[1][1]
print(a[1][1])


# In[73]:


#Accuracy精確度 表示模型預估正確的機率
#所有人中有病、沒病被正確預測出來的比例
Accuracy =(TP+TN)/(TP+TN+FP+FN)
print(Accuracy)

#Precision 精確度  
# 預測出來生有病的人，有多少比例真的有病
Precision = (TP/(TP+FP))
print(Precision)

#Recall #真的有病的人，有多少比例的人預測出來有病
Recall =(TP/(TP+FN))
print(Recall)

# F score #運用Precision and recall 的總和評比價值
F = 2*((Precision*Recall)/(Precision+Recall))
print(F)


# In[74]:


#true_positive_rate 
true_positive_rate = TP/(TP+FN)
#fake_negitive_rate 
fake_negative_rate = FP/(FP+TN)
print(true_positive_rate)
print(fake_negative_rate)


# In[75]:


disp = plot_roc_curve(clf,X_test,y_test)


# ### DecisionTree

# In[45]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
X_dt = heart.drop(columns=['target'])
y_dt= heart.target


# In[46]:


X_train,X_test,y_train,y_test = train_test_split(X_dt,y_dt,random_state=0)
dt.fit(X_train,y_train);


# In[47]:


y_pred_dt=clf.predict(X_test)


# In[48]:


print(dt.score(X_test, y_test))


# In[49]:


print(confusion_matrix(y_test,y_pred_dt))

print(dt.score(X_test,y_test))
plot_confusion_matrix(dt,X_test,y_test,cmap='gray_r')


# In[50]:


a = confusion_matrix(y_test,y_pred_dt)
TN =a[0][0]
print(a[0][0])
FP =a[0][1]
print(a[0][1])
FN =a[1][0]
print(a[1][0])
TP =a[1][1]
print(a[1][1])


# In[51]:


#Accuracy精確度 表示模型預估正確的機率
#所有人中有病、沒病被正確預測出來的比例
Accuracy =(TP+TN)/(TP+TN+FP+FN)
print(Accuracy)

#Precision 精確度  
# 預測出來生有病的人，有多少比例真的有病
Precision = (TP/(TP+FP))
print(Precision)

#Recall #真的有病的人，有多少比例的人預測出來有病
Recall =(TP/(TP+FN))
print(Recall)

# F score #運用Precision and recall 的總和評比價值
F = 2*((Precision*Recall)/(Precision+Recall))
print(F)

#true_positive_rate 
true_positive_rate = TP/(TP+FN)
#fake_negitive_rate 
fake_negative_rate = FP/(FP+TN)
print(true_positive_rate)
print(fake_negative_rate)


# In[52]:


disp = plot_roc_curve(dt,X_test,y_test)


# #### RandomForest

# In[53]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()


# In[54]:


X_rf = heart.drop(columns=['target'])
y_rf= heart.target
X_train,X_test,y_train,y_test = train_test_split(X_rf,y_rf,random_state=0)


# In[55]:


rf.fit(X_train,y_train);


# In[56]:


y_pred_rf=rf.predict(X_test)


# In[57]:


print(rf.score(X_test, y_test))


# In[58]:


print(confusion_matrix(y_test,y_pred_rf))

print(rf.score(X_test,y_test))
plot_confusion_matrix(rf,X_test,y_test,cmap='gray_r')


# In[59]:


a = confusion_matrix(y_test,y_pred_rf)
TN =a[0][0]
print(a[0][0])
FP =a[0][1]
print(a[0][1])
FN =a[1][0]
print(a[1][0])
TP =a[1][1]
print(a[1][1])
#true_positive_rate 
true_positive_rate = TP/(TP+FN)
#fake_negitive_rate 
fake_negative_rate = FP/(FP+TN)
print(true_positive_rate)
print(fake_negative_rate)


# In[60]:


disp = plot_roc_curve(rf,X_test,y_test)


# In[61]:


#Accuracy精確度 表示模型預估正確的機率
#所有人中有病、沒病被正確預測出來的比例
Accuracy =(TP+TN)/(TP+TN+FP+FN)
print(Accuracy)

#Precision 精確度  
# 預測出來生有病的人，有多少比例真的有病
Precision = (TP/(TP+FP))
print(Precision)

#Recall #真的有病的人，有多少比例的人預測出來有病
Recall =(TP/(TP+FN))
print(Recall)

# F score #運用Precision and recall 的總和評比價值
F = 2*((Precision*Recall)/(Precision+Recall))
print(F)


# In[76]:


disp = plot_roc_curve(lr,X_test,y_test)
plot_roc_curve(clf,X_test,y_test, ax=disp.ax_);
plot_roc_curve(knn,X,y, ax=disp.ax_);
plot_roc_curve(dt,X_test,y_test, ax=disp.ax_);
plot_roc_curve(rf,X_test,y_test, ax=disp.ax_);

