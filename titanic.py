#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd  
titanic =pd.read_csv('titanic.csv') #load data


# In[78]:


titanic.head(10)


# In[79]:


type(titanic)


# In[80]:


titanic = titanic.join(pd.get_dummies(titanic.Sex))
titanic = titanic.join(pd.get_dummies(titanic.Pclass))


# In[81]:


titanic.head(10)


# In[82]:


titanic = titanic.drop(columns=['Pclass'])
titanic = titanic.drop(columns=['male'])
titanic = titanic.drop(columns=['Sex']) #將原先sec drop掉
titanic.drop(1, axis='columns', inplace=True)


# In[83]:


titanic.head(10)


# In[84]:


X=titanic.drop(columns=['Survived']) #將原先sec drop掉
y = titanic["Survived"]


# In[85]:


from sklearn.linear_model import LogisticRegression #載入LogisticRegression
from sklearn.model_selection import train_test_split  #載入train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve


# In[86]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# In[87]:


lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()


# In[88]:


lr.fit(X_train,y_train);
dt.fit(X_train,y_train);
rf.fit(X_train,y_train);


# In[89]:


disp = plot_roc_curve(lr,X_test,y_test)
plot_roc_curve(dt,X_test,y_test, ax=disp.ax_);
plot_roc_curve(rf,X_test,y_test, ax=disp.ax_);


# In[90]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
##卡方可算出每一個feature對整體model 的performance是否顯著


# In[92]:


ordered_rank_features = SelectKBest(score_func = chi2, k = 6)
#score_func為我們想要使用什麼模型去選擇重要的feature，算出feature impoertance
#k為我們想要從data frame裡面選出幾個重要的features，這裡選擇6個

##可drop dummy or 不drop dummy，因為有可能drop到更有意義的feature
ordered_feature = ordered_rank_features.fit(X,y)


# In[93]:


dfscores = pd.DataFrame(ordered_feature.scores_,columns = ['score'])
dfcolumns = pd.DataFrame(X.columns)


# In[94]:


features_rank = pd.concat([dfcolumns, dfscores],axis = 1)


# In[95]:


features_rank.columns = ['features','Score']
features_rank


# In[96]:


features_rank.nlargest(6,'Score')


# In[97]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model =  ExtraTreesClassifier()
model.fit(X,y)


# In[98]:


print(model.feature_importances_)


# In[104]:


ranked_features = pd.Series(model.feature_importances_,index = X.columns)
ranked_features.nlargest(8).plot(kind='barh',color='pink')
plt.show()


# In[100]:


from sklearn.feature_selection import mutual_info_classif
#算出不純的資訊(entropy)計算出資訊含量多不多


# In[101]:


mutual_info = mutual_info_classif(X,y)


# In[102]:


mutual_data = pd.Series(mutual_info, index= X.columns)
#並將每個feature name 對應mutual_info 算出來的資訊含量數值
mutual_data.sort_values(ascending = False)
#數值越高，資訊含量越高


# In[ ]:




