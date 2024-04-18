#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
dia = pd.read_csv('diabetes.csv')
dia.head()


# In[3]:


dia.describe()


# In[5]:


dia.shape


# In[6]:


dia.isnull().sum()


# In[7]:


X=dia.iloc[:,:-1].to_numpy()
y=dia.iloc[:,-1].to_numpy()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=99)


# In[8]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="entropy",random_state=99)
clf.fit(X_train,y_train)


# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['no','yes'])
plt.show()


# In[13]:


clf.set_params(max_depth=3)
clf.fit(X_train,y_train)
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[15]:


predictions=clf.predict(X_test)
clf.predict([[90,20],[200,30]])


# In[16]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X_train,y_train,cv=5,scoring='accuracy')
accuracy=scores.mean()
accuracy


# In[19]:


from sklearn import metrics
cf=metrics.confusion_matrix(y_test,predictions)
cf


# In[21]:


tp=cf[1][1]
tn=cf[0][0]
fp=cf[0][1]
fn=cf[1][0]
tp


# In[22]:


tn


# In[23]:


fp


# In[24]:


fn


# In[25]:


print("accuracy",metrics.accuracy_score(y_test,predictions))
print("precision",metrics.precision_score(y_test,predictions))
print("recall",metrics.recall_score(y_test,predictions))


# In[26]:


feature_importances=clf.feature_importances_
print("feature importances:",feature_importances) 

