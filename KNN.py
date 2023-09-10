#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[2]:


def Euclidean_dist(pt1,pt2):
    distance=0.0
    for i in range(len(pt1)):
        distance += (pt1[i]-pt2[i])**2
    return math.sqrt(distance)


# In[3]:


def Nearest_neighbors(train,test_obs,n):
    neighbor_distance= []
    for i in range(len(train)):
        l1=list(train.iloc[i,:])+[Euclidean_dist(train.iloc[i,:-1],test_obs)]
        neighbor_distance= neighbor_distance+[l1]
    neighbor_distance.sort(key=lambda x: x[-1])
    nearest_neighbors= [neighbor_distance[i] for i in range(0,n)]
    y_pred= [i[-2] for i in nearest_neighbors]
    return(int(max(y_pred,key=y_pred.count)))


# In[4]:


def Prediction(train,test_obs,n):
    
    NN=Nearest_neighbors(train,test_obs,3)
    M= [i[n-1] for i in NN]
    
    return(test_obs+[max(M)])


# In[5]:


def Normalize(data):
    df1=[]
    for i in range(len(data.columns)):
        z=[]
        z= [(k-np.mean(df.iloc[:,i]))/np.std(df.iloc[:,i]) for k in df.iloc[:,i]]
        df1.append(z)
    df1=pd.DataFrame(df1)
    df1=df1.T
    df1.columns=data.columns
    return(df1)


# In[6]:


def F_score(Act,Pred):
    ConfusionMatrix= confusion_matrix(Act,Pred)
    
    return((2*ConfusionMatrix[1,1])/(2*ConfusionMatrix[1,1]+ConfusionMatrix[1,0]+ConfusionMatrix[0,1]))


# In[7]:


def Accuracy(Act,Pred):
    ConfusionMatrix= confusion_matrix(Act,Pred)
    #return(ConfusionMatrix)
    return((ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/(len(Act)))


# In[8]:


df= pd.read_csv('LatticeVectorDatabse.csv')


# In[9]:


df.describe()


# In[10]:


df[['x1','x2','y1','y2']]= df[['x1','x2','y1','y2']].replace(0,np.NaN)


# In[11]:


df.fillna(df.mean(),inplace=True)


# In[12]:


df.describe()


# In[13]:


df.hist(bins=10,figsize=(15,10))


# In[14]:


plt.figure(figsize=(15,10))
p=sns.heatmap(df.corr(),annot=True)


# In[15]:


sns.pairplot(data=df,vars=['x1','x2','y1','y2'],hue='Outcome')


# In[16]:


X=df.drop(columns='Outcome')
Y=df['Outcome']


# In[17]:


X= Normalize(X)


# In[18]:


X.head()


# In[19]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=5)


# In[20]:


X_train=X_train.join(Y_train)


# In[21]:


print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape,sep='\n')


# In[22]:


Acc=[] 
for j in range(1,20):
    pred=[]
    for i in range(len(X_test)):
        pred.append([Nearest_neighbors(X_train,X_test.iloc[i,:],j)])
    Acc= Acc+([Accuracy(Y_test,pred)])
k=7    


# In[23]:


Acc


# In[24]:


pred=[]
for i in range(len(X_test)):
    pred.append(Nearest_neighbors(X_train,X_test.iloc[i,:],Acc.index(max(Acc))+1))
    
X_test['Pred']= pred
X_test['Outcome']= Y_test


# In[25]:


from sklearn.metrics import accuracy_score


# In[30]:


print(accuracy_score(X_test['Outcome'], X_test['Pred']))


# In[27]:


pd.crosstab(X_test['Outcome'], X_test['Pred'], rownames=['True'], colnames=['Predicted'], margins=True)


# In[ ]:




