
# coding: utf-8

# In[1]:

import io
import requests
import pandas as pd
import numpy as np


# In[3]:

url="https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/ISLR/College.csv"
s=requests.get(url).content
data =pd.read_csv(io.StringIO(s.decode('utf-8')), delimiter = ",")
del data["Unnamed: 0"]
data["Private"] = pd.get_dummies(data["Private"],drop_first=True)
data.head()


# In[4]:

from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data, train_size = 0.7)


# In[5]:

train, validate, test = np.split(data.sample(frac=1), [int(.5*len(data)), int(.75*len(data))])


# In[6]:

scale_train = data_train.iloc[:,1::]
scale_test = data_test.iloc[:,1::]
scale_train.head()


# In[7]:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(scale_train)


# In[8]:

scale_train.iloc[:,:] = scaler.transform(scale_train)
scale_test.iloc[:,:] = scaler.transform(scale_test)
scale_test.head()


# In[9]:

from sklearn import linear_model
logreg = linear_model.LogisticRegression()
logreg.fit(scale_train, data_train["Private"])


# In[10]:

from sklearn.metrics import confusion_matrix
Private_predicted = logreg.predict(scale_train)
conf = confusion_matrix(data_train["Private"],Private_predicted) 
conf


# In[11]:

from sklearn.metrics import accuracy_score
acc = accuracy_score(data_train["Private"],Private_predicted) 
acc


# In[12]:

from sklearn.metrics import log_loss
loss = log_loss(data_train["Private"],Private_predicted)
loss


# In[13]:

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
prec = precision_score(data_train["Private"],Private_predicted, average = None)
prec


# In[14]:

from sklearn.metrics import recall_score
rec = recall_score(data_train["Private"],Private_predicted, average = None)
rec


# In[17]:

prob_predicted = logreg.predict_proba(scale_train)[:,1]


# In[18]:

def conf_matrix(prob_predicted,values, treshold):
    predicted_values = np.where(prob_predicted > treshold, 1, 0)
    matrix = confusion_matrix(values, predicted_values) 
    return matrix


# In[19]:

conf_matrix(prob_predicted,data_train["Private"], .5)


# In[20]:

def sensitivity(prob_predicted,values, treshold):
    matrix = conf_matrix(prob_predicted,values, treshold)
    return matrix[1,1] / (matrix[1,0] + matrix[1,1])


# In[21]:

sensitivity(prob_predicted,data_train["Private"], .5)


# In[22]:

def precision(prob_predicted,values, treshold):
    matrix = conf_matrix(prob_predicted,values, treshold)
    return matrix[1,1] / (matrix[1,1] + matrix[0,1])


# In[23]:

precision(prob_predicted,data_train["Private"], .5)


# In[24]:

def specifity(prob_predicted,values, treshold):
    matrix = conf_matrix(prob_predicted,values, treshold)
    return matrix[0,0] / (matrix[0,0] + matrix[0,1])


# In[25]:

specifity(prob_predicted,data_train["Private"], .5)


# In[26]:

def f_beta(prob_predicted,values, beta, treshold):
    rec = sensitivity(prob_predicted,values, treshold)
    prec = precision(prob_predicted,values, treshold)
    return 2 * prec * rec / (beta*prec + rec)


# In[27]:

f_beta(prob_predicted,data_train["Private"],1, .5)


# In[28]:

from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
f1 = f1_score(data_train["Private"],Private_predicted)
f1


# In[30]:

fbeta = fbeta_score(data_train["Private"],Private_predicted, 4)
fbeta


# In[31]:

np.median(prob_predicted)


# In[32]:

def optimal_cutoff(prob_predicted,values,cost_matrix):
    X = np.linspace(0.0,1.0,101)
    Y = [calculate_cost(prob_predicted,values,cost_matrix, x) for x in X]
    return X[Y.index(max(Y))]
    


# In[33]:

def calculate_cost(prob_predicted,values,cost_matrix, treshold):
    conf = conf_matrix(prob_predicted,values, treshold)
    return conf[0,0]*cost_matrix[0,0] + conf[1,0]*cost_matrix[1,0] + conf[0,1]*cost_matrix[0,1] + conf[1,1]*cost_matrix[1,1]


# In[34]:

cost_matrix = np.array([[-50, -100],[-100, -50]])
cost_matrix


# In[35]:

calculate_cost(prob_predicted,data_train["Private"],cost_matrix, 0.5)


# In[36]:

optimal_cutoff(prob_predicted,data_train["Private"],cost_matrix)


# In[37]:

from sklearn.model_selection import cross_val_score
scores = cross_val_score(logreg, scale_train, data_train["Private"], cv=10)
scores     


# In[38]:

from sklearn import metrics
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(logreg, scale_train, data_train["Private"], cv=10)
metrics.accuracy_score( data_train["Private"],  predicted) 


# In[ ]:



