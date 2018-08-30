
#DATA SOURCE
#https://archive.ics.uci.edu/ml/datasets/automobile

import pandas as pd
import numpy as np
from sklearn import preprocessing


df = pd.read_csv("dataset.csv")


data_types = df.dtypes

#get numeric columns
# df._get_numeric_sdata()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = df.select_dtypes(include=numerics).columns
categorical_columns=df.select_dtypes(exclude=numerics).columns

#get all  object type
df.select_dtypes(include='object')

#REMOVE LACKS OF DATA AND TRANSFORM FROM TYPE OBJECT TO FLOAT
df["price"].apply(lambda x: x.replace("?", "0")).astype("float")


from sklearn import tree
num = df[numerical_columns]
NUM =  [list(num[num.columns[i]]) for i in range(len(num.columns))]
x1 = num[num.columns[0]]

x2 = num[num.columns[1]]


yy = np.array(num[num.columns[0]]).reshape(-1,1)

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(x2)



from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x1.values.reshape(-1,1),encoded)
