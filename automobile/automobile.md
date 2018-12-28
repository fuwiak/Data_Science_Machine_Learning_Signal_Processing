
# Table of Contents
1. [Goals](#Goals)
2. [Description of Attributes](#Description)
3. [Preprocessing](#Preprocessing)
4. [EDA](#EDA)
5. [Models](#Models)
6. [Results](#Results)
7. [Summary](#Summary)
# Goals

Prediction price of car using machine learning algorithms.

DATA SOURCE
https://archive.ics.uci.edu/ml/datasets/automobile


## Description


| Attribute | Attribute Range | Type of variable |Short description | 
| --- | --- | --- | --- |
| symboling | -3, -2, -1, 0, 1, 2, 3.  | discrete | A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe. |
| normalized-losses|from 65 to 256 | discrete  |  represents the average loss per car per year. |
| make |alfa-romero, audi, bmw, chevrolet, etc  | discrete  | desc |
| fuel-type | diesel, gas.   | discrete  | desc |
| aspiration | std, turbo | discrete  | desc |
| num-of-doors | four, two | discrete  | desc |
| body-style | hardtop, wagon, sedan, hatchback, convertible.  | discrete  | desc |
| drive-wheels | 4wd, fwd, rwd | discrete | desc |
| engine-location|  front, rear. | discrete | desc |
| wheel-base| from 86.6 to 120.9.|continuous| desc|
| length |141.1 to 208.1. |continuous| desc|
| width | from 60.3 to 72.3.| continuous | desc|
| height | from 47.8 to 59.8| continuous | desc|
| curb-weight| from 1488 to 4066. | continuous | desc|
| engine-type| dohc, dohcv, l, ohc, ohcf, ohcv, rotor | discrete | desc|
| num-of-cylinders | eight, five, four, six, three, twelve, two | discrete | desc|
| engine-size | from 61 to 326 | continuous | desc | 
| fuel-system | 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi | discrete | desc |
| bore | from 2.54 to 3.94. | continuous | desc |
| stroke | 2.07 to 4.17. |  continuous | desc |
| compression-ratio | 7 to 23. | continuous | desc |
| horsepower | from 48 to 288 | continuous | desc |
| peak-rpm | from 4150 to 6600. | continuous | desc |
| city-mpg | from 13 to 49.  | continuous | desc | 
| highway-mpg | from 16 to 54. | continuous | desc |
| price | from 5118 to 45400.  | continuous | desc |

## Preprocessing


```python
import pandas as pd
import numpy as np
from sklearn import preprocessing
```


```python
df = pd.read_csv("dataset.csv") # read a data
```


```python
data_types = df.dtypes #check datatypes
```


```python
data_types
```




    symboling              int64
    normalized-losses     object
    make                  object
    fuel-type             object
    aspiration            object
    num-of-doors          object
    body-style            object
    drive-wheels          object
    engine-location       object
    wheel-base           float64
    length               float64
    width                float64
    height               float64
    curb-weight            int64
    engine-type           object
    num-of-cylinders      object
    engine-size            int64
    fuel-system           object
    bore                  object
    stroke                object
    compression-ratio    float64
    horsepower            object
    peak-rpm              object
    city-mpg               int64
    highway-mpg            int64
    price                 object
    dtype: object




```python
df["price"] = df["price"].apply(lambda x: x.replace("?", "0")) # replace "?" with "0"
df["horsepower"] = df["horsepower"].apply(lambda x: x.replace("?", "0")) # replace "?" with "0"

```


```python

df["price"] = df["price"].astype(float) # convert type from object to float64
df["horsepower"] = df["horsepower"].astype(float) # convert type from object to float64
```


```python
df = df[df["price"]>0] # get only positive values of price 
df = df[df["horsepower"]>0] # get only positive values of horsepower
```


```python
#CREATE PRICE INTERVALS FOR CLASSIFICATION
count_intervals = 8
Z = pd.cut(df.price,count_intervals)

```


```python
#SHOW INTERVALS
Z.dtypes;

```


```python
#Prepare intervals to classification

CATS = Z.dtypes.categories
LABELS = [str((x.left, x.right)) for x in CATS]
Y = pd.cut(df.price, count_intervals, labels=LABELS)

```


```python
#GET ONLY NUMERIC COLUMNS
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = df.select_dtypes(include=numerics).columns
X = df[numerical_columns]

```


```python
#SHOW FIRST 10 ROWS
X.head(10)

```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style> -->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>130</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>130</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>2823</td>
      <td>152</td>
      <td>9.0</td>
      <td>154.0</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>2337</td>
      <td>109</td>
      <td>10.0</td>
      <td>102.0</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>2824</td>
      <td>136</td>
      <td>8.0</td>
      <td>115.0</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>99.8</td>
      <td>177.3</td>
      <td>66.3</td>
      <td>53.1</td>
      <td>2507</td>
      <td>136</td>
      <td>8.5</td>
      <td>110.0</td>
      <td>19</td>
      <td>25</td>
      <td>15250.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>105.8</td>
      <td>192.7</td>
      <td>71.4</td>
      <td>55.7</td>
      <td>2844</td>
      <td>136</td>
      <td>8.5</td>
      <td>110.0</td>
      <td>19</td>
      <td>25</td>
      <td>17710.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>105.8</td>
      <td>192.7</td>
      <td>71.4</td>
      <td>55.7</td>
      <td>2954</td>
      <td>136</td>
      <td>8.5</td>
      <td>110.0</td>
      <td>19</td>
      <td>25</td>
      <td>18920.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>105.8</td>
      <td>192.7</td>
      <td>71.4</td>
      <td>55.9</td>
      <td>3086</td>
      <td>131</td>
      <td>8.3</td>
      <td>140.0</td>
      <td>17</td>
      <td>20</td>
      <td>23875.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>101.2</td>
      <td>176.8</td>
      <td>64.8</td>
      <td>54.3</td>
      <td>2395</td>
      <td>108</td>
      <td>8.8</td>
      <td>101.0</td>
      <td>23</td>
      <td>29</td>
      <td>16430.0</td>
    </tr>
  </tbody>
</table>
</div>



## EDA


```python
X.columns
```




    Index(['symboling', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
           'engine-size', 'compression-ratio', 'horsepower', 'city-mpg',
           'highway-mpg', 'price'],
          dtype='object')




```python
X.describe().T
```




<div>
<!-- <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style> -->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>symboling</th>
      <td>199.0</td>
      <td>0.839196</td>
      <td>1.257009</td>
      <td>-2.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>wheel-base</th>
      <td>199.0</td>
      <td>98.824121</td>
      <td>6.090838</td>
      <td>86.6</td>
      <td>94.50</td>
      <td>97.0</td>
      <td>102.40</td>
      <td>120.9</td>
    </tr>
    <tr>
      <th>length</th>
      <td>199.0</td>
      <td>174.151256</td>
      <td>12.371905</td>
      <td>141.1</td>
      <td>166.55</td>
      <td>173.2</td>
      <td>183.50</td>
      <td>208.1</td>
    </tr>
    <tr>
      <th>width</th>
      <td>199.0</td>
      <td>65.882412</td>
      <td>2.110996</td>
      <td>60.3</td>
      <td>64.10</td>
      <td>65.5</td>
      <td>66.70</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>height</th>
      <td>199.0</td>
      <td>53.775879</td>
      <td>2.447039</td>
      <td>47.8</td>
      <td>52.00</td>
      <td>54.1</td>
      <td>55.55</td>
      <td>59.8</td>
    </tr>
    <tr>
      <th>curb-weight</th>
      <td>199.0</td>
      <td>2556.030151</td>
      <td>519.855544</td>
      <td>1488.0</td>
      <td>2157.00</td>
      <td>2414.0</td>
      <td>2930.50</td>
      <td>4066.0</td>
    </tr>
    <tr>
      <th>engine-size</th>
      <td>199.0</td>
      <td>126.824121</td>
      <td>41.752932</td>
      <td>61.0</td>
      <td>97.50</td>
      <td>119.0</td>
      <td>143.00</td>
      <td>326.0</td>
    </tr>
    <tr>
      <th>compression-ratio</th>
      <td>199.0</td>
      <td>10.178995</td>
      <td>4.022424</td>
      <td>7.0</td>
      <td>8.55</td>
      <td>9.0</td>
      <td>9.40</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>199.0</td>
      <td>103.396985</td>
      <td>37.553843</td>
      <td>48.0</td>
      <td>70.00</td>
      <td>95.0</td>
      <td>116.00</td>
      <td>262.0</td>
    </tr>
    <tr>
      <th>city-mpg</th>
      <td>199.0</td>
      <td>25.201005</td>
      <td>6.451826</td>
      <td>13.0</td>
      <td>19.00</td>
      <td>24.0</td>
      <td>30.00</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>highway-mpg</th>
      <td>199.0</td>
      <td>30.683417</td>
      <td>6.849410</td>
      <td>16.0</td>
      <td>25.00</td>
      <td>30.0</td>
      <td>34.00</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>price</th>
      <td>199.0</td>
      <td>13243.432161</td>
      <td>7978.707609</td>
      <td>5118.0</td>
      <td>7775.00</td>
      <td>10345.0</td>
      <td>16501.50</td>
      <td>45400.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.hist(bins=10)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x10c1cef28>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10fbb5198>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10fbde5c0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x10fc05b38>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10fc380f0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10fc60668>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x10fc86be0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10fcb91d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10fcb9208>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x10fd06c88>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10fd37240>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10fd5d7b8>]],
          dtype=object)



## Models


```python
#First classifier 

from sklearn import tree
```


```python
my_class = tree.DecisionTreeClassifier()
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, np.array(Y), test_size=0.5)
```


```python
my_class.fit(X_train, Y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
predictions = my_class.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score
```


```python
print(accuracy_score(Y_test, predictions))
```

    0.95



```python
#SHOW DECISION TREE GRAPH
# import graphviz 
# dot_data = tree.export_graphviz(my_class, out_file=None)
# graph = graphviz.Source(dot_data)
# graph
```

## Results

## Summary
