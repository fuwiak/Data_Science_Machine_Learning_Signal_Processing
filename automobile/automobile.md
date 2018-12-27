
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

## Description


| Attribute | Attribute Range | Type of variable |Short description | 
| --- | --- |
| symboling | -3, -2, -1, 0, 1, 2, 3.  | discrete | desc |
| normalized-losses|from 65 to 256 | discrete  | desc |
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
#DATA SOURCE
#https://archive.ics.uci.edu/ml/datasets/automobile

```


```python
import pandas as pd
import numpy as np
from sklearn import preprocessing
```


```python

df = pd.read_csv("dataset.csv")
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
#SHOW FIRST 20 ROWS
X.head(20)

```




<div>
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
</style>
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
    <tr>
      <th>11</th>
      <td>0</td>
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
      <td>16925.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>101.2</td>
      <td>176.8</td>
      <td>64.8</td>
      <td>54.3</td>
      <td>2710</td>
      <td>164</td>
      <td>9.0</td>
      <td>121.0</td>
      <td>21</td>
      <td>28</td>
      <td>20970.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>101.2</td>
      <td>176.8</td>
      <td>64.8</td>
      <td>54.3</td>
      <td>2765</td>
      <td>164</td>
      <td>9.0</td>
      <td>121.0</td>
      <td>21</td>
      <td>28</td>
      <td>21105.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>103.5</td>
      <td>189.0</td>
      <td>66.9</td>
      <td>55.7</td>
      <td>3055</td>
      <td>164</td>
      <td>9.0</td>
      <td>121.0</td>
      <td>20</td>
      <td>25</td>
      <td>24565.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>103.5</td>
      <td>189.0</td>
      <td>66.9</td>
      <td>55.7</td>
      <td>3230</td>
      <td>209</td>
      <td>8.0</td>
      <td>182.0</td>
      <td>16</td>
      <td>22</td>
      <td>30760.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>103.5</td>
      <td>193.8</td>
      <td>67.9</td>
      <td>53.7</td>
      <td>3380</td>
      <td>209</td>
      <td>8.0</td>
      <td>182.0</td>
      <td>16</td>
      <td>22</td>
      <td>41315.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>110.0</td>
      <td>197.0</td>
      <td>70.9</td>
      <td>56.3</td>
      <td>3505</td>
      <td>209</td>
      <td>8.0</td>
      <td>182.0</td>
      <td>15</td>
      <td>20</td>
      <td>36880.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>88.4</td>
      <td>141.1</td>
      <td>60.3</td>
      <td>53.2</td>
      <td>1488</td>
      <td>61</td>
      <td>9.5</td>
      <td>48.0</td>
      <td>47</td>
      <td>53</td>
      <td>5151.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>94.5</td>
      <td>155.9</td>
      <td>63.6</td>
      <td>52.0</td>
      <td>1874</td>
      <td>90</td>
      <td>9.6</td>
      <td>70.0</td>
      <td>38</td>
      <td>43</td>
      <td>6295.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>94.5</td>
      <td>158.8</td>
      <td>63.6</td>
      <td>52.0</td>
      <td>1909</td>
      <td>90</td>
      <td>9.6</td>
      <td>70.0</td>
      <td>38</td>
      <td>43</td>
      <td>6575.0</td>
    </tr>
  </tbody>
</table>
</div>



## EDA

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

    0.98



```python
#SHOW DECISION TREE GRAPH
# import graphviz 
# dot_data = tree.export_graphviz(my_class, out_file=None)
# graph = graphviz.Source(dot_data)
# graph
```

## Results

## Summary
