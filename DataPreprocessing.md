
# Data Preprocessing

Explains the procedure of Data Preprocessing in Python, including removal/ redefinition of undefined or missing values, applying categorisation etc.

## Template

### Importing Libraries


```python
# For Basic Operations

import numpy as np                     # Computer
import matplotlib.pyplot as plt        # Plotter
import pandas as pd                    # Data Handler
from sklearn.preprocessing import *    # Data Preprocessor
from sklearn.model_selection import * # Data CrossValidator

# For Displaying Dataset

from IPython.display import Markdown, display

# utility functions

# line print
def ln(): print("\n")
    
# markdown print
def md(string): display(Markdown(str(string)))
def bi(string): return "***"+str(string)+"***"
def bo(string): return "**"+str(string)+"**"
def it(string): return "*"+str(string)+"*"

# table print
def tab(data): display(pd.DataFrame(data))
def table(data, names=[]): display(pd.DataFrame(data, columns = names))
```

### Importing Dataset


```python
dataset = pd.read_csv('./Data.csv')
tab(dataset)
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
      <th>Country</th>
      <th>Age</th>
      <th>Salary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>44.0</td>
      <td>72000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>27.0</td>
      <td>48000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>30.0</td>
      <td>54000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spain</td>
      <td>38.0</td>
      <td>61000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>France</td>
      <td>35.0</td>
      <td>58000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spain</td>
      <td>NaN</td>
      <td>52000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>48.0</td>
      <td>79000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>50.0</td>
      <td>83000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>France</td>
      <td>37.0</td>
      <td>67000.0</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>


### Splitting the variables

This is required since models can be applied only on independent variables to *predict* the corresponding dependent variables.

#### Independent Variables


```python
X = dataset.iloc[:, :-1].values # Independent Variables
table(X,['Country','Age','Salary'])
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
      <th>Country</th>
      <th>Age</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>44</td>
      <td>72000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>27</td>
      <td>48000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>30</td>
      <td>54000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spain</td>
      <td>38</td>
      <td>61000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>France</td>
      <td>35</td>
      <td>58000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spain</td>
      <td>NaN</td>
      <td>52000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>48</td>
      <td>79000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>50</td>
      <td>83000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>France</td>
      <td>37</td>
      <td>67000</td>
    </tr>
  </tbody>
</table>
</div>


As you can see, the above command retrieves a numpy array of all rows (:) and all columns except last (:-1)

#### Dependent Variables

In this case, there is only one column, *"Purchased"* that is dependent on the other 3 independent variables.


```python
Y = dataset.iloc[:, 3].values # Dependent Variables
table(Y,['Purchased'])
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
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>


### Check Missing Data


```python
md("Does the dataset have missing data:" + bo(dataset.isnull().values.any()))
ln()
md(bi("Col-wise Missing Values"))
table(dataset.isnull().sum(),['NaN Count'])
ln()
md("Total number of missing values in the dataset is " + bo(dataset.isnull().sum().sum()))
```


Does the dataset have missing data:**True**


    
    



***Col-wise Missing Values***



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
      <th>NaN Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Country</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Salary</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Purchased</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    



Total number of missing values in the dataset is **2**

Causes the following error when found by Model:
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
### Handling Missing Data

Missing Data can be handled in Python using a class in *sklearn.preprocessing* module known as the **Imputer**.


***missing_values : integer or “NaN”, optional (default=”NaN”)***

The placeholder for the missing values. All occurrences of missing_values will be imputed. For missing values encoded as np.nan, use the string value “NaN”.

***strategy : string, optional (default=”mean”)***

The imputation strategy.

If “mean”, then replace missing values using the mean along the axis.
If “median”, then replace missing values using the median along the axis.
If “most_frequent”, then replace missing using the most frequent value along the axis.
axis : integer, optional (default=0)

The axis along which to impute.

If axis=0, then impute along columns.
If axis=1, then impute along rows.

***verbose : integer, optional (default=0)***

Controls the verbosity of the imputer.

***copy : boolean, optional (default=True)***

If True, a copy of X will be created. If False, imputation will be done in-place whenever possible. 
Note that, in the following cases, a new copy will always be made, even if copy=False:
    - If X is not an array of floating values
    - If X is sparse and missing_values=0
    - If axis=0 and X is encoded as a CSR matrix
    - If axis=1 and X is encoded as a CSC matrix


```python
# Creating imputer objects to fit data with

def imputeBy (strata, dataset, missing='NaN'):
    return Imputer(missing_values = missing, strategy = strata, axis =0, copy = False).fit(dataset).transform(dataset)

```


```python
# Simple imputer function call

X[:, 1:3] = imputeBy ('mean', X[:, 1:3])
table(X,['Country','Age','Salary'])
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
      <th>Country</th>
      <th>Age</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>44</td>
      <td>72000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>27</td>
      <td>48000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>30</td>
      <td>54000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spain</td>
      <td>38</td>
      <td>61000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>40</td>
      <td>63777.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>France</td>
      <td>35</td>
      <td>58000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spain</td>
      <td>38.7778</td>
      <td>52000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>48</td>
      <td>79000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>50</td>
      <td>83000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>France</td>
      <td>37</td>
      <td>67000</td>
    </tr>
  </tbody>
</table>
</div>


### Obtaining the Training and Test sets from the Dataset

#### Imports required from scikit learn ```(cross_validation)```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
```


```python
md(bi("X from Training Set"))
tab (X_train)
ln()
md(bi("X from Test Set"))
tab (X_test)
ln()
md(bi("Y from Training Set"))
tab (Y_train)
ln()
md(bi("Y from Test Set"))
tab (Y_test)
```


***X from Training Set***



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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Germany</td>
      <td>40</td>
      <td>63777.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>37</td>
      <td>67000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spain</td>
      <td>27</td>
      <td>48000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spain</td>
      <td>38.7778</td>
      <td>52000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>48</td>
      <td>79000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spain</td>
      <td>38</td>
      <td>61000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>France</td>
      <td>44</td>
      <td>72000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>35</td>
      <td>58000</td>
    </tr>
  </tbody>
</table>
</div>


    
    



***X from Test Set***



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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Germany</td>
      <td>30</td>
      <td>54000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Germany</td>
      <td>50</td>
      <td>83000</td>
    </tr>
  </tbody>
</table>
</div>


    
    



***Y from Training Set***



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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>


    
    



***Y from Test Set***



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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>


### How to Convert Categorical Data to Numerical Data?

This involves two steps:

Integer Encoding (done by LabelEncoder)
One-Hot Encoding (done by OneHotEncoder)

1. ***Integer Encoding***

As a first step, each unique category value is assigned an integer value.

For example, ```“red” is 1, “green” is 2, and “blue” is 3.```

This is called a label encoding or an integer encoding and is easily reversible.

For some variables, this may be enough.

The integer values have a natural ordered relationship between each other and machine learning algorithms may be able to understand and harness this relationship.

For example, ordinal variables like the “place” example above would be a good example where a label encoding would be sufficient.

2. ***One-Hot Encoding***

For categorical variables where no such ordinal relationship exists, the integer encoding is not enough.

In fact, using this encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results (predictions halfway between categories).

In this case, a one-hot encoding can be applied to the integer representation. This is where the integer encoded variable is removed and a new binary variable is added for each unique integer value.

In the “color” variable example, there are 3 categories and therefore 3 binary variables are needed. A “1” value is placed in the binary variable for the color and “0” values for the other colors.

For example:

**in Label Encoding or Integer Encoding**
```
1
2
3
```
**in One Hot Encoding**
```
red,	green,	blue
1,		0,		0
0,		1,		0
0,		0,		1
```
The binary variables are often called ***“dummy variables”*** in other fields, such as statistics.

### Encoding Categorical Variables


```python
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def labelEncode (dataset, column):
    return LabelEncoder().fit_transform(dataset[:,column])

def oneHotEncode (dataset, column):
    return OneHotEncoder(categorical_features = [column]).fit_transform(X).toarray()
```


```python
Y = X

# Label Encoding
X[:, 0] = labelEncode (X,0)
table(X,['Country','Age','Salary'])
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
      <th>Country</th>
      <th>Age</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>44</td>
      <td>72000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>27</td>
      <td>48000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>30</td>
      <td>54000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>38</td>
      <td>61000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>40</td>
      <td>63777.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>35</td>
      <td>58000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>38.7778</td>
      <td>52000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>48</td>
      <td>79000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>50</td>
      <td>83000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>37</td>
      <td>67000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# One Hot Encoding
Y = oneHotEncode (Y,0)
table(Y.astype(int),['France','Germany','Spain','Age','Salary'])
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
      <th>France</th>
      <th>Germany</th>
      <th>Spain</th>
      <th>Age</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>72000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>27</td>
      <td>48000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>30</td>
      <td>54000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>38</td>
      <td>61000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>40</td>
      <td>63777</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>58000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>38</td>
      <td>52000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>79000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>50</td>
      <td>83000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>67000</td>
    </tr>
  </tbody>
</table>
</div>


***Note:*** 

   **fit_transform(y)** :	
       Fit label encoder and return encoded labels
   
   **transform(y)** :	
       Transform labels to normalized encoding.

### Feature Scaling ```(standardScaler)```

Applied for scaling of features of datasets, in case of being sparsely populated. This is usually not required for larger datasets with more features.


```python
def featureScale(dataset):
    return StandardScaler().fit_transform(dataset)
```
