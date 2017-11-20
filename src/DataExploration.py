
# coding: utf-8

# In[12]:

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
#import seaborn as sns

# Configure visualisations
#get_ipython().magic(u'matplotlib inline')
#mpl.style.use( 'ggplot' )
#sns.set_style( 'white' )
#pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# In[13]:

# get home price train & test csv files as a DataFrame
train = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")
full = train.append(test, ignore_index=True)
print (train.shape, test.shape, full.shape)


# In[14]:

train.head()


# In[15]:

train.describe()


# In[16]:


#Checking for missing data
NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
NAs[NAs.sum(axis=1) > 0]


# In[17]:

train.dtypes


# In[18]:

train_labels = train.pop('target')
test_id = test.id

features = pd.concat([train, test], keys=['train', 'test'])
features.shape

del train
del test

# In[19]:

for col in features.columns:
    if col[-3:] == "cat":
        features[col] = features[col].astype(str)


# In[20]:

features.dtypes


# In[21]:

# Getting Dummies from all categorical vars
for col in features.dtypes[features.dtypes == 'object'].index:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)


# In[22]:

print (features.shape)


# In[23]:

### Splitting features
train_features = features.loc['train'].drop('id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('id', axis=1).select_dtypes(include=[np.number]).values

del features

# In[17]:

# RFC Parameters tunning 
RFC = RandomForestClassifier(n_estimators=2000, n_jobs=-1, verbose=1)

RFC.fit(train_features, train_labels)

print ("Done Training")

# In[26]:

test_y = RFC.predict_proba(test_features)


# In[27]:

test_y.shape


# In[28]:

test_y = test_y[:,1]


# In[29]:

print (test_y.shape)


# In[30]:
print ("Starting Submission")


test_submit = pd.DataFrame({'id': test_id, 'target': test_y})
test_submit.shape
test_submit.head()
test_submit.to_csv('safe_driver_rf.csv', index=False)


# ## History

# - Benchmark: RF with the stock data (no data manipulation). Gini score: 0.184

# ## Remarks

# - Training takes a lot of time
# - Probability for target = 1 are in second column
