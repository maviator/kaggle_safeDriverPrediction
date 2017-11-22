
# coding: utf-8

# In[1]:

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

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# In[2]:

# get home price train & test csv files as a DataFrame
train = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")

print (train.shape, test.shape)


# In[3]:

#train.head()


# In[4]:

#train.describe()


# ## Checking for missing values

# In[5]:


#Checking for missing data
#NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
#NAs[NAs.sum(axis=1) > 0]


# In[6]:

#train.dtypes


# ## Memory Usage

# In[7]:

#--- memory consumed by train dataframe ---
mem = train.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n')
#--- memory consumed by test dataframe ---
mem = test.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))


# In[8]:

def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['int']).columns)
    for col in float_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

change_datatype(train)
change_datatype(test) 


# In[9]:

#--- Converting columns from 'float64' to 'float32' ---
def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
change_datatype_float(train)
change_datatype_float(test)


# In[10]:

#--- memory consumed by train dataframe ---
mem = train.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n') 
#--- memory consumed by test dataframe ---
mem = test.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))


# ## Concatenating train and test

# In[11]:

train_labels = train.pop('target')
test_id = test.id

features = pd.concat([train, test], keys=['train', 'test'])
features.shape


# In[13]:

#--- memory consumed by train dataframe ---
mem = features.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))


# In[14]:

del train
del test

# ## Data preprocessing

# In[14]:

col_to_drop = features.columns[features.columns.str.startswith('ps_calc_')]
features = features.drop(col_to_drop, axis=1)


# In[15]:

features = features.replace(-1, np.nan)


# ## One Hot encoding of categorical variables

# In[17]:

cat_features = [a for a in features.columns if a.endswith('cat')]

# Getting Dummies from all categorical vars
for col in cat_features:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)


# In[18]:

#features.shape


# In[19]:

#features.dtypes


# ## Splitting train and test variables

# In[20]:

### Splitting features
train_features = features.loc['train'].drop('id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('id', axis=1).select_dtypes(include=[np.number]).values


# In[21]:

del features


# ## Stacking

# In[22]:

class EnsembleStack(object):
    def __init__(self, stacker, base_models):
        self.stacker = stacker
        self.base_models = base_models
        
    def fit_predict(self, train_features, train_target, test_features):
        X = np.array(train_features)
        y = np.array(train_target)
        T = np.array(test_features)
        
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        
        for i, clf in enumerate(self.base_models):
            clf.fit(X,y)
            S_train[:,i] = clf.predict_proba(X)[:,1]
            S_test[:,i] = clf.predict_proba(T)[:,1]
        
        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res
        


# ## Modelling

# In[23]:

# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['seed'] = 99


# In[24]:

lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 99


# In[25]:

lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['seed'] = 99


# In[26]:

lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)


# In[27]:

log_model = LogisticRegression()


# In[28]:

stack = EnsembleStack(log_model, (lgb_model, lgb_model2, lgb_model3))        


# In[ ]:

test_y = stack.fit_predict(train_features, train_labels, test_features)


# In[ ]:

print (test_y.shape)


# ## Submission

# In[ ]:

test_submit = pd.DataFrame({'id': test_id, 'target': test_y})
test_submit.shape
test_submit.head()
test_submit.to_csv('stacking_conv.csv', index=False)


# ## History

# - Benchmark: RF with the stock data (no data manipulation). Gini score: 0.184
# - Converted categorical variables to one-hot encoding. Gini score: 0.194
# - Increased number of trees in RF to 2000. Gini score: 0.227
# - Stacking using 3 lightgbm. Gini score: 0.282

# ## Remarks

# - Training takes a lot of time
# - Probability for target = 1 are in second column
# - Train off of jupyter and add Del statement to remove unecessary data
