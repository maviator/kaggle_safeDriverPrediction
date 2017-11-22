
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

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
get_ipython().magic(u'matplotlib inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


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


# ## Concatenating train and test

# In[7]:

train_labels = train.pop('target')
test_id = test.id

features = pd.concat([train, test], keys=['train', 'test'])
features.shape

del train
del test


# ## Converting categorical variables' type to str

# In[8]:

for col in features.columns:
    if col[-3:] == "cat":
        features[col] = features[col].astype(str)


# In[9]:

#features.dtypes


# ## One Hot encoding of categorical variables

# In[10]:

# Getting Dummies from all categorical vars
for col in features.dtypes[features.dtypes == 'object'].index:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)


# In[11]:

#features.shape


# ## Splitting train and test variables

# In[12]:

### Splitting features
train_features = features.loc['train'].drop('id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('id', axis=1).select_dtypes(include=[np.number]).values

del features

# ## Stacking

# In[27]:

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

# In[14]:

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


# In[15]:

lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 99


# In[16]:

lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['seed'] = 99


MAX_ROUNDS = 300
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50

xgbc_params = {}
xgbc_params['n_estimators'] = MAX_ROUNDS
xgbc_params['max_depth'] = 4
xgbc_params['objective'] = "binary:logistic"
xgbc_params['learning_rate'] = LEARNING_RATE
xgbc_params['subsample'] = .8
xgbc_params['min_child_weight'] = 6
xgbc_params['colsample_bytree'] = .8
xgbc_params['scale_pos_weight'] = 1.6
xgbc_params['gamma'] = 10
xgbc_params['reg_alpha'] = 8
xgbc_params['reg_lambda'] = 1.3

# In[17]:

lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

xgbc_model = XGBClassifier(**xgbc_params)

# In[18]:

log_model = LogisticRegression()


# In[31]:

stack = EnsembleStack(log_model, (lgb_model, lgb_model2, lgb_model3, xgbc_model))        


# In[32]:

test_y = stack.fit_predict(train_features, train_labels, test_features)

# In[33]:

test_y.shape


# ## Submission

# In[30]:

test_submit = pd.DataFrame({'id': test_id, 'target': test_y})
test_submit.shape
test_submit.head()
test_submit.to_csv('stacking_xbgc.csv', index=False)


# ## History

# - Benchmark: RF with the stock data (no data manipulation). Gini score: 0.184
# - Converted categorical variables to one-hot encoding. Gini score: 0.194
# - Increased number of trees in RF to 2000. Gini score: 0.227

# ## Remarks

# - Training takes a lot of time
# - Probability for target = 1 are in second column
# - Train off of jupyter and add Del statement to remove unecessary data
