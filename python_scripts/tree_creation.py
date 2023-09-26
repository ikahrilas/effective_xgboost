import numpy as np
import numpy.random as rn
import pandas as pd
import matplotlib.pyplot as plt
import dtreeviz
import urllib.request
import zipfile
from feature_engine import encoding, imputation
from sklearn import base, pipeline
from sklearn import model_selection

from sklearn import tree
from sklearn import dummy
from sklearn import preprocessing
import xgboost as xgb
from xgboost_tweak import my_dot_export
from xgboost_tweak import TweakKagTransformer
from xgboost_tweak import get_rawX_y
from xgboost_tweak import extract_zip


# generate random samples using numpy
pos_center = 12
pos_count = 100
neg_center = 7
neg_count = 1000

rs = rn.RandomState(rn.MT19937(rn.SeedSequence(42)))

# simulate data set
gini = pd.DataFrame(
    {
        'value':np.append(
            pos_center + rs.randn(pos_count),
            neg_center + rs.randn(neg_count)
        ),
        'label':['pos'] * pos_count + ['neg'] * neg_count
    }
)

# plot
fig, ax = plt.subplots(figsize=(8, 4))

(gini
 .groupby('label')
 [['value']]
     .plot.hist(bins=30, alpha=.5, ax=ax, edgecolor='black')
)

# gini index function

def calc_gini(df, val_col, label_col, pos_val, split_point,
              debug=False):
    """
    This function calculates the Gini impurity of a dataset. Gini impurity 
    is a measure of the probability of a random sample being classified 
    incorrectly when a feature is used to split the data. The lower the 
    impurity, the better the split.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    val_col (str): The column name of the feature used to split the data
    label_col (str): The column name of the target variable
    pos_val (str or int): The value of the target variable that represents 
        the positive class
    split_point (float): The threshold used to split the data.
    debug (bool): optional, when set to True, prints the calculated Gini
        impurities and the final weighted average

    Returns:
    float: The weighted average of Gini impurity for the positive and 
        negative subsets.
    """    
    ge_split = df[val_col] >= split_point
    eq_pos = df[label_col] == pos_val
    tp = df[ge_split & eq_pos].shape[0]
    fp = df[ge_split & ~eq_pos].shape[0]
    tn = df[~ge_split & ~eq_pos].shape[0]
    fn = df[~ge_split & eq_pos].shape[0]
    pos_size = tp+fp
    neg_size = tn+fn
    total_size = len(df)
    if pos_size == 0:
        gini_pos = 0
    else:
        gini_pos = 1 - (tp/pos_size)**2 - (fp/pos_size)**2
    if neg_size == 0:
        gini_neg = 0
    else:
        gini_neg = 1 - (tn/neg_size)**2 - (fn/neg_size)**2
    weighted_avg = gini_pos * (pos_size/total_size) + \
                   gini_neg * (neg_size/total_size)
    if debug:
        print(f'{gini_pos=:.3} {gini_neg=:.3} {weighted_avg=:.3}')
    return weighted_avg

# test the function
calc_gini(gini, val_col='value', 
          label_col='label', 
          pos_val='pos',
          split_point=9.24, 
          debug=True)

# loop over possible values for the split point and calculate the Gini impurity
values = np.arange(5, 15, .1)
ginis = []

for v in values:
    ginis.append(calc_gini(
        gini, 
        val_col='value',
        label_col='label',
        pos_val='pos',
        split_point=v
    ))
    
# plot the gini values
fig, ax = plt.subplots(figsize=(8, 4))    
ax.plot(values, ginis, color = 'purple')
ax.set_title('Gini Coefficient')
ax.set_ylabel('Gini Coefficient')
ax.set_xlabel('Split Point')

# looks like 10 is the optimal split point
pd.Series(ginis, index=values).loc[9.5:10.5]

# values of 9.9 to 10.2 will minimize the Gini coefficient
pd.DataFrame({'gini':ginis, 'split':values}).query('gini == gini.min()')

# Making a decision tree with a single node 
stump = tree.DecisionTreeClassifier(max_depth = 1)

stump.fit(gini[['value']], gini['label'])

fig, ax = plt.subplots(figsize = (8,4))
tree.plot_tree(stump, feature_names = ['value'],
               filled = True,
               class_names = list(stump.classes_),
               ax = ax)

## now using XGBoost
xg_stump = xgb.XGBClassifier(n_estiamtors = 1, max_depth = 1)
xg_stump.fit(gini[['value']], gini.label == 'pos')

xgb.plot_tree(xg_stump, num_trees = 0)

my_dot_export(xg_stump, num_trees=0, filename='img/stump_xg.dot', title='A demo stump')

viz = dtreeviz.model(xg_stump, X_train=gini[['value']], 
                     y_train=gini.label=='pos',
    target_name='positive',
    feature_names=['value'], class_names=['negative', 'positive'],
    tree_index=0)
viz.view()

## using kaggle data for our stump ##
# reusable function to extract zip files
url = 'https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip'

fname = 'kaggle-survey-2018.zip'
member_name = 'multipleChoiceResponses.csv'

extract_zip(url, fname, member_name)

## Create a pipeline
kag_pl = pipeline.Pipeline(
    [('tweak', TweakKagTransformer()),
     ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True, 
           variables=['Q1', 'Q3', 'major'])),
     ('num_impute', imputation.MeanMedianImputer(imputation_method='median',
          variables=['education', 'years_exp']))]
    )

raw = extract_zip(url, fname, member_name)

# run the code
kag_X, kag_y = get_rawX_y(raw, 'Q6')

kag_X_train, kag_X_test, kag_y_train, kag_y_test = \
    model_selection.train_test_split(
        kag_X, kag_y, test_size=.3, random_state=42, stratify=kag_y
    )
    
X_train = kag_pl.fit_transform(kag_X_train, kag_y_train)
X_test = kag_pl.transform(kag_X_test)

stump_dt = tree.DecisionTreeClassifier(max_depth=1)
X_train = kag_pl.fit_transform(kag_X_train)
stump_dt.fit(X_train, kag_y_train)

fig, ax = plt.subplots(figsize=(8, 4))
features = list(c for c in X_train.columns)
tree.plot_tree(stump_dt, feature_names=features, 
               filled=True, 
               class_names=list(stump_dt.classes_),
               ax=ax)

X_test = kag_pl.transform(kag_X_test)

X_test

stump_dt.score(X_test, kag_y_test)

# dummy classifier for comparison
dummy_model = dummy.DummyClassifier()
dummy_model.fit(X_train, kag_y_train)

dummy_model.score(X_test, kag_y_test)
## 0.55, so quite a bit worse than our stump

# use xgboost with a single tree
kag_stump = xgb.XGBClassifier(n_estimators=1, max_depth=1)
kag_stump.fit(X_train, kag_y_train)
## throws error! need to handle labels

label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(kag_y_train)
y_test = label_encoder.transform(kag_y_test)

label_encoder.classes_
## 0 = Data Scientist, 1 = Software Engineer 

kag_stump.fit(X_train, y_train)
kag_stump.score(X_test, y_test)
## .55, same as dummy classifier

my_dot_export(kag_stump, num_trees = 0, filename = 'img/xgb_kag_stump.dot', title = 'XGBoost stump')

fig, ax = plt.subplots(figsize=(8, 4))
tree.plot_tree(kag_stump, feature_names=features, 
               filled=True, 
               class_names=list(label_encoder.classes_),
               ax=ax)