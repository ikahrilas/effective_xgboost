import dtreeviz
from feature_engine import encoding, imputation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import base, compose, datasets, ensemble, \
    metrics, model_selection, pipeline, preprocessing, tree
# import scikitplot
import xgboost as xgb
import yellowbrick.model_selection as ms
from yellowbrick import classifier
from sklearn import tree
from sklearn import ensemble
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import dummy
from xgboost_tweak import *
import urllib
import zipfile
# import xg_helpers as xhelp

# reusable function to extract zip files
url = 'https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip'

fname = 'kaggle-survey-2018.zip'
member_name = 'multipleChoiceResponses.csv'

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

label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(kag_y_train)
y_test = label_encoder.transform(kag_y_test)

X = pd.concat([X_train, X_test], axis='index')
y = pd.Series([*y_train, *y_test], index=X.index)

# classifier model
xg_oob = xgb.XGBClassifier()
xg_oob.fit(X_train, y_train)
xg_oob.score(X_test, y_test)

# tuning the modle
xg2 = xgb.XGBClassifier(max_depth = 2, n_estimators = 2)
xg2.fit(X_train, y_train)
xg2.score(X_test, y_test)

import dtreeviz

viz = dtreeviz.model(xg2, X_train=X, y_train=y,
                     target_name='job', 
                     feature_names=list(X_train.columns),
                     class_names=['DS', 'SE'], tree_index=1)
viz.view(depth_range_to_display=[0,5])

xgb.plot_tree(xg2, num_trees = 0)

# using the .predict_proba attribute

se7894 = pd.DataFrame({'age': {7894: 22},                                            
'education': {7894: 16.0},
'years_exp': {7894: 1.0},
'compensation': {7894: 0},
'python': {7894: 1},
'r': {7894: 0},
'sql': {7894: 0},
'Q1_Male': {7894: 1},                                   
'Q1_Female': {7894: 0},
'Q1_Prefer not to say': {7894: 0},
'Q1_Prefer to self-describe': {7894: 0},
'Q3_United States of America': {7894: 0},
'Q3_India': {7894: 1},
'Q3_China': {7894: 0},
'major_cs': {7894: 0},
'major_other': {7894: 0},
'major_eng': {7894: 0},
'major_stat': {7894: 0}})
xg2.predict_proba(se7894)

xg2.predict(pd.DataFrame(se7894))

# plotting the second tree
xgb.plot_tree(xg2, num_trees = 1)

# early stopping
xg = xgb.XGBClassifier(early_stopping_rounds=20)
xg.fit(X_train, y_train,
       eval_set = [(X_train, y_train),
                   (X_test, y_test)])

xg.best_iteration

results = xg.evals_result()

fig, ax = plt.subplots(figsize=(8,4))

(
    pd.DataFrame({'training': results['validation_0']['logloss'],
                  'testing': results['validation_1']['logloss']})
    .assign(ntrees = lambda adf: range(1, len(adf)+1))
    .set_index('ntrees')
    .plot(figsize=(5,4), ax = ax,
          title = 'eval_results with early_stopping')
)
ax.annotate('Best number \nof trees (14)', xy=(14, .498),
            xytext=(20,.42), arrowprops={'color':'k'})
ax.set_xlabel('ntrees')

# training the moel with 14 trees
xg14 = xgb.XGBClassifier(n_estimators = 14)

xg14.fit(X_train, y_train,
         eval_set = [(X_train, y_train),
                     (X_test, y_test)])
xg14.score(X_test, y_test)

# different error metrics
xg_err = xgb.XGBClassifier(early_stopping_rounds = 20,
                           eval_metric = 'error')

xg_err.fit(X_train, y_train,
           eval_set = [(X_train, y_train),
                       (X_test, y_test)])

xg_err.best_iteration