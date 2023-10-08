from xgboost_tweak import *
from feature_engine import encoding, imputation
from sklearn import tree
from sklearn import ensemble
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import dummy
from yellowbrick.model_selection import validation_curve
from sklearn import model_selection
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
import dtreeviz

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

# Random forest classifier model
rf = ensemble.RandomForestClassifier(
    random_state=42
)

rf.fit(X_train, kag_y_train)

rf.score(X_test, kag_y_test)

rf.get_params()

rf.estimators_

# 100 trees 
len(rf.estimators_)

# first tree
print(rf.estimators_[0])

## visualize the first tree
fig, ax = plt.subplots(figsize=(8, 4))
features = list(c for c in X_train.columns)
tree.plot_tree(rf.estimators_[0], 
               feature_names=features, 
               filled=True, 
               class_names=list(rf.classes_), 
               ax=ax,
               max_depth=2, fontsize=6)

# Creating a random forest tree with XGBoost
rf_xg = xgb.XGBRFClassifier(random_state=42)
rf_xg.fit(X_train, y_train)
rf_xg.score(X_test, y_test)

rf_xg.get_params()

xgb.plot_tree(rf_xg, num_trees=0, rankdir='LR')

viz = dtreeviz.model(rf_xg, X_train=X_train,
    y_train=y_train,
    target_name='Job', feature_names=list(X_train.columns), 
    class_names=['DS', 'SE'], tree_index=0)
viz.view(depth_range_to_display=[0,2])

fig, ax = plt.subplots(figsize=(10,4))    
viz = validation_curve(xgb.XGBClassifier(random_state=42),
    X=pd.concat([X_train, X_test], axis='index'),
    y=np.concatenate([y_train, y_test]),
    param_name='n_estimators', param_range=range(1, 100, 2),
    scoring='accuracy', cv=3, 
    ax=ax)

xgb_rf_19 = xgb.XGBRFClassifier(random_state=42, n_estimators=19)
xgb_rf_19.fit(X_train, y_train)
xgb_rf_19.score(X_test, y_test)
