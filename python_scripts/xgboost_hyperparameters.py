import dtreeviz
from feature_engine import encoding, imputation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import base, compose, datasets, ensemble, \
    metrics, model_selection, pipeline, preprocessing, tree
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

# Tuning gamma hyperparameter
fig, ax = plt.subplots(figsize = (8, 4))
ms.validation_curve(xgb.XGBClassifier(), X_train, y_train,
                    param_name='gamma',
                    param_range=[0, .5, 1, 5, 10, 20, 30],
                    n_jobs=-1, ax=ax)