import xgboost_tweak as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.objects as so

from feature_engine import encoding, imputation
from sklearn import base, pipeline
from sklearn import model_selection

# url of the dataset
url = 'https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip'

fname = 'kaggle-survey-2018.zip'
member_name = 'multipleChoiceResponses.csv'

raw = xt.extract_zip(url, fname, member_name)

class TweakKagTransformer(base.BaseEstimator,
    base.TransformerMixin):
    """
    A transformer for tweaking Kaggle survey data.

    This transformer takes a Pandas DataFrame containing 
    Kaggle survey data as input and returns a new version of 
    the DataFrame. The modifications include extracting and 
    transforming certain columns, renaming columns, and 
    selecting a subset of columns.

    Parameters
    ----------
    ycol : str, optional
        The name of the column to be used as the target variable. 
        If not specified, the target variable will not be set.

    Attributes
    ----------
    ycol : str
        The name of the column to be used as the target variable.
    """
    
    def __init__(self, ycol=None):
        self.ycol = ycol
        
    def transform(self, X):
        return xt.tweak_kag(X)
    
    def fit(self, X, y=None):
        return self

kag_pl = pipeline.Pipeline(
    [('tweak', TweakKagTransformer()),
     ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True, 
           variables=['Q1', 'Q3', 'major'])),
     ('num_impute', imputation.MeanMedianImputer(imputation_method='median',
          variables=['education', 'years_exp']))]
    )

# run the code
kag_X, kag_y = xt.get_rawX_y(raw, 'Q6')

kag_X_train, kag_X_test, kag_y_train, kag_y_test = \
    model_selection.train_test_split(
        kag_X, kag_y, test_size=.3, random_state=42, stratify=kag_y
    )
    
X_train = kag_pl.fit_transform(kag_X_train, kag_y_train)
X_test = kag_pl.transform(kag_X_test)

# correlations
(X_train
 .assign(data_scientist = kag_y_train == 'Data Scientist')
 .corr(method = 'spearman')
 .style
 .background_gradient(cmap='RdBu', vmax=1, vmin=-1)
 .set_sticky(axis = 'index')
)

# bar plot
fig, ax = plt.subplots(figsize=(8, 4))

(X_train
 .assign(data_scientist = kag_y_train)
 .groupby('r')
 .data_scientist
 .value_counts()
 .unstack()
 .plot.bar(ax=ax)
)

fig, ax = plt.subplots(figsize=(8, 4))

(pd.crosstab(index = X_train['major_cs'], 
             columns = kag_y)
   .plot.bar(ax=ax)
)

fig, ax = plt.subplots(figsize=(8, 4))
(X_train
 .plot.scatter(x='years_exp', y='compensation', alpha=.3, ax=ax, c='purple')
)

# seaborn objects scatterplot 

fig = plt.figure(figsize=(8, 4))

(so.Plot(X_train.assign(title=kag_y_train),
         x = 'years_exp', 
         y = 'compensation',
         color = 'title')
 .add(so.Dots(alpha = .3, pointsize = 2), so.Jitter(x = .5, y = 10000))
 .add(so.Line(), so.PolyFit())
)

# facetting by country

(
    so.Plot(X_train
            .assign(
                title=kag_y_train,
                country=(X_train
                         .loc[:, 'Q3_United States of America': 'Q3_China']
                         .idxmax(axis='columns')
                    )
            ), x = 'years_exp', y = 'compensation', color = 'title')
    .facet('country')
    .add(so.Dots(alpha = 0.1, pointsize = 2, color = 'grey'), 
         so.Jitter(x = .5, y = 10000),
         col = None)
    .add(so.Dots(alpha = .5, pointsize = 2),
         so.Jitter(x = .5, y = 10000))
    .add(so.Line(pointsize = 1), so.PolyFit(order = 2))
    .scale(x=so.Continuous().tick(at=[0,1,2,3,4,5]))
    .limit(y=(-10_000, 200_000), x=(-1, 6))    
)

# Questions

## Use .corr to quantify the correlations in your data.

## already derived spearman rank correlation coefficient above,
## derive pearson correlation coefficient for numeric data
(X_train
 [['age', 'education', 'years_exp', 'compensation']]
 .corr(method='pearson')
 .style
 .background_gradient(cmap='Greens_r', vmax=1, vmin=-1)
)

# Use a scatter pot to visualize the correlations in your data.
## looks like age is also correlated with compensation

X_train.plot.scatter(x='age', y='compensation', alpha=.3, c='purple')

(so.Plot(
    X_train,
    x = 'age',
    y = 'compensation')
 .add(so.Dots(alpha = .3, pointsize = 2, color = 'purple'), so.Jitter(x = .5, y = 10000))
 .add(so.Line(color = 'blue'), so.PolyFit(order  = 1))
 .limit(x = (15, 65))
)

# Use .value_counts to quantify counts of categorical data.
(X_train[['python', 'r', 'sql']]
 .value_counts()
 .unstack()
 )

# Use a bar plot to visualize the counts of categorical data.
(X_train[['python', 'r', 'sql']]
 .value_counts()
 .unstack()
 .plot(kind = 'bar')
 )
