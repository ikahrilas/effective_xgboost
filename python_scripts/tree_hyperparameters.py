from xgboost_tweak import *
from feature_engine import encoding, imputation
from sklearn import tree
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

# examine parameters of DecisionTreeClassifier constructor
tree_class = tree.DecisionTreeClassifier()

tree_class.get_params()

# creating a validation curve
accuracies = []
for depth in range(1, 15):
    between = tree.DecisionTreeClassifier(max_depth = depth)
    between.fit(X_train, kag_y_train)
    accuracies.append(between.score(X_test, kag_y_test))
    
print(accuracies)

fig, ax = plt.subplots(figsize=(10, 4))
(pd.Series(accuracies, name='Accuracy', index=range(1, len(accuracies)+1))
 .plot(ax=ax, title='Accuracy at a given Tree Depth'))
ax.set_ylabel('Accuracy')
ax.set_xlabel('max_depth')

accuracies_df = pd.DataFrame(
    {'accuracies':accuracies,
     'depth':np.arange(1,15)}
)

(
    so.Plot(accuracies_df,
            x='depth',
            y='accuracies')
    .add(so.Line())
)

# Looks like a depth of 7 maximizes our accuracy
between = tree.DecisionTreeClassifier(max_depth = 7)

between.fit(X_train, kag_y_train)

between.score(X_test, kag_y_test)

fig, ax = plt.subplots(figsize=(10,4))    
viz = validation_curve(tree.DecisionTreeClassifier(),
    X=pd.concat([X_train, X_test]),
    y=pd.concat([kag_y_train, kag_y_test]),   
    param_name='max_depth', param_range=range(1,14),
    scoring='accuracy', cv=5, ax=ax, n_jobs=6)

# grid search for cross validation
params = {
    'max_depth':[3, 5, 7, 8],
    'min_samples_leaf':[1, 3, 4, 5, 6],
    'min_samples_split':[2, 3, 4, 5, 6],
}

grid_search = model_selection.GridSearchCV(
    estimator = tree.DecisionTreeClassifier(),
    param_grid = params,
    cv = 4,
    n_jobs = 1,
    verbose = 1,
    scoring = 'accuracy'
)

grid_search.fit(
    pd.concat([X_train, X_test]),
    pd.concat([kag_y_train, kag_y_test])
)

# check out the best parameters and score
grid_search.best_params_

# use the best parameters to construct a new model
between2 = tree.DecisionTreeClassifier(
    **grid_search.best_params_
)

between2.fit(X_train, kag_y_train)

between2.score(X_test, kag_y_test)

(pd.DataFrame(grid_search.cv_results_)
 .sort_values(by='rank_test_score')
 .style
 .background_gradient(axis='rows')
 )


results = model_selection.cross_val_score(
   tree.DecisionTreeClassifier(max_depth=7),
   X=pd.concat([X_train, X_test], axis='index'),
   y=pd.concat([kag_y_train, kag_y_test], axis='index'),
   cv=4
)

results.mean()