from xgboost_tweak import *
from feature_engine import encoding, imputation
from sklearn import tree
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import dummy
import xgboost as xgb
import matplotlib.pyplot as plt

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

# underfit decision tree
underfit = tree.DecisionTreeClassifier(max_depth=1)

X_train = kag_pl.fit_transform(kag_X_train)

underfit.fit(X_train, kag_y_train)

underfit.score(X_test, kag_y_test)

# overfit decision tree
hi_variance = tree.DecisionTreeClassifier(max_depth=None)

hi_variance.fit(X_train, kag_y_train)

hi_variance.score(X_test, kag_y_test)

fig, ax = plt.subplots(figsize=(8, 4))
features = list(c for c in X_train.columns)
tree.plot_tree(hi_variance, feature_names=features, filled=True)
tree.plot_tree(hi_variance, feature_names=features, filled=True, 
                  class_names=list(hi_variance.classes_),
                  max_depth=2, fontsize=6)