import xgboost_tweak as xt

# url of the dataset
url = 'https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip'

fname = 'kaggle-survey-2018.zip'
member_name = 'multipleChoiceResponses.csv'

raw = xt.extract_zip(url, fname, member_name)

