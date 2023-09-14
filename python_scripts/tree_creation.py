import numpy as np
import numpy.random as rn
import pandas as pd
import matplotlib.pyplot as plt

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