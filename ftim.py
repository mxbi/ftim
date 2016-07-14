## Copyright (c) 2016 Mikel Bober-Irizar
## Feature-Time Instability Metric
# Script to evaluate the change in properties of features over time
# To use, call the function find_ovalue(FEATURE, TARGET)

import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool

########## CONFIG #########

# Resolution of time axis to test
time_res = 2
# Resolution of feature value binning
x_res = 200
# Higher values will find smaller fluctuations in data but may have more noise

# Threshold of values that have to be in a histogram bin for it to be considered:
thresh = 0.001

# Method to measure stabilitiy, either 'inter' for histogram intersection or 'purity' for tree split purity
method = 'purity'

# Only used for the purity metric, gives weight to the different splits based on how many samples exist in the bin
weighted = True
ignore_zero = False

######### END CONFIG ##########

def hist_inter(a, b, bins):
    # Uses Histogram intersection distance function to measure instability

    # Find range
    hist_max = max(max(a), max(b))
    hist_min = min(min(a), min(b))

    # np.histogram 'normed' is broken, normalisation must be done manually
    hist_a = np.histogram(a, bins=bins, range=(hist_min, hist_max), normed=False)[0].tolist()
    hist_b = np.histogram(b, bins=bins, range=(hist_min, hist_max), normed=False)[0].tolist()

    # Manual normalisation of histograms
    size_a = len(a)
    size_b = len(b)
    hist_a = [x / size_a for x in hist_a]
    hist_b = [x / size_b for x in hist_b]

    k = 0
    i = 0
    # Evaluate histogram intersection
    for d in zip(hist_a, hist_b):
        if sum(d) > thresh:
                k += min(d)
                i += 1

    return k

def hist_purity(a, b, target_a, target_b, bins, weighted=True, ignore_nan=False):

    # Get range of histogram to use
    hist_max = max(max(a), max(b))
    hist_min = min(min(a), min(b))

    hist = pd.DataFrame()
    hist['a'] = a
    hist['b'] = b
    hist['ta'] = target_a
    hist['tb'] = target_b

    # Separate data into labels
    a_true = hist.loc[hist.ta == 1]['a'].values#.tolist()
    a_false = hist.loc[hist.ta < 1]['a'].values#.tolist()
    b_true = hist.loc[hist.tb == 1]['b'].values#.tolist()
    b_false = hist.loc[hist.tb < 1]['b'].values#.tolist()

    # Compute histograms
    hist_a_true = np.histogram(a_true, bins=bins, range=(hist_min, hist_max), normed=False)[0]
    hist_a_false = np.histogram(a_false, bins=bins, range=(hist_min, hist_max), normed=False)[0]
    hist_b_true = np.histogram(b_true, bins=bins, range=(hist_min, hist_max), normed=False)[0]
    hist_b_false = np.histogram(b_false, bins=bins, range=(hist_min, hist_max), normed=False)[0]

    # Compute split purity
    hist_a_tot = hist_a_true + hist_a_false
    hist_b_tot = hist_b_true + hist_b_false
    hist_a_purity = hist_a_true / hist_a_tot
    hist_b_purity = hist_b_true / hist_b_tot

    if ignore_nan is False:
        hist_a_purity = np.nan_to_num(hist_a_purity)
        hist_b_purity = np.nan_to_num(hist_b_purity)

    # Compute histogram weights
    hist_weights = ((hist_a_true + hist_a_false) / np.sum(hist_a_true + hist_a_false)) + ((hist_b_true + hist_b_false) / np.sum(hist_b_true + hist_b_false)) / 2

    if weighted is False:
        k = np.nansum(np.abs(hist_a_purity - hist_b_purity)) / len(a + b)
    if weighted is True:
        k = np.nansum(np.abs(hist_a_purity - hist_b_purity) * hist_weights)

    #print(k)
    return k

def find_ovalue_inter(feature, target):
    ftr_true = []
    ftr_false = []

    # Separate into positive and negative samples
    for x in zip(feature, target):
        if x[1] == 1:
            ftr_true.append(x[0])
        else:
            ftr_false.append(x[0])

    # Split into time bins
    chunks_true = [x.tolist() for x in np.array_split(ftr_true, time_res)]
    chunks_false = [x.tolist() for x in np.array_split(ftr_false, time_res)]

    cross = []
    # Shoddy method for cross-checking chunks
    for x in chunks_true:
        for y in chunks_true:
            if x != y and y > x:
                dist = hist_inter(x, y, x_res)
                cross.append(dist)

    for x in chunks_false:
        for y in chunks_false:
            if x != y and y > x:
                dist = hist_inter(x, y, x_res)
                cross.append(dist)

    return 1 - (sum(cross) / len(cross))

def find_ovalue_purity(feature, target):
    feature_chunks = [x.tolist() for x in np.array_split(feature, time_res)]
    target_chunks = [x.tolist() for x in np.array_split(target, time_res)]

    cross = []
    # Shoddy method for cross-checking chunks
    for xi, x in enumerate(feature_chunks):
        for yi, y in enumerate(feature_chunks):
            if y > x:  # Avoid repeating the same chunk pair
                xt = target_chunks[xi]
                yt = target_chunks[yi]
                dist = hist_purity(x, y, xt, yt, x_res, weighted, ignore_zero)
                cross.append(dist)
    try:
        return sum(cross) / len(cross)
    except:
        # Will return -1 if there is an error
        return -1


if method == 'inter':
    find_ovalue = find_ovalue_inter
elif method == 'purity':
    find_ovalue = find_ovalue_purity
else:
    print("Method must be set to either 'inter' or 'purity'")

######## EVALUATION #########

# Load dataset into memory however you choose
df_train = pickle.load(open('/home/mikel/kaggle-avito/cache/ftrs_train.bin', 'rb'))

# Replace missing values
df_train = df_train.replace([np.inf, -np.inf], np.nan).fillna(0)

print('loaded')
# Function for evaluating features in turn and then writing result to csv
def process(c):
    # Here is the function that evaluates the feature, please provide it with a LIST
    # It returns a float which is the overfitting value
    o = find_ovalue(df_train[c].tolist(), df_train['isDuplicate'].tolist())
    print(c.ljust(60) + ' ' + str(o))
    f = open('overfit_valid_mikel_purity.csv', 'a')
    f.write(c+','+str(o)+'\n')
    f.close()

# Multithreading, because I am too impatient to wait 10 minutes :)
pool = Pool(12)
pool.map(process, df_train.columns.tolist())
