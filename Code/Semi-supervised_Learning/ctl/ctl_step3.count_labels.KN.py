import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import hashlib

# Add MDATA3 file from ctl_step1.R here
df_output = pd.read_csv(MDATA3)

del df_output['Unnamed: 0']
del df_output['hashKey']
# Add INPUT_DATA file from ctl_step1.R here
df_data = pd.read_csv(INPUT_DATA)
df_colNames = list(df_data)
df_colNames.append('hashString_x')
verboseBool = False
maxK = 2 # this is the k for the clustering
str_maxK = 'k' + str(maxK)
float_threshold = 0.75 # set cutoff threshold here
str_threshold = str(float_threshold) 

def createUniqueHashColumn(df):
    hashKeyList = []
    for idx,row in df.iterrows():
        l = row['hashString']
        idx_ = str(idx)
        l += idx_
        hashKeyList.append(hashlib.md5(l.encode('utf-8')).hexdigest())
    return hashKeyList

def createHashKeyList(df):
    hash_list = []
    hashKey_list = []
    idxInts = []
    for idx,row in df.iterrows():
        idxInts.append(idx)
        l = row.tolist()
        idx_ = str(idx)
        s = [str(x) for x in l]
        hash_list.append(hashlib.md5(','.join(s).encode('utf-8')).hexdigest())
        s_unique = [str(x) for x in l]
        s_unique.append(idx_)
        hashKey_list.append(hashlib.md5(','.join(s_unique).encode('utf-8')).hexdigest())
    return (hash_list, hashKey_list, idxInts)

df_output_hkl = createUniqueHashColumn(df_output)
df_output['hashKey'] = df_output_hkl
hlist, tmp, indices = createHashKeyList(df_data)
df_data['hashString'] = hlist
df_data['indices'] = indices
hklist = createUniqueHashColumn(df_data)
df_data['hashKey'] = hklist

# merge output and data
df_final = pd.merge(df_data, df_output, how='inner', on='hashKey')

# select columns to keep
df_colNames.append('clusterIdxs')
df_colNames.append('indices')
df_final_output = df_final[df_colNames]

# import MDATA1 from ctl_step1.R, then merge with output data
labels_R = pd.read_csv(MDATA1)
del labels_R['Unnamed: 0']
indicesList = [x for x in range(0, len(labels_R))]
labels_R['indices'] = indicesList
df_labels = pd.merge(df_final_output, labels_R, how='inner', on='indices')
tmp = [-1]*len(df_labels)
df_labels.loc[:,'updatedLabel'] = tmp

# create a list of clusters, and run a validation check
# grabIndices = [2,3,4]
# df_labels.loc[df_labels['indices'].isin(grabIndices)]

listOfClusters = []
k_rangeLimit = maxK + 1 # maxK + 1
for i in range(0, k_rangeLimit): 
    listOfClusters.append(df_labels.loc[df_labels['clusterIdxs'] == i,])
verboseBool = True
# validation section
rangeCounter = 0
for i in listOfClusters:
    rangeCounter += int(len(i))
if verboseBool == True:
    print('created groups with ' + str(rangeCounter) + ' total items')
    print('dataframe has ' + str(len(df_labels)) + ' total items')
    print('there should be ' + str(len(df_final_output)) + ' total datapoint rows')
else:
    validationInt = int(len(df_labels)) - rangeCounter
    if not validationInt:
        print('Failed validation, mismatch in number of datapoint rows.')
    else:
        print('Passed validation for number of datapoint rows.')
# end validation section

updated_labelList = []
inf_labeled_list = []
for idx in range(0, len(listOfClusters)):
    c = listOfClusters[idx].copy()
    print('for cluster at index ' + str(idx))
    inf_count = len(c.loc[c['infec_orig'] == 1,])
    uninf_count = len(c.loc[c['infec_orig'] == 0,])
    total_labeled = inf_count + uninf_count
    if total_labeled == 0:
        print('insufficient labeled in cluster\n')
        inf_labeled_list.append((idx, inf_count, uninf_count, -1, False))
        continue
    inf_percent = inf_count/total_labeled
    if inf_percent >= 0.75:
        print('majority is infected')
        print('at ' + str(round(inf_percent, 3)))
        c.loc[c['infec_orig'] == 0.0, 'updatedLabel'] = 0.0
        c.loc[c['infec_orig'] != 0.0, 'updatedLabel'] = 1.0
        updated_labelList.append(c)
        inf_labeled_list.append((idx, inf_count, uninf_count, round(inf_percent, 3), True))
    elif inf_percent <= 0.25:
        print('majority is uninfected')
        print('at ' + str(round(inf_percent, 3)))
        inf_labeled_list.append((idx, inf_count, uninf_count, round(inf_percent, 3), True))
        c.loc[c['infec_orig'] == 1.0, 'updatedLabel'] = 1.0
        c.loc[c['infec_orig'] != 1.0, 'updatedLabel'] = 0.0
        updated_labelList.append(c)
    else:
        print('no consensus')
        updated_labelList.append(c)
        inf_labeled_list.append((idx, inf_count, uninf_count, round(inf_percent, 3), False))
    print('\n')

df_labeled = pd.concat(updated_labelList)
# finally, output the counts as a csv file, then output the labeled dataset 
# as a csv file
# print(inf_labeled_list)
df_labeling_k = [x[0] for x in inf_labeled_list]
df_labeling_infct = [x[1] for x in inf_labeled_list]
df_labeling_uninfct = [x[2] for x in inf_labeled_list]
df_labeling_pct = [x[3] for x in inf_labeled_list]
df_labeling_bool = [x[4] for x in inf_labeled_list]
df_labeling = pd.DataFrame({'cluster_index' : df_labeling_k, 'infected_count' : df_labeling_infct, 'uninf_ct' : df_labeling_uninfct, 'pct_inf_labeled' : df_labeling_pct, 'labels_updated' : df_labeling_bool})
outFileName = 'infected_labeling_updates.' + str_threshold + '.' + str_maxK + '.csv'
# debug output only, can be ignored
df_labeling.to_csv(outFileName, index=False)

# KLABELS
outFileName = 'updated_dataset.' + str_threshold + '.' + str_maxK + '.csv'
df_labeled.to_csv(outFileName, index=False)
