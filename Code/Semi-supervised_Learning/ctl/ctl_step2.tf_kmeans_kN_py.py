import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import hashlib


# INPUT_DATA file from previous step goes here
df = pd.read_csv(INPUT_DATA)

# any k can be used when clustering
# as an example, k == 2 is used below

# create hashes to map back to the datapoints
hash_list = []
hash_ints = [x for x in range(1, (len(df)+1))]
colIdxs = len(df.columns)
hashKey_list = []
for idx,row in df.iterrows():
    l = row.tolist()
    idx_ = str(idx)
    s_unique = [str(x) for x in l]
    s_unique.append(idx_)
    hashKey_list.append(hashlib.md5(','.join(s_unique).encode('utf-8')).hexdigest())
    s = [str(x) for x in l]
    h = hashlib.md5(','.join(s).encode('utf-8')).hexdigest()
    hash_list.append(h)

df_colNames = list(df)
df['hashString'] = hash_list
df['hashInt'] = hash_ints
df['hashKey'] = hashKey_list
df_data = df.iloc[:, 0:colIdxs]
df_colNames.append('hashInt')
df_colNames.append('hashString_x')

# now run KMeans clustering via TensorFlow v1.x
point_mapping = [] # list of tuples with (point, cluster_index, cluster_center)
sess = tf.compat.v1.Session()
with sess.as_default():
    points = np.array(df_data)
    def input_fn():
      return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

    num_clusters = 2
    kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)
    # train

    kmeans_scores = []
    previous_centers = None
    for _ in range(10):
        kmeans.train(input_fn)
        cluster_centers = kmeans.cluster_centers()
        previous_centers = cluster_centers

    # link datapoints to index in np.array()
    cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    for i, point in enumerate(points):
        cluster_index = cluster_indices[i]
        center = cluster_centers[cluster_index]
        point_mapping.append((point, cluster_index, center))
    sess.close()

# now, map the cluster identities to the data points
point_mapping_list = []
point_mapping_points = []
point_mapping_clusterIdxs = []
point_mapping_hkey = []
for i in point_mapping:
    pt_list = i[0].tolist()
    pt_list_asString = [str(x) for x in pt_list]
    h = hashlib.md5(','.join(pt_list_asString).encode('utf-8')).hexdigest()
    s_unique = [str(x) for x in i]
    s_unique.append(idx_)
    point_mapping_hkey.append(hashlib.md5(','.join(s_unique).encode('utf-8')).hexdigest())
    point_mapping_list.append(h)
    point_mapping_points.append(i[0])
    point_mapping_clusterIdxs.append(i[1])

df_output = pd.DataFrame(point_mapping_points)
df_output['hashString'] = point_mapping_list
df_output['clusterIdxs'] = point_mapping_clusterIdxs
df_output['hashKey'] = point_mapping_hkey
df_colNames.append('clusterIdxs')
# MDATA3 
df_output.to_csv('df_output_archive.k2.csv')

# run SQL-style inner merge to finish mapping
df_final = pd.merge(df, df_output, how='inner', on='hashKey')
df_final = df_final[df_colNames]

# output mapped KMeans file to disk
df_final.to_csv('kmeans_k2_clustering.csv')
print('job done')
