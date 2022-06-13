#load packages
library(kableExtra)
library(caret)
library(dplyr)

# Setup
path = "/path/to/files/"
ts_string = paste0(format(Sys.time(), "%H%M%S_%Y%m%d"))

`%notin%` <- Negate(`%in%`)
set.seed(42)

IMPUTED_LABELED = '' # imputed training dataset
IMPUTED_UNLABELED = '' # imputed unlabeled test dataset

# Not shown: gold standard test dataset -> TEST_DATA
# end setup

# read in training data and unlabeled data
imp_train_label = read.csv(paste0(path, IMPUTED_LABELED) )
imp_train_label$labeled = 1
imp_train_unlabel = read.csv(paste0(path, IMPUTED_UNLABELED) )
imp_train_unlabel$labeled = NA

# partition by site
partition_80_20 = createDataPartition(imp_train_label$site,p = .8, list = FALSE)

# subset into training and validation
train_train = imp_train_label[partition_80_20, ]
validate = imp_train_label[-partition_80_20, ]

# VFILE1 -> validate to file for later merge step
write.csv(validate, paste0(path, "imputed_validate_24h.baseline_", ts_string, ".csv"))

# here we combine predictors for 80% training labeled and all unlabeled 
for_clusters = rbind(train_train,imp_train_unlabel)
num_obs_train = dim(for_clusters)[1]

# make dataset for labels and cluster assignments
label_cluster_data = data.frame(infec_orig = for_clusters$infection, labeled_orig = for_clusters$labeled, encounter_id = for_clusters$encounter_id)


# scale continuous vars, with example code below
scaled_cluster_data = scale(subset(for_clusters, select = -c(encounter_id, labeled, site, infection, loc_catM.1, loc_catM.2, loc_catM.3, loc_catM.4, loc_catM.6, max_vent)))
# add back in onehot
for_clusters_scaled = cbind(scaled_cluster_data, subset(for_clusters, select = c(loc_catM.1, loc_catM.2, loc_catM.3, loc_catM.4, loc_catM.6, max_vent)))

label_cluster_data$relabeled_inf_kmeans_75 = label_cluster_data$infec_orig
label_cluster_data$relabeled_inf_kmeans_90 = label_cluster_data$infec_orig

# then, write to disk three datasets: 
# INPUT_DATA -> scaled data for clustering
# MDATA1, MDATA2 -> two metadata datasets that will be required after clustering is completed

# INPUT_DATA
output_for_clusters = paste0('sepsisData_ctl.', ts_string, '.csv', row.names=FALSE)
write.csv(for_clusters_scaled, output_for_clusters, row.names=FALSE)

label_cluster_data_output = label_cluster_data[which(names(label_cluster_data) %in% c('infec_orig', 'labeled_orig', 'encounter_id'))]
# MDATA1
write.csv(label_cluster_data_output, 'label_cluster_data.df.csv')
# MDATA2
write.csv(label_cluster_data, 'label_cluster_data.archive.df.csv')
print('job done')

