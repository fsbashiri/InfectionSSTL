library(MASS)
library(plyr)
library(dplyr)
library(pROC)
library(caret)
library(ROCR)
library(xgboost)
library(haven)
library(spatstat.utils)
library(imputeTS)
library(purrr)


# Setup

`%notin%` <- Negate(`%in%`)
ts_string = paste0(format(Sys.time(), "%H%M%S_%Y%m%d"))

# Other Setup
train_inFile = TTRAIN
validate_inFile = TVAL
# modify the filename of the finished RData file as needed
rdata_file = paste0('ctl_xgb.Kmeans.k4.csv')
set.seed(42)
# End Setup

# Workflow
ds_training_fc_24h = read.csv(train_inFile)
ds_validation_24h = read.csv(validate_inFile)


# Get prelim stats
inf_count = nrow(ds_training_fc_24h[which(ds_training_fc_24h$updatedLabel == c('infectedYes')),])
uninf_count = nrow(ds_training_fc_24h[which(ds_training_fc_24h$updatedLabel == c('infectedNo')),])


print('starting workflow, updated labels have infected count:')
print(inf_count)
print('updated labels have uninfected count:')
print(uninf_count)

# preprocessing
# validation: remove encounter_id, site
# training: remove c('hashString_x', 'clusterIdxs', 'indices', 'infec_orig', 'encounter_id')
# training: assign labeled_orig as infection

# ds_training_24h_onlyLabeled = ds_training_fc_24h[!is.na(ds_training_fc_24h$labeled_orig),]
ds_training_24h_onlyLabeled = ds_training_fc_24h[which(ds_training_fc_24h$updatedLabel != -1),]
ds_training_24h_onlyLabeled = ds_training_24h_onlyLabeled[ -which(names(ds_training_24h_onlyLabeled) %in% c('hashString_x', 'clusterIdxs', 'indices', 'infec_orig', 'encounter_id', 'labeled_orig'))]
ds_training_24h = apply(ds_training_24h_onlyLabeled, 2, as.numeric)
ds_training = as.data.frame(ds_training_24h)
ds_training$infection = ifelse(ds_training$updatedLabel == 1, "infectedYes", "infectedNo")
ds_training$infection = as.character(ds_training$infection)
ds_training = ds_training[ -which(names(ds_training) %in% c('updatedLabel'))]

# downsample
ds_training$infected = factor(ds_training$infection)
downsampled_data = downSample(x=ds_training, y=ds_training$infected, yname="outcome01")
ds_training = downsampled_data[ which(names(downsampled_data) %notin% c('outcome01', 'infected'))]

inf_count = nrow(ds_training[which(ds_training$infection == c('infectedYes')),])
uninf_count = nrow(ds_training[which(ds_training$infection == c('infectedNo')),])

# Show stats on downsampling
print('after downsampling, infection labels have infected count:')
print(inf_count)
print('infection labels have uninfected count:')
print(uninf_count)


ds_validation_m = apply(ds_validation_24h, 2, as.numeric)
ds_validation = as.data.frame(ds_validation_m)
ds_validation = ds_validation[ -which(names(ds_validation) %in% c("encounter_id", "site", "labeled", "X"))]
ds_validation$infection = ifelse(ds_validation$infection == 1, "infectedYes", "infectedNo")
ds_validation$infection = as.character(ds_validation$infection)

# Build XGB model using validation and training data
x_train_train = ds_training
x_train_test = ds_validation

cvCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = TRUE)
print('finished cvCtrl')
xgbGrid<-expand.grid(nrounds=c(500, 1000, 1500, 2000), max_depth=c(2, 5, 10, 15), eta=c(0.001, 0.01, 0.1), gamma = c(0), colsample_bytree = c(1), min_child_weight = c(2, 5), subsample = c(0.5))
print('finished xgbGrid')
xgbGridmod<- train(infection~., method = "xgbTree", data = x_train_train, metric = "ROC", tree_method = "gpu_hist", gpu_id = 0, trControl = cvCtrl, tuneGrid = xgbGrid, na.action = na.pass)
print('finished xgbGridmod')
xgbGridProbs<-predict(xgbGridmod, newdata=x_train_test, type="prob", na.action = na.pass)
xgbGridProbsDF<-as.data.frame(xgbGridProbs)

set.seed(42)
enetGridRoc_baseline<-roc(x_train_test$infection, round(xgbGridProbsDF$infectedYes, digits=4))


print('validation AUC is')
print(enetGridRoc_baseline$auc)


# Finally, build XGB Model with combined 
# all labeled and relabeled data

x_train = bind_rows(x_train_train, x_train_test)

cvCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = TRUE)
print('finished cvCtrl')
xgbGrid<-expand.grid(nrounds=c(500, 1000, 1500, 2000), max_depth=c(2, 5, 10, 15), eta=c(0.001, 0.01, 0.1), gamma = c(0), colsample_bytree = c(1), min_child_weight = c(2, 5), subsample = c(0.5))
xgbGridmod<- train(infection~., method = "xgbTree", data = x_train, metric = "ROC", tree_method = "gpu_hist", gpu_id = 0, trControl = cvCtrl, tuneGrid = xgbGrid, na.action = na.pass)


ds_testing_fc_xgb_24h = read.csv(TEST_DATA)
# format testing data to match columns in training data, 
# and to conform to caret requirements for predictor column. Example usage is shown here:
ds_testing_fc_xgb_24h = ds_testing_fc_xgb_24h[which(names(ds_testing_fc_xgb_24h) %notin% c('loc_catM', 'encounter_id', 'suspected_infection', 'hours_since_admit', 'sepsis3', 'sepsis3_bool', 'site', 'trainingFlag'))]
ds_testing_fc_xgb_24h$infectedV = ifelse(ds_testing_fc_xgb_24h$infection == 1,"infectedYes","infectedNo")
x_test = ds_testing_fc_xgb_24h[ -which(names(ds_testing_fc_xgb_24h) %in% c('infection', 'site', 'loc_catM', 'encounter_id'))]

xgbGridProbs<-predict(xgbGridmod, newdata=x_test, type="prob", na.action = na.pass)
xgbGridProbsDF<-as.data.frame(xgbGridProbs)

# calculate AUC
xgbGridRoc_baseline<-roc(x_test$infectedV, round(xgbGridProbsDF$infectedYes, digits=4))
print(xgbGridRoc_baseline)

xgb_Probs = xgbGridProbs[,2]
x_test$infectedBinary = ifelse(x_test$infectedV == c('infectedYes'), 1, 0)
x_test_binary = x_test$infectedBinary
x_train_binary = xgb_Probs

# calculate confidence interval
print(ci.auc(x_test_binary, x_train_binary, conf.level = 0.95))
save.image(rdata_file, ascii = TRUE)
