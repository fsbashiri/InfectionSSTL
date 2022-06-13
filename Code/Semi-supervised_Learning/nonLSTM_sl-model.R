# #### 

library(MASS)
library(plyr)
library(dplyr)
library(pROC)
library(caret)
library(ROCR)


# MODIFY ME

dataTrainFile = '/path/to/file'
dataTrainUnlFile = '/path/to/file'
dataTestFile = '/path/to/file'

# if not using a GPU, then modify code below at "non-GPU usage"

# end MODIFY ME


# Setup
set.seed(42)
`%notin%` <- Negate(`%in%`)
tstampString = format(Sys.time(), "%H%M_%m%d")

# Set confThreshold to be a confidence value that predictions must exceed 
# to be relabeled.
confThreshold = 0.98999999

# Other setup
# Data must be formatted to have one predictor column named 'infectedV',
# and all other columns are feature variables. 

# A commented-out example here converts a binary encoding for the predictor variable 
# in a column named "infection" to the format that caret requires:
# df_training$infectedV = ifelse(df_training$infection == 1, "infectedYes", "infectedNo")
# x_train = ds_training[which(names(ds_training %notin% c('infection')))]

# In addition, the unlabeled dataset needs to have the same infectedV column
# with all NA values

# IMPORTANT: this script assumes you are using a GPU to speed up XGBoost. 
# If your are not, then you need to modify the train lines of code below, 
# replacing TRAINING_DATA appropriately:
# xgbGridmod<- train(infectedV~., method = "xgbTree", data = TRAINING_DATA, metric = "ROC", trControl = cvCtrl, tuneGrid = xgbGrid, na.action = na.pass)

# end setup


# import data code block
# Do NOT run this code block if you are running Supplemental Code Block for testing code
x_train = read.csv(dataTrainFile)
x_test = read.csv(dataTestFile)
x_train_unlabeled = read.csv(dataTrainUnlFile)
x_train_unlabeled$infectedV = NA
# end import data code block

# Create train-test split for self-learning. Depenting on your dataset, you may wish 
# to do a stratified split instead of the simple split listed below
# Do NOT run the next 4 lines of code if running Supplemental Code Block for testing code
x_train_split = floor(0.8 * nrow(x_train))
x_train_set = sample(seq_len(nrow(x_train)), size = x_train_split)
x_train_train = x_train[ x_train_set,]
x_train_test = x_train[ -x_train_set,]

# create model to start self-learning
cvCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = TRUE)

# xgboost grid
xgbGrid<-expand.grid(nrounds=c(500, 1000, 1500, 2000), max_depth=c(2, 5, 10, 15), eta=c(0.001, 0.01, 0.1), gamma = c(0), colsample_bytree = c(1), min_child_weight = c(2, 5), subsample = c(0.5))
#xgboost model
xgbGridmod<- train(infectedV~., method = "xgbTree", data = x_train_train, metric = "ROC", tree_method = "gpu_hist", gpu_id = 0, trControl = cvCtrl, tuneGrid = xgbGrid, na.action = na.pass)
xgbGridProbs<-predict(xgbGridmod, newdata=x_train_test, type="prob", na.action = na.pass)
xgbGridProbsDF<-as.data.frame(xgbGridProbs)

xgbGridRoc_baseline<-roc(x_train_test$infectedV, round(xgbGridProbsDF$infectedYes, digits=4))

print('starting AUC is')
print(xgbGridRoc_baseline)

debugOutput1 = paste0('training model has this number of predictions above ', confThreshold, '% confidence:')
print(debugOutput1)
confPredsY = xgbGridProbsDF[which(xgbGridProbsDF$infectedYes >= confThreshold),]
confPredsN = xgbGridProbsDF[which(xgbGridProbsDF$infectedNo >= confThreshold),]
confPreds = rbind(confPredsY, confPredsN)

encCt = nrow(x_train_train)
startStr = paste0('starting self-training with ', encCt, ' encounters')
print(startStr)

# next, predict unlabeled data using threshold probability, probsTH
probsTH = confThreshold
loopCt = 0
plateau_ct = 0
auc_prev = xgbGridRoc_baseline$auc
auc_max = xgbGridRoc_baseline$auc
xgbGridRoc_aucValue = 0

# vectors for progress dataframe
loopCtVec = c(NA)
updatedLabelCt = c(NA)
trainingCt = c(encCt)
unlabeledCt = c(nrow(x_train_unlabeled))
loopAUC = c(auc_prev)
ts_string = paste0(format(Sys.time(), "%H%M%S_%Y%m%d"))
maxAUC_dataset = x_train_train

# the while loop will run until 
# at least one of the following criteria is met:
# # - completing over 1000 iterations total
# # - the AUC does not increase for 3 consecutive iterations
# # - no updated labels were assigned

while (plateau_ct < 4) {
  loopCt = loopCt + 1
  
  set.seed(42)
  if (loopCt > 1000) {
    # if there are over 1k training loops in a non-deep learning analysis
    # then something is wrong 
    print('run tripped breaker of 1000 loops')
    break 
  }
  xgbLogistic_probs_unlabeledAdded = predict(xgbGridmod, newdata=x_train_unlabeled, type="prob", na.action = na.pass)
  xgbLogistic_probsDF_unlabeledAdded = as.data.frame(xgbLogistic_probs_unlabeledAdded)

  print('nrow in unlabeledAdded is')
  print(nrow(xgbLogistic_probsDF_unlabeledAdded))

  # cbind predictions to unlabeled dataset
  x_train_unlabeled = cbind(x_train_unlabeled, xgbLogistic_probsDF_unlabeledAdded)
  
  # relabel if probability is above threshold
  x_train_unlabeled$infectedV = ifelse(x_train_unlabeled$infectedYes >= probsTH,c("infectedYes"),NA)
  x_train_unlabeled$infectedV = ifelse(x_train_unlabeled$infectedNo >= probsTH,c("infectedNo"),x_train_unlabeled$infectedV)
  # finally, remove updated labels from unlabeled, and then add updated labels to training
  xgbLogistic_probs_addedLabels = x_train_unlabeled[which((x_train_unlabeled$infectedV == c("infectedYes")) | x_train_unlabeled$infectedV == c("infectedNo")),]
  x_train_unlabeled = x_train_unlabeled[ -which((x_train_unlabeled$infectedV == c("infectedYes")) | x_train_unlabeled$infectedV == c("infectedNo")),]
  
  # write results of loop to disk, this can be skipped
  step_filename = paste0('relabeled_xgb24h_99_', loopCt,'.csv')
  write.csv(xgbLogistic_probs_addedLabels, step_filename)

  xgbLogistic_probs_addedLabels = select(xgbLogistic_probs_addedLabels, -c("infectedYes", "infectedNo"))
  x_train_unlabeled = select(x_train_unlabeled, -c("infectedYes", "infectedNo"))  
  
  encAdded = (nrow(xgbLogistic_probs_addedLabels) + nrow(x_train_train)) - encCt
  encAddedStr = paste0('updated labels for ', encAdded, ' encounters')
  print(encAddedStr)
  encCt = encAdded + encCt
  
  if (encAdded == 0) {
    # if no new encounter labels were updated, then break from loop
    print('no new updates to labels, stopping self-training')
    break
  }
  
  x_train_train = rbind(x_train_train, xgbLogistic_probs_addedLabels)
  
  # build and train new XGBoost model using relabeled datapoints
  xgbGridmod <- train(infectedV~., method = "xgbTree", data = x_train_train, metric = "ROC", tree_method = "gpu_hist", gpu_id = 0, trControl = cvCtrl, tuneGrid = xgbGrid, na.action = na.pass)
  
  # then predict an updated AUC
  xgbLogistic_probs = predict(xgbGridmod, newdata=x_train_test, type="prob", na.action = na.pass)
  
  xgbLogistic_probsDF = as.data.frame(xgbLogistic_probs)
  xgbGridRoc<-roc(x_train_test$infectedV, round(xgbLogistic_probsDF$infectedYes, digits=4))
  xgbGridRoc_aucValue = xgbGridRoc$auc

  # we are only interested in the highest AUC
  if (xgbGridRoc_aucValue > auc_max) {
    auc_max = xgbGridRoc_aucValue
    maxAUC_dataset = x_train_train
  }
  
  # finally, we need to see how the AUC is doing
  # and save the progress in a vector
  auc_diff = xgbGridRoc_aucValue - auc_prev
  auc_prev = xgbGridRoc_aucValue
  loopCtVec = c(loopCtVec, loopCt)
  updatedLabelCt = c(updatedLabelCt, encAdded)
  trainingCt = c(trainingCt, encCt)
  unlabeledCt = c(unlabeledCt, nrow(x_train_unlabeled))
  loopAUC = c(loopAUC, xgbGridRoc_aucValue)
  
  # if the AUC is not increasing, then it is time to stop self-learning
  if (auc_diff < 0.02) {
     plateau_ct = plateau_ct + 1
  } else {
    plateau_ct = 0
  }

}

# After exiting self-learning, write the self-learning stats to disk
df_log = data.frame(loopCtVec, updatedLabelCt, trainingCt, unlabeledCt, loopAUC)
ts_string = paste0('xgb24h_99_', format(Sys.time(), "%H%M%S_%Y%m%d"))
df_log_outFileName = paste0('self_learning_stats.', ts_string, '.csv')
write.csv(df_log, df_log_outFileName)

print('finished training, max AUC value was')
print(auc_max)

# then, create a new training dataset that includes datapoints with updated labels
fModelData = rbind(maxAUC_dataset, x_train_test)

modelFit <- train(infectedV~., method = "xgbTree", data = fModelData, metric = "ROC", tree_method = "gpu_hist", gpu_id = 0, trControl = cvCtrl, tuneGrid = xgbGrid, na.action = na.pass)
print('testing model against test dataset')
xgb_probs_testing = predict(enFit, newdata=x_test, type="prob", na.action = na.pass)
xgb_probsDF_testing = as.data.frame(xgb_probs_testing)
xgbGridRoc<-roc(x_test$infectedV, round(xgb_probsDF_testing$infectedYes, digits=4))
print('AUC is')
print(xgbGridRoc)

# optional, write training dataset to disk
# write.csv(fModelData, 'x_train.dataset.sl99.xgb.csv')

# now calculate a CI
xgb_Probs = xgb_probs_testing[,2]
x_test$infectedBinary = ifelse(x_test$infectedV == c('infectedYes'), 1, 0)
x_test_binary = x_test$infectedBinary
x_train_binary = xgb_Probs
print(ci.auc(x_test_binary, x_train_binary, conf.level = 0.95))

# finally, generate a plot, then save the workflow as an Rdata file

jpeg('aucplot_selflearning_xgb_24h.99.jpg')
plot(xgbGridRoc)
dev.off()

save.image('nonLSTM_24h_selfLearning_xgb.99.Rdata', ascii = TRUE)
