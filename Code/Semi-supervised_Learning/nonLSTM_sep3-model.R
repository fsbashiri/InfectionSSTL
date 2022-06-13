# #### 

library(MASS)
library(plyr)
library(dplyr)
library(pROC)
library(caret)
library(ROCR)



# MODIFY ME

dataTrainFile = '/path/to/file'
dataTestFile = '/path/to/file'
# dataTrainUnlFile = '/LAB_SHARED/PROJECTS/001-Semisupervised_Learning/Sandbox/Data_sandbox/chart_review_eligible_dataset.csv'

# if not using a GPU, then modify code below at "non-GPU usage"

# end MODIFY ME



# Setup
set.seed(42)
`%notin%` <- Negate(`%in%`)
tstampString = format(Sys.time(), "%H%M_%m%d")

# import data code block
# Do NOT run this code block if you are running Supplemental Data Code block for testing code
# x_train = read.csv(dataTrainFile)
# x_test = read.csv(dataTestFile)
# end import data code block

# Other setup
# Data must be formatted to have one predictor column named 'infectedV',
# and all other columns are feature variables. 

# In addition, for this analysis, the infection status 
# is determined by the sepsis3 status. A similar approach to the example 
# below can be used to update infection labels using another column

# A commented-out example here converts a binary encoding for the predictor variable 
# in a column named "infection" to the format that caret requires:
# df_training$infectedV = ifelse(df_training$infection == 1, "infectedYes", "infectedNo")
# x_train = df_training[which(names(df_training %notin% c('infection')))]

# end Setup


# create model
cvCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = TRUE)

# build xgboost grid
xgbGrid<-expand.grid(nrounds=c(500, 1000, 1500, 2000), max_depth=c(2, 5, 10, 15), eta=c(0.001, 0.01, 0.1), gamma = c(0), colsample_bytree = c(1), min_child_weight = c(2, 5), subsample = c(0.5))

# build XGBoost model
xgbGridmod<- train(infectedV~., method = "xgbTree", data = x_train, metric = "ROC", tree_method = "gpu_hist", gpu_id = 0, trControl = cvCtrl, tuneGrid = xgbGrid, na.action = na.pass)

# non-GPU usage:
# xgbGridmod<- train(infectedV~., method = "xgbTree", data = x_train, metric = "ROC", trControl = cvCtrl, tuneGrid = xgbGrid, na.action = na.pass)

# test XGBoost model
xgbGridProbs<-predict(xgbGridmod, newdata=x_test, type="prob", na.action = na.pass)
xgbGridProbsDF<-as.data.frame(xgbGridProbs)

# get AUC and generate plot
xgbGridRoc_sep3<-roc(x_test$infectedV, round(xgbGridProbsDF$infectedYes, digits=4))

print('sepsis3 model AUC is')
print(xgbGridRoc_sep3)
jpg_name = paste0('nonLSTM_aucplot_sep3-model', tstampString, '.jpg')
jpeg(jpg_name)
plot(xgbGridRoc_sep3)
dev.off()

# optional, can comment out
print('checking variable importance')
xgbImp = varImp(xgbGridmod, scale=FALSE)
print(xgbImp)

# get Confidence Interval
x_test$infectedBinary = ifelse(x_test$infectedV == c('infectedYes'), 1, 0)
x_test_binary = x_test$infectedBinary
x_train_binary = xgbGridProbs[,2]
print(ci.auc(x_test_binary, x_train_binary, conf.level = 0.95))

# save workspace image
save.image('nonLSTM_24h_sep3-model.Rdata', ascii = TRUE)
