# R script that performs DeLong's test for two correlated ROC curves and 
# computes their p-value for differences in the AUC. As a complementary output,
# it also computes 95% confidence interval of AUC(s).
# This script reads in a column of response and all predictions provided in a 
# .csv file. The header label for the response data must be "y_true". The header
# label for predictions can be anything meaningful to the user. 
#
# Notes: Some variables and definitions are hard-coded in the script. When using
# this R script please revisit these definitions and modify them accordingly. 
# These definitions are:
#     -address to .csv file that contains y_true and prediction probabilities 
#     -index of the column that stores y_true (default: 1)
#     -index of the column that predictions will be compared with (default: 10)


# if pROC is not included in your default libPaths, uncomment the line below
# .libPaths(c('/Address/to/pROClibrary/', .libPaths()))
library("pROC")

# read data from 
data <- read.csv("/Code/Transfer_Learning/Output/All_PredScores.csv")
cnames <- colnames(data)[-1]  # first column is y_true, the rest are different predictors
target_predictor <- cnames[10]  # compare all predictions with target_predictor

# Loop over all predictors
for (c in cnames)
{
  cat("====\n predictors: ", target_predictor, "vs.", c, "\n")
  # create roc curves
  roc1 <- roc(data$y_true, data[[target_predictor]])
  roc2 <- roc(data$y_true, data[[c]])
  # print out 95% CI
  cat(c, ": ")
  print(ci.auc(data$y_true, data[[c]], method=c("delong")))
  # apply roc.test
  print(roc.test(roc1, roc2, method=c("delong")))
  # print out p-value in a separate line
  testOut <- roc.test(roc1, roc2, method = c("delong"))
  cat("p-value: ", testOut$p.value, "\n")
}
  
