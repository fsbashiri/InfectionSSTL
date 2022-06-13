# R script that calculates AUC and 95% CI (with DeLong method) of predictions 
# made by one or more predictors. There must be a single y_true associated to 
# all predictions. The y_true must be in the first column of input data. The R 
# script can compare their predictions and calculate p-value. The input to the 
# script is a path to a .csv file containing at least two columns: y_true in the
# first column, and predicted probabilities of other predictors in the second 
# column and so forth. The R script can be called from the command line.
# Arguments:
#     -i: address to log_scores.csv file
#     -p: label of the predictor to which other predictions will be compared. 
#         Leave it empty if you don't want get p-values.


# if pROC is not included in your default libPaths, uncomment the line below
# .libPaths(c('/Address/to/pROClibrary/', .libPaths()))
library("pROC")
library("optparse")

# specify desired options in a list
option_list <- list(
  make_option(c("-i", "--input"), type="character", 
              default="/Code/Transfer_Learning/Output/log_YYYMMDD-hhmm/log_scores.csv",
              help="Path to .csv file that contains true lables and predicted scores [default %default]"),
  make_option(c("-p", "--pred"), type="character",
              default=" ", 
              help="The predictor object or column to be tested paired with other predictions. If None, no p-value is returned [default %default]")
)


# get command line options, if help option encountered print help and exit, 
# otherwise if options not found on command line then set defaults,
opt <- parse_args(OptionParser(option_list=option_list))

# read data from .csv file
data <- read.csv(opt$input)
cnames <- colnames(data)[-1]  # first column is y_true, the rest are different predictors

# compute AUC and 95% CI (DeLong)
for (c in cnames)
{
  cat("predictor: ", c, "\n")  # print out predictor name
  print(auc(data$y_true, data[[c]]))  # print out AUC
  print(ci.auc(data$y_true, data[[c]], method=c("delong")))  # print out 95% CI
}

# get the p-value by comparing opt$pred prediction with other predictions
if (opt$pred != " ")
{
  print("\n\nComparisons for p-value:")
  for (c in cnames[!cnames %in% opt$pred])
  {
    cat("====\n* predictors: ", opt$pred, "vs.", c, "\n")
    # create roc curves
    roc1 <- roc(data$y_true, data[[opt$pred]])
    roc2 <- roc(data$y_true, data[[c]])
    # apply roc.test
    print(roc.test(roc1, roc2, method=c("delong")))
  }
}


