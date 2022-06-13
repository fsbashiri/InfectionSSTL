# R script that calculates calibration values of ROC curves. It is useful for 
# validating predicted probabilities against binary events. 
# For each predictor, y_true and y_pred are collected from the corresponding 
# log_scores.csv. Calibration measurements of each predictor will be stored in
# a .csv file in the predictors' log directory. A table of all measurements for 
# all predictors will be stored in the output directory. 

# Notes: Some variables and definitions are hard-coded in the script. When using
# this R script please revisit these definitions and modify them accordingly. 
# These definitions are:
#     -Output_dir: address to output directory;
#     -Output_fname: output filename for storing calibration values from all 
#             predictors;
#     -calib_list: A list of information from each predictor in the following 
#             format
#             * log directory of the saved model
#             * name of the .csv file that contains y_true and y_pred values
#             * label for the trained model
#             * output filename to store calibration values for each predictor


# if rms is not included in your default libPaths, uncomment the line below
# .libPaths(c('/Address/to/rmslibrary/', .libPaths()))
library("rms")


# Output address and filename
Output_dir <- '/Code/Transfer_Learning/Output/'
Output_fname <- 'AllModels_CalibValues_YYYY.MM.DD.csv'

# Ver02 of DL predictions - used for AMIA abstract and paper
calib_list = list(
  c("log_YYYYMMDD-hhmm", "log_scores.csv", "CNN-LSTM Sepsis3", "calib_values.csv"),
  c("log_YYYYMMDD-hhmm", "log_scores_fine_tune.csv", "CNN-LSTM FT", "calib_values_FT.csv"),
  c("log_YYYYMMDD-hhmm", "log_scores_fex.csv", "CNN-LSTM FEx", "calib_values_FEx.csv"),
)

# empty dataframe to store all calibration measurements
df_total <- data.frame(matrix(ncol = 5, nrow = 0))  

# loop over items (predictors) in the list
for (item in calib_list){
  # read scores
  data <- read.csv(paste(Output_dir, item[[1]], item[[2]], sep = "/"))
  # get calib values
  res <- val.prob(data$y_pred, as.numeric(data$y_true), pl = TRUE, 
                  statloc = FALSE)
  # collect results
  df <- data.frame("Intercept" = res["Intercept"],
                   "Slope" = res["Slope"],
                   "U" = res["U"],
                   "U:p" = res["U:p"],
                   "Brier" = res["Brier"],
                   row.names = item[[3]])
  # save results for a single predictor in .csv file
  write.csv(df, paste(Output_dir, item[[1]], item[[4]], sep = "/"))
  # append rows to df_total
  df_total <- rbind(df_total, df)
}
# save calib measurements for all models into one .csv file
write.csv(df_total, paste(Output_dir, Output_fname, sep = "/"))



