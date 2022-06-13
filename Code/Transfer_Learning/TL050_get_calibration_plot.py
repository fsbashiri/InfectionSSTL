"""
Project: Identifying Infected Patients Using Semi-supervised and Transfer Learning
Branch: Transfer Learning
Author: Azi bashiri
Last Modified: May 2021
Description: A script to get a calibration curve of one or more saved models. It collects the prediction probabilities
                and the corresponding true labels from log_scores.csv files.

Notes: Some variables and definitions are hard-coded in the script. When using this python script to plot calibration
        curves revisit these definitions and modify them depending on the needs of your project.
        These definitions are:
        - 'OUTPUT_DIR', path to output directory where log folders are stored
        - 'DATA_FILE', a list of tuples. Each item in the list associates with one trained model that will be tested for
                    calibration. Elements of a tuple are in the following order:
                    * log directory of the saved model
                    * name of the .csv file that contains y_true and y_pred values
                    * label for the trained model
        - 'out_fig_name', preferred name for the output figure, which will be stored in the 'OUTPUT_DIR/Figures' folder

"""
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Output directory
OUTPUT_DIR = os.path.join(os.path.abspath(''), "Output")

# address to output files from DL methods
DATA_FILE = [("log_YYYYMMDD-hhmm", "log_scores.csv", "CNN-LSTM Sepsis3"),
             ("log_YYYYMMDD-hhmm", "log_scores_fine_tune.csv", "CNN-LSTM FT"),
             ("log_YYYYMMDD-hhmm", "log_scores_fex.csv", "CNN-LSTM FEx"),
             ]
out_fig_name = "CNN-LSTM_calib_curve_YYYY.MM.DD.eps"


if __name__ == '__main__':
    # start a figure
    plt.plot([0, 1], [0, 1], "k:", label='Perfectly calibrated')
    # loop over all data files in the list
    for i in range(len(DATA_FILE)):
        # get predicted probabilities
        pred_proba = pd.read_csv(os.path.join(OUTPUT_DIR, DATA_FILE[i][0], DATA_FILE[i][1]))

        # Calibration curve
        prob_true, prob_pred = calibration_curve(pred_proba["y_true"], pred_proba["y_pred"], n_bins=10)
        plt.plot(prob_pred, prob_true, 's-', label=f"{DATA_FILE[i][2]}")
        plt.xlabel('Predicted probability')
        plt.ylabel('Actual probability')
        plt.legend(loc="upper left")
    # save figure
    plt.savefig(os.path.join(OUTPUT_DIR, "Figures", out_fig_name), dpi=350)

