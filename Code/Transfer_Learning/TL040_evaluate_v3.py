"""
Project: Identifying Infected Patients Using Semi-supervised and Transfer Learning
Branch: Transfer Learning
Author: Azi bashiri
Last Modified: May 2021
Description: A python script that evaluates a trained model, in case evaluation was not performed or completed at the
                end of training session.
                Providing the mdl_info of desired models that are stored in log directories, this script loads stored
                models one after the other, test data, and a corresponding scaling object. The scaling object must be
                stored in the same directory as stored model. It then makes predictions for the data and computes
                prediction metrics, predicted probabilities, and 95% CI of AUC. Prediction metrics (e.g, auc) are stored
                in a .txt log file at a log directory that is created when evaluation is started. Predicted
                probabilities (prediction scores) are stored in a .csv log file at the location of the restored model.
                Scores can later be used to compute calibration curves/values.

Notes: Some variables and definitions are hard-coded in the script. When using this python script to evaluate a model
        revisit these definitions and modify them depending on the needs of your project.
        These definitions are:
        - 'DATA_DIR', path to data directory where train and test files are stored
        - 'TEST_FILE', filename of imputed and pre-processed test file to be used for evaluation of a trained model
        - 'mdl_info', a list of tuples. Each item in the list associates with one trained model that will be tested in
                    the chart-reviewed test dataset. Elements of a tuple are in the following order:
                    * the type of the model; options: 'CNN-LSTM', 'TCN', 'LSTM/GRU'
                    * log directory of the saved model
                    * name of the saved model
                    * name of a desired output .csv file to store predicted probabilities

"""

import os
import subprocess
import pandas as pd
from TL099_utils import get_data_v2, log_string, LOG_FOUT
from keras.models import load_model
from tcn import TCN

# Data directory
DATA_DIR = '/Path/to/Data/DIR'
TEST_FILE = os.path.join(DATA_DIR, 'chart_reviewed_test_ImputedData.csv')  # gold-standard test labels

PROJ_DIR = os.path.abspath('')
SCALING = {'scaling': True, 'min_max': True, 'mean_Std': False}

mdl_info = [("CNN-LSTM", "log_YYYYMMDD-hhmm", "best_model.h5", "log_scores.csv"),
            ("TCN", "log_YYYYMMDD-hhmm", "best_model.h5", "log_scores.csv"),
            ("LSTM/GRU", "log_YYYYMMDD-hhmm", "best_model.h5", "log_scores.csv")]

# *****
for i in range(len(mdl_info)):
    # model directory
    model_dir = os.path.join(PROJ_DIR, "Output", mdl_info[i][1])
    SC_PATH = os.path.join(model_dir, 'scaling_obj_tlbase.pkl')
    # load data and model
    if mdl_info[i][0] == "CNN-LSTM":
        # get data
        x, y = get_data_v2([TEST_FILE], loadcreate01=0, target="infection", log_obj=None,
                           scaling=SCALING, sc_path=SC_PATH, expand_dim=True)
        # load the saved model
        model = load_model(os.path.join(model_dir, mdl_info[i][2]))
    elif mdl_info[i][0] == "TCN":
        # get data
        x, y = get_data_v2([TEST_FILE], loadcreate01=0, target="infection", log_obj=None,
                           scaling=SCALING, sc_path=SC_PATH, expand_dim=False)
        # load the saved model
        model = load_model(os.path.join(model_dir, mdl_info[i][2]), custom_objects={'TCN': TCN})
    else:
        # get data
        x, y = get_data_v2([TEST_FILE], loadcreate01=0, target="infection", log_obj=None,
                           scaling=SCALING, sc_path=SC_PATH, expand_dim=False)
        # load the saved model
        model = load_model(os.path.join(model_dir, mdl_info[i][2]))
    log_string(LOG_FOUT, f"Data loaded from: {[TEST_FILE]}")
    log_string(LOG_FOUT, f"Model loaded from: {os.path.join(model_dir, mdl_info[i][2])}")

    # check the performance
    log_string(LOG_FOUT, "\nEvaluation of the trained model:")
    log_string(LOG_FOUT, "\t on data set: " + ' - '.join(
        f"{m} : {v:.4f}" for m, v in zip(model.metrics_names, model.evaluate(x, y, verbose=0))))

    # prediction probability
    y_pred = model.predict(x).ravel()

    # save y_pred and y_true in a CSV file
    log_evaluate = pd.DataFrame({'y_true': y, 'y_pred': y_pred})
    log_evaluate.to_csv(os.path.join(model_dir, mdl_info[i][3]), float_format="%.4f", index=False)

    # ROC CI
    out = subprocess.check_output(
        ["Rscript", os.path.join(PROJ_DIR, 'TL052_run_roc_test.R'),
         "--i", os.path.join(model_dir, mdl_info[i][3])],
        universal_newlines=True)
    log_string(LOG_FOUT, out)


# close the log file
LOG_FOUT.close()
