"""
Project: Identifying Infected Patients Using Semi-supervised and Transfer Learning
Branch: Transfer Learning -> Feature Extraction
Author: Azi Bashiri
Last Modified: April 2021
Description: Load a pre-trained base model, and perform transfer learning with feature extraction: train a shallow
                classifier in features extracted from chart-reviewed training set using a pre-trained model.

Notes: Some variables and definitions are hard-coded in the script. When using this python script to train a model with
        feature extraction method revisit these definitions and modify them depending on the needs of your project.
        These definitions are:
        - 'MODEL' variable, type of the pre-trained model;
        - 'SC_PATH', address to scaling object fitted to non-chart reviewed dataset. It will be used to transform chart-
                    reviewed training and testing sets;
        - 'model_path', path to log directory in which pre-trained model is stored;
        - 'feature_layer_index', layer number from which features will be extracted. It is different for each model.
                    Usually -4 for a CNN-LSTM model, and -4 otherwise;
        - 'DATA_DIR', path to data directory where train and test files are stored
        - 'TRAIN_FILE' and 'TEST_FILE', filename of imputed and pre-processed train and test files
        - 'expand_dim' arg of 'get_data_v2', bool, TRUE if MODEL is mdl_cnnlstm_v40.py
        - 'target' arg of 'get_data_v2', string, label for the target data in the dataset

"""
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import os
import sys
import timeit
import subprocess
import numpy as np
import pandas as pd
from TL099_utils import log_string, get_data_v2, PROJ_DIR, LOG_DIR, LOG_FOUT, METRICS
from tcn import TCN

from keras.models import load_model, Model

from scipy.stats import uniform, loguniform
from pickle import dump

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, auc, roc_curve


# define constant parameters
MODEL = 'mdl_cnnlstm_v40.py'  # options: mdl_tcn.py, mdl_cnnlstm_v40.py, mdl_lstm_gru.py
SC_PATH = os.path.join(PROJ_DIR, 'Output', 'log_YYYYMMDD-hhmm', 'scaling_obj_tlbase.pkl')  # path to scaler obj
model_path = os.path.join(PROJ_DIR, 'Output', 'log_YYYYMMDD-hhmm', 'best_model.h5')  # path to pre-trained model
feature_layer_index = -4  # index of a layer that features will be extracted from; usually -4 for CNNLSTM, -2 otherwise

GPU_INDEX = 0  # GPU index to use. To run on CPU, input a negative value
N_SPLITS = 5  # cross-validation splits
SCALING = {'scaling': True, 'min_max': True, 'mean_Std': False}


# Processing unit: GPU or CPU. Negative value will run the code on CPU
if GPU_INDEX < 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_INDEX}"

# Do not squeeze when printing
np.set_printoptions(threshold=sys.maxsize)  # print numpy arrays completely
pd.set_option("display.max_rows", None, "display.max_columns", None)  # print pandas dataframes completely

# Data directory
DATA_DIR = '/Path/to/Data/DIR'
TRAIN_FILE = os.path.join(DATA_DIR, 'chart_reviewed-train_ImputedData.csv')  # gold-standard train labels
TEST_FILE = os.path.join(DATA_DIR, 'chart_reviewed_test_ImputedData.csv')  # gold-standard test labels

# ***** Start logging *****
# copy this file for backup. LOG_DIR, PROJ_DIR, LOG_FOUT are imported from utils
os.system(f"cp TL031_feature_extraction.py {LOG_DIR}")  # bkp of train procedure
os.system(f"cp {os.path.join(PROJ_DIR, 'Models', MODEL)} {LOG_DIR}")  # bkp of model
log_string(LOG_FOUT, 'Python %s on %s' % (sys.version, sys.platform))
log_string(LOG_FOUT, f'\nLOG_DIR: {LOG_DIR}')

if __name__ == '__main__':
    # ***** get the data
    x_train, y_train = get_data_v2([TRAIN_FILE], loadcreate01=0, target="infection", log_obj=LOG_FOUT,
                                   scaling=SCALING, sc_path=SC_PATH, expand_dim=True)
    x_test, y_test = get_data_v2([TEST_FILE], loadcreate01=0, target="infection", log_obj=LOG_FOUT,
                                 scaling=SCALING, sc_path=SC_PATH, expand_dim=True)

    # load a model
    log_string(LOG_FOUT, '\nLoading the trained model:')
    log_string(LOG_FOUT, '===========================\n')
    log_string(LOG_FOUT, f"Model path: {model_path}")
    if MODEL == 'mdl_tcn.py':
        base_model = load_model(model_path, custom_objects={'TCN': TCN})
    else:
        base_model = load_model(model_path)
    base_model.summary(print_fn=lambda x: log_string(LOG_FOUT, x))
    log_string(LOG_FOUT, "Performance of the base model:")
    log_string(LOG_FOUT, "on training set: " + ' - '.join(
        f"{m} : {v:.4f}" for m, v in zip(base_model.metrics_names, base_model.evaluate(x_train, y_train, verbose=0))))
    log_string(LOG_FOUT, "on testing set: " + ' - '.join(
        f"{m} : {v:.4f}" for m, v in zip(base_model.metrics_names, base_model.evaluate(x_test, y_test, verbose=0))))

    # ***** Extracting features
    log_string(LOG_FOUT, '\nExtracting features')
    log_string(LOG_FOUT, '===========================\n')
    # remove the output layer
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[feature_layer_index].output)
    # get extracted features
    features_train = model.predict(x_train)
    features_test = model.predict(x_test)
    features_train = features_train.reshape((features_train.shape[0], -1))
    features_test = features_test.reshape((features_test.shape[0], -1))
    log_string(LOG_FOUT, f"Train set feature shape: {features_train.shape}")
    log_string(LOG_FOUT, f"Test set feature shape: {features_test.shape}")

    # ***** Train a shallow model
    log_string(LOG_FOUT, '\nTraining a shallow classifier')
    log_string(LOG_FOUT, '===========================\n')
    start_time = timeit.default_timer()
    svc = SVC(probability=True)
    # It is recommended using np.logspace(-3, 2, 6) for both C and gamma
    distributions = dict(C=loguniform(1e-6, 1e2),
                         kernel=['linear', 'rbf', 'poly'],
                         degree=range(1, 4),
                         gamma=loguniform(1e-6, 1e2))
    scoring = {'Accuracy': 'accuracy', 'auroc': 'roc_auc'}
    cv = N_SPLITS
    clf = RandomizedSearchCV(estimator=svc,
                             param_distributions=distributions,
                             scoring=scoring,
                             random_state=2996,
                             refit='auroc',
                             cv=cv,
                             return_train_score=True,
                             n_iter=50,
                             verbose=3,
                             n_jobs=-1)
    log_string(LOG_FOUT, f"Classifier: {clf}")
    clf.fit(features_train, y_train)
    # elapsed time
    elapsed = timeit.default_timer() - start_time
    log_string(LOG_FOUT, f'Execution time of cross-validated search: {elapsed / 60} min')
    # pickle dump the clf
    with open(os.path.join(LOG_DIR, "clf.pkl"), 'wb') as output:
        dump(clf, output)
        log_string(LOG_FOUT, f"Cross validated model (clf.pkl) saved in the LOG_DIR.")
    # get cross validated results
    df_cv_result = pd.DataFrame(clf.cv_results_)
    # log some results
    log_string(LOG_FOUT, f"Best params: {clf.best_params_}")
    log_string(LOG_FOUT, f"Best test score: {clf.best_score_}")
    # save df_cv_result in a .csv file - it's a long table
    df_cv_result.to_csv(os.path.join(LOG_DIR, 'clf_cv_result.csv'), float_format="%.4f", index=False)

    # retrain
    log_string(LOG_FOUT, '\nRe-Train w/ best_params')
    log_string(LOG_FOUT, '===========================\n')
    svc.set_params(**clf.best_params_)
    svc.fit(X=features_train, y=y_train)
    y_pred = svc.predict_proba(features_test)
    log_string(LOG_FOUT, f"Test auroc: {roc_auc_score(y_test, y_pred[:,1])}")

    # store predicted probabilities
    log_evaluate = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred[:, 1]})
    log_evaluate.to_csv(os.path.join(LOG_DIR, 'log_scores_fex.csv'), float_format="%.4f", index=False)

    # ROC CI
    out = subprocess.check_output(
        ["Rscript", os.path.join(PROJ_DIR, 'TL052_run_roc_test.R'),
         "--i", os.path.join(LOG_DIR, 'log_scores_fex.csv')],
        universal_newlines=True)
    log_string(LOG_FOUT, out)

    # close the log file
    LOG_FOUT.close()
