"""
Project: Identifying Infected Patients Using Semi-supervised and Transfer Learning
Branch: Transfer Learning
Author: Azi Bashiri
Last Modified: October 2021
Description: Training a base model on the non-chart reviewed (large) dataset using silver standard Sepsis-3 labels.
                The output model will be used as the source model for transfer learning. At the begining of training a
                log directory will be created in the Output folder with 'log_YYYYMMDD-hhmm' format. The model checkpoint
                along with other log files including log_scores.csv will be stored in the log directory.

Notes: Some variables and definitions are hard-coded in the script. When using this python script to train a model,
        revisit these definitions and modify them depending on the needs of your project. These definitions are:
        - 'MODEL' variable, the model that will be trained and where 'create_model' function will be imported from;
        - 'TUNER_NAME', the hyperparameter optimization algorithm;
        - 'DATA_DIR', path to data directory where train and test files are stored;
        - 'TRAIN_FILE' and 'TEST_FILE', filename of imputed and pre-processed train and test files;
        - 'expand_dim' arg of 'get_data_v2', bool, TRUE if MODEL is mdl_cnnlstm_v40.py
        - 'target' arg of 'get_data_v2', string, label for the target data in the dataset
"""

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import os
import sys
import timeit
import subprocess  # to run an Rscript that gives 95% CI with DeLong method
import numpy as np
import pandas as pd
from TL099_utils import log_string, get_data_v2, PROJ_DIR, LOG_DIR, LOG_FOUT

import tensorflow as tf
from tcn import TCN

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from kerastuner import Objective
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
from kerastuner.engine import tuner

MODEL = 'mdl_cnnlstm_v40.py'  # options: 'mdl_cnnlstm_v40.py', 'mdl_tcn.py', 'mdl_lstm_gru'
if MODEL == 'mdl_cnnlstm_v40.py':
    from Models.mdl_cnnlstm_v40 import create_model, log_string_search_space_summary
elif MODEL == 'mdl_lstm_gru.py':
    from Models.mdl_lstm_gru import create_model, log_string_search_space_summary
elif MODEL == 'mdl_tcn.py':
    from Models.mdl_tcn import create_model, log_string_search_space_summary


# define constant parameters
GPU_INDEX = 0  # GPU index to use. To run on CPU, input a negative value
MAX_EPOCH = 20  # Epoch to run
MAX_TRIALS = 50
SCALING = {'scaling': True, 'min_max': True, 'mean_Std': False}
TUNER_SEED = 2996
TUNER_NAME = 'BayesianOptimization'  # options: RandomSearch, BayesianOptimization, Hyperband

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
TRAIN_FILE = os.path.join(DATA_DIR, 'non_chart_reviewed_ImputedData.csv')  # silver-standard train labels
TEST_FILE = os.path.join(DATA_DIR, 'chart_reviewed_test_ImputedData.csv')  # gold-standard test labels

# ***** Start logging *****
# copy files for backup. LOG_DIR, PROJ_DIR, LOG_FOUT are imported from utils
os.system(f"cp TL020_train_base.py {LOG_DIR}")  # bkp of train procedure
os.system(f"cp {os.path.join(PROJ_DIR, 'Models', MODEL)} {LOG_DIR}")  # bkp of model
log_string(LOG_FOUT, 'Python %s on %s' % (sys.version, sys.platform))
log_string(LOG_FOUT, f'\nLOG_DIR: {LOG_DIR}')

# Scaler object path
SC_PATH = os.path.join(LOG_DIR, 'scaling_obj_tlbase.pkl')  # save scaling object for future reference


# to add a hyperparamter outside of the create_model function, we need to define a sub-class
if TUNER_NAME == 'RandomSearch':
    class MyTuner(RandomSearch):  # options: RandomSearch, BayesianOptimization, Hyperband
        def run_trial(self, trial, *args, **kwargs):
            kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', [32, 64, 128, 256, 512], default=256)
            super(MyTuner, self).run_trial(trial, *args, **kwargs)
elif TUNER_NAME == 'BayesianOptimization':
    class MyTuner(BayesianOptimization):  # options: RandomSearch, BayesianOptimization, Hyperband
        def run_trial(self, trial, *args, **kwargs):
            kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', [32, 64, 128, 256, 512], default=256)
            super(MyTuner, self).run_trial(trial, *args, **kwargs)
else:
    class MyTuner(Hyperband):  # options: RandomSearch, BayesianOptimization, Hyperband
        def run_trial(self, trial, *args, **kwargs):
            kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', [32, 64, 128, 256, 512], default=256)
            super(MyTuner, self).run_trial(trial, *args, **kwargs)


log_metrics = {}  # to collect the best metrics (wrt es1 objective) for each run
log_trialids = {}  # only needed when tuning with Hyperband
search_num = [0]


class CustomLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # get the most related Trial_id
        t_id = tuner.oracle.ongoing_trials[tuner.tuner_id].trial_id  # current trial id
        t_state = tuner.oracle.trials.get(t_id).get_state()  # current tuner state
        if 'tuner/trial_id' in t_state['hyperparameters']['values'].keys():
            # this happens only to Hyperband when the tuner picks up one of the previous trials
            old_t_id = t_state['hyperparameters']['values']['tuner/trial_id']
            # this happens when a trial_id is connected to a second or higher degree trial_id
            t_id_list = [log_trialids[key] for key in log_trialids]
            for i, sublist in enumerate(t_id_list):
                if old_t_id in sublist:
                    if t_id not in sublist:
                        # append new t_id if it's not already in the sublist
                        log_trialids[[*log_trialids][i]].append(t_id)
                    t_id = [*log_trialids][i]  # update t_id to the first degree t_id
                    break
        # log logs
        log_string(LOG_FOUT, f"**** Epoch {epoch+1:03}: " +
                   ' - '.join(f"{m}: {v:.4f}" for m, v in zip(logs.keys(), logs.values())))
        if epoch == 0:
            # add a new key to the dict when a new training starts
            log_metrics[t_id] = [epoch+1, *logs.values()]
            log_trialids[t_id] = [t_id]
        else:
            # replace metrics if it's improved
            if log_metrics[t_id][-2] < logs['val_auroc']:
                log_metrics[t_id] = [epoch+1, *logs.values()]

    def on_train_end(self, logs=None):
        search_num[0] += 1
        log_string(LOG_FOUT, f"Search {search_num[0]}/{MAX_TRIALS} Completed.\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ***** get the data
    x_train, y_train, _ = get_data_v2([TRAIN_FILE], loadcreate01=1, target="sepsis3_bool", log_obj=LOG_FOUT,
                                      scaling=SCALING, sc_path=SC_PATH, expand_dim=True)
    x_test, y_test, _ = get_data_v2([TEST_FILE], loadcreate01=0, target="infection", log_obj=None,
                                    scaling=SCALING, sc_path=SC_PATH, expand_dim=True)

    # ***** Search space
    log_string(LOG_FOUT, '\nSetting up a search space')
    log_string(LOG_FOUT, '===========================\n')
    hp = HyperParameters()
    if TUNER_NAME == 'RandomSearch':
        tuner = MyTuner(create_model, hyperparameters=hp, tune_new_entries=True,
                        objective=Objective("val_auroc", direction="max"),
                        # num_initial_points=5,  # only for Bayesian Optimization
                        max_trials=MAX_TRIALS,  # comment for Hyperband
                        # max_epochs=MAX_TRIALS,   # uncomment for Hyperband
                        executions_per_trial=1, seed=TUNER_SEED,
                        directory=LOG_DIR, project_name=TUNER_NAME)  # edit according to the tuner
    elif TUNER_NAME == 'BayesianOptimization':
        tuner = MyTuner(create_model, hyperparameters=hp, tune_new_entries=True,
                        objective=Objective("val_auroc", direction="max"),
                        num_initial_points=5,  # for Bayesian Optimization
                        max_trials=MAX_TRIALS,  # comment for Hyperband
                        # max_epochs=MAX_TRIALS,   # uncomment for Hyperband
                        executions_per_trial=1, seed=TUNER_SEED,
                        directory=LOG_DIR, project_name=TUNER_NAME)  # edit according to the tuner
    else:
        tuner = MyTuner(create_model, hyperparameters=hp, tune_new_entries=True,
                        objective=Objective("val_auroc", direction="max"),
                        # num_initial_points=5,  # for Bayesian Optimization
                        # max_trials=MAX_TRIALS,  # comment for Hyperband
                        max_epochs=MAX_TRIALS,   # uncomment for Hyperband
                        executions_per_trial=1, seed=TUNER_SEED,
                        directory=LOG_DIR, project_name=TUNER_NAME)  # edit according to the tuner
    log_string_search_space_summary(tuner)

    # ***** finding optimal hyperparameters
    log_string(LOG_FOUT, '\nTuning hyper-parameters')
    log_string(LOG_FOUT, '===========================\n')
    start_time = timeit.default_timer()
    # setting callback functions
    es1 = EarlyStopping(monitor='val_auroc', mode='max', patience=5, min_delta=0.005, verbose=1)
    # fit the model
    tuner.search(x=x_train, y=y_train,
                 epochs=MAX_EPOCH,  # batch_size is a hyperparameter
                 validation_split=0.2, verbose=0,
                 callbacks=[es1, CustomLoggingCallback()])
    # elapsed time
    elapsed = timeit.default_timer() - start_time
    log_string(LOG_FOUT, f'Execution time of {TUNER_NAME} search: {elapsed / 60} min')

    # ***** log the search summary
    log_string(LOG_FOUT, '\nResults summary:')
    log_string(LOG_FOUT, '===========================\n')
    log_metrics = pd.DataFrame.from_dict(log_metrics, orient='index',
                                         columns=['Epoch', 'loss', 'accuracy', 'auroc', 'auprc', 'val_loss',
                                                  'val_accuracy', 'val_auroc', 'val_auprc'])
    log_metrics = log_metrics.reset_index().rename(columns={'index': 'Trial_id'})
    log_string(LOG_FOUT, f"Best metrics wrt val_auroc at each run: \n{log_metrics}\n")
    # store log_metrics in .csv file for future visualization
    log_metrics.to_csv(os.path.join(LOG_DIR, 'log_metrics.csv'), float_format="%.4f", index=False)

    # ***** re-instantiate and retrain on the full dataset
    # For best performance, It is recommended to retrain your Model on the full dataset using the best hyperparameters
    # found during `search`
    log_string(LOG_FOUT, '\nRe-instantiate and Train')
    log_string(LOG_FOUT, '===========================\n')
    best_hp = tuner.get_best_hyperparameters(1)[0]  # Returns the best hyperparameters, as determined by the objective
    model = tuner.hypermodel.build(best_hp)  # reinstantiate the (untrained) best model found during the search process
    # train on the full dataset
    mc = ModelCheckpoint(os.path.join(LOG_DIR, 'best_model.h5'), monitor='auroc', mode='max', verbose=1,
                         save_best_only=True)
    model.fit(x=x_train, y=y_train, epochs=MAX_EPOCH, batch_size=best_hp.values['batch_size'], verbose=2,
              callbacks=[mc])
    df_history = pd.DataFrame(model.history.history)
    log_string(LOG_FOUT, f"\nTraining history:\n{df_history}")

    # re-load the saved model and check the performance
    if MODEL == 'mdl_tcn.py':
        model = load_model(os.path.join(LOG_DIR, 'best_model.h5'), custom_objects={'TCN': TCN})
    else:
        model = load_model(os.path.join(LOG_DIR, 'best_model.h5'))
    log_string(LOG_FOUT, "\nEvaluation of the trained model:")
    log_string(LOG_FOUT, "\t on training set: " + ' - '.join(
        f"{m} : {v:.4f}" for m, v in zip(model.metrics_names, model.evaluate(x_train, y_train, verbose=0))))
    # check the performance in the gold standard test labels (further evaluation with 040_evaluate_v3.py is not needed)
    log_string(LOG_FOUT, "\t on gold standard test set: " + ' - '.join(
        f"{m} : {v:.4f}" for m, v in zip(model.metrics_names, model.evaluate(x_test, y_test, verbose=0))))

    # store predicted probabilities of gold standard test set in log_scores.csv
    y_pred = model.predict(x_test).ravel()
    log_evaluate = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    log_evaluate.to_csv(os.path.join(LOG_DIR, 'log_scores.csv'), float_format="%.4f", index=False)

    # ROC CI
    out = subprocess.check_output(
        ["Rscript", os.path.join(PROJ_DIR, 'TL052_run_roc_test.R'),
         "--i", os.path.join(LOG_DIR, 'log_scores.csv')],
        universal_newlines=True)
    log_string(LOG_FOUT, out)

    # close the log file
    LOG_FOUT.close()
