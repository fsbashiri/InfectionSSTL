"""
Project: Identifying Infected Patients Using Semi-supervised and Transfer Learning
Branch: Transfer Learning
Author: Azi bashiri
Last Modified: October 2021
Description: A few functions and definitions to use for data quality check, AUC CI calculations (bootstrap), and more.
                When training a model, it is recommended to read in data with get_data_v2 function defined here.

Note: when using get_data_v2 for your project, make sure loc_cat variable is handled properly, and modify cols_to_drop
                variable as needs to be.
"""

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import os
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import metrics
from pathlib import Path

# metrics to monitor during training and validation
METRICS = ['accuracy', metrics.AUC(curve='ROC', name='auroc'), metrics.AUC(curve='PR', name='auprc')]
# project directory
PROJ_DIR = os.path.abspath('')
# logging directory and output file
LOG_DIR = os.path.join(PROJ_DIR, 'Output', 'log_'+datetime.datetime.now().strftime("%Y%m%d-%H%M"))
# other code...
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)  # create (log_date-time) folder if doesn't exist
# if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)  # create (log_date-time) folder if doesn't exist
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')  # create a log file


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def log_string(log_obj=None, out_str="\n"):
    """
    Function to log strings in a [.txt] file
    :param log_obj: The object into which log_string writes.
    :param out_str: String to log. If no out_str is provided, a "new line" will be written.
    :return: None
    """
    if log_obj is not None:
        log_obj.write(out_str+'\n')
        log_obj.flush()
    print(out_str)
    return None


def check_reshape_is_ok(data, hour_block_indx=54, tseq_len=25):
    """
    THIS IS NOT AN EXACT CHECK
    if reshape is ok, then hour_block slice of the formatted data is equal to np.arange(25).
    Then, the sum of all rows must be equal to 300
    :param data: 3D numpy array of size (sample, tseq_len, feature_len)
    :param hour_block_indx: the index of "hour_block" in the column names
    :param tseq_len: length of time sequence (time axis)
    :return Boolean
    usage: check_reshape_is_ok(np_reshaped_data)
    """
    return set(data[:, :, hour_block_indx].sum(axis=1)) == {np.arange(tseq_len).sum()}


def check_column_is_all_1_or_0(data, tseq_len=25):
    """
    A function to verify a column from a dataframe (e.g., target column) contains one value (i.e., either 0 or 1) for at
    all time steps.
    :param data: dataframe containing only one column (the target column)
    :param tseq_len: used to reshape data from dataframe to 2D numpy array
    :return Boolean
    usage: check_column_is_all_1_or_0(df_train['sepsis3_bool'])
    """
    # convert data to 2D numpy array of shape (sample x tseq_len)
    data = data.to_numpy().reshape((-1, tseq_len))
    # sum along axis
    row_sum = data.sum(axis=1)
    # row_sum must contain only 0 or 1*25=25
    return set(np.unique(row_sum)) == {0.0, 1.0*tseq_len}


def check_pull_forward(data, exclude=None):
    """
    check if data is pulled forward for each and every sample
    :param data: 2D numpy array in shape (tseq_len x feature_len)
    :param exclude: list of column indices to drop (excluded from checking process). The remaining columns cannot be of
    dtype 'object'. Example: drop columns [0,1,2,3,57,58,59] associated with ['Unnamed: 0',site','patient_id',
    'encounter_id','loc_cat','sepsis3','sepsis3_bool','infection','traintest01']
    :return Boolean
    usage:
    result = []
    for i in range(reshaped_data_array.shape[0]):
        result.append(check_pull_forward(reshaped_data_array[i,:,:]))
    np.all(result)
    """
    data = np.delete(data, exclude, 1).astype(float)
    # for every column, index of the last occurrence of NaN. if no Nan exist, output -1
    max_nan_ids = np.where(np.isnan(np.sum(data, axis=0)), 24 - np.argmax(np.isnan(data)[::-1, :], axis=0), -1)
    # for every column, index of the first occurrence of a number. if all NaN, output 25
    min_num_ids = np.where(np.sum(np.isnan(data), axis=0) == 25, 25, np.argmin(np.isnan(data), axis=0))
    # for every row, it must be: max_nan_ids <= min_num_ids
    return np.all(max_nan_ids <= min_num_ids)


def data_qc(df, t_len=25, target="infection", log_obj=None):
    """
    A few data quality checks.
    :param df: input dataframe.
    :param t_len: length of time sequence, e.g., 25.
    :param target: string. Name of the target column.
    :param log_obj: a file object that the function will log information into. If none, it only prints out information.
    :return: None. The output of each check-up will be printed out.
    """
    # list of variables
    cols = df.columns.to_list()
    log_string(log_obj, f"*\t List of variables: \n{cols}")
    # number of NAs for each column (variable)
    num_nas = df.isna().sum(axis=0)   # Pandas series
    # log_string(log_obj, f"*\t Number of NAs: \n{num_nas}")
    nonzero_na_index = num_nas.index[num_nas > 0].to_list()
    log_string(log_obj, f"*\t Columns with NAs: {nonzero_na_index}")
    # check if hour_block is in order, and reshape will not distort the data
    df_reordered = df.to_numpy().reshape((-1, t_len, len(cols)))
    log_string(log_obj, f"*\t Shape of reformatted data: {df_reordered.shape}")
    log_string(log_obj, f"*\t Hour index for samples are in order: "
                        f"{check_reshape_is_ok(df_reordered, hour_block_indx=cols.index('hour_block'))}")
    # check if data is pulled forward for each and every sample
    result = []   # store check results
    exclude_names = df.dtypes[df.dtypes == object].index.to_list()   # name of columns with dtype==object
    exclude_list = [cols.index(item) for item in exclude_names]   # index of columns with dtype=object in cols
    for i in range(df_reordered.shape[0]):
        result.append(check_pull_forward(df_reordered[i, :, :], exclude=exclude_list))
        if not result[-1]:
            log_string(log_obj, f"*\t\t Pull forward error in sample: {i}")
    log_string(log_obj, f"*\t Verification of pull forward for all columns except "
                        f"{exclude_names} passed: {np.all(result)}")
    # check target column contains only either 0 or 1
    log_string(log_obj, f"*\t {target} column contains only either 0 or 1: {check_column_is_all_1_or_0(df[target])}")
    return None


def auc_ci(y_true, y_pred, interval=0.95, n_bootstraps=1000, seed=123):
    """
    A function that estimates AUC CI with bootstraping the ROC computations. This is different from DeLong method, even
    though the results are very close.
    :param y_true: true outcome
    :param y_pred: predicted scores
    :param interval: percentage of CI in decimal points, e.g., for 95% CI the input is as 0.95
    :param n_bootstraps: number of bootstraps for estimating the CI
    :param seed: seed point. It controls reproducibility
    :return: lower and upper bound of AUC CI
    """
    bootstrapped_scores = []
    rng = np.random.RandomState(seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    # Sort the samples to get a confidence interval
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    interval_bound = [(1. - interval) / 2., (1. + interval) / 2.]
    confidence_lower = sorted_scores[int(interval_bound[0] * len(sorted_scores))]
    confidence_upper = sorted_scores[int(interval_bound[1] * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper))
    return confidence_lower, confidence_upper


def get_data_v2(filenames, loadcreate01=0, target="infection", log_obj=None, scaling={}, sc_path='', expand_dim=False):
    """
    This function reads an input file and processes the data. The data must have been pulled forward with no NA at any
    column except loc_cat. The processing section includes: filling backward the loc_cat variable, converting the
    loc_cat into one-hot encoding, dropping extra columns, scaling the data, and reformatting it into desired shape.
    In the end, it returns x_train and y_train.
    :param filenames: list of input filenames to read and process, e.g. [fname1, fname2]
    :param loadcreate01: bool, whether to load or create a scaler object.
            load scaler object from SC_PATH if loadcreate01==0; create an object and fit to data otherwise
    :param log_obj: file object to log information into. If none, it will only print out information
    :param scaling: a dictionary containing scaling information. keys are "scaling", "min_max", and "mean_Std". values
            must be boolean
    :param sc_path: path to scaling object. Depending on loadcreate01, it will be loaded from or stored at sc_path
    :param expand_dim: a boolean argument to expand dimension for CONV2D. True only for mdl_cnnlstm models.
    :return: x_train, y_train
    """
    log_string(log_obj, '\nInput data')
    log_string(log_obj, '===========================\n')
    # read data
    df_input_data = pd.DataFrame()
    for fname in filenames:
        log_string(log_obj, f"Reading file: {fname}")
        df_tmp = pd.read_csv(fname)
        log_string(log_obj, f"\t Shape of data: {df_tmp.shape}")
        df_input_data = pd.concat([df_input_data, df_tmp], ignore_index=True).reset_index(drop=True)
    tseq_len = len(df_input_data['hour_block'].unique())  # time sequence length: 25

    # A few checks on the input file
    log_string(log_obj, f"\nInput data inspection before pre-processing:")
    data_qc(df_input_data, t_len=tseq_len, target=target, log_obj=log_obj)

    # use lines commented if loc_cat variable is not handled in advance. For example, it has not been filled backward,
    # contains unknown values, not converted to numeric categories
    # fill backward the loc_cat variable
    # df_input_data['loc_cat'].fillna(method='bfill', inplace=True) # not necessary, since Kyle has taken care of it
    # one-hot encoding of loc_cat variable (Ward:1, ICU:2, OR:3, ER:4, Inv/Diag/Other:6)
    # loc_cat_drop_list = ['loc_cat_' + item for item in set(df_input_data['loc_cat'].unique()) - {'WARD', 'ER', 'OR', 'ICU', 'INVT/DIAG/OTHER'}]
    # df_input_data = pd.get_dummies(df_input_data, columns=['loc_cat']).drop(loc_cat_drop_list, axis=1)
    # df_input_data.rename({'loc_cat_ER': 'loc_catM.4', 'loc_cat_ICU': 'loc_catM.2',
    #                       'loc_cat_INVT/DIAG/OTHER': 'loc_catM.6', 'loc_cat_OR': 'loc_catM.3',
    #                       'loc_cat_WARD': 'loc_catM.1'}, axis=1, inplace=True)
    # use line below if loc_cat variable is already handled. No unknown value must exist
    df_input_data = pd.get_dummies(df_input_data, columns=['loc_cat'])
    # list of columns to drop
    cols_to_drop = ['site', 'patient_id', 'encounter_id', 'hour_block', 'sepsis3', 'sepsis3_bool',
                    'infection', 'traintest01', 'troponin', 'infxn_enc', 'last_vitals_hour']
    df_target = df_input_data[[target]]  # keep target column separate
    df_input_data.drop(labels=cols_to_drop, axis=1, inplace=True)  # drop extra columns
    features = df_input_data.columns.to_list()

    # check sc_path
    if sc_path == '':
        sc_path = os.path.join(os.path.abspath(__file__), 'scaling_obj.pkl')

    # Scaling the data
    # SCALING = {'scaling': True, 'min_max': True, 'mean_Std': False}
    if scaling['scaling']:
        log_string(log_obj, '\nScaling the data')
        if scaling['min_max']:  # normalization (min=0, max=1)
            log_string(log_obj, '** Normalizing to min=0, max=1')
            if loadcreate01 == 1:   # create scaler object
                scaler = MinMaxScaler().fit(df_input_data[df_input_data.columns])
                with open(sc_path, 'wb') as output:
                    pickle.dump(scaler, output)
                    log_string(log_obj, f"Scaler object is saved at: {sc_path}")
            else:                   # load scaler object otherwise
                scaler = pickle.load(open(sc_path, 'rb'))
                log_string(log_obj, f"Scaler object is read from: {sc_path}")
            df_input_data[df_input_data.columns] = scaler.transform(df_input_data[df_input_data.columns])
            log_string(log_obj, f"{df_input_data.describe()}")
        elif scaling['mean_std']:  # standardization (mean=0, std=1)
            log_string(log_obj, '** Standardize to mean=0, std=1')
            if loadcreate01 == 1:  # create scaler object
                scaler = StandardScaler().fit(df_input_data[df_input_data.columns])
                with open(sc_path, 'wb') as output:
                    pickle.dump(scaler, output)
                    log_string(log_obj, f"Scaler object is saved at: {sc_path}")
            else:                  # load scaler object otherwise
                scaler = pickle.load(open(sc_path, 'rb'))
                log_string(log_obj, f"Scaler object is read from: {sc_path}")
            # df_input_data[df_input_data.drop(['loc_catM.1','loc_catM.2','loc_catM.3','loc_catM.4','loc_catM.6'],
            #                                  axis=1).columns] = \
            #     scaler.fit_transform(df_input_data.drop(['loc_catM.1','loc_catM.2','loc_catM.3','loc_catM.4',
            #                                              'loc_catM.6'], axis=1))
            df_input_data[df_input_data.columns] = scaler.transform(df_input_data[df_input_data.columns])
            # print(df_input_data.describe())
        else:
            log_string(log_obj, 'UserWarning! Scaling is NOT performed. Choose min_max or mean_std by making them True.')

    # Pre-processing and reshaping
    x = df_input_data.to_numpy().reshape(-1, tseq_len, df_input_data.shape[1])
    # expand dimentions and permute. For Conv2D it must be: (n_sample,n_feature,n_timeseq,n_channel)
    if expand_dim:
        x = np.expand_dims(x, axis=3)
        x = np.transpose(x, (0, 2, 1, 3))
    y = np.max(df_target.to_numpy().reshape((-1, tseq_len)), axis=1)

    # Log info about training data
    log_string(log_obj, '\nData after pre-processing:')
    log_string(log_obj, f"x shape: {x.shape}")
    log_string(log_obj, f"y shape: {y.shape}")
    log_string(log_obj, f"y stats: {target}==1 ({y.sum()}) - {target}==0 ({y.shape[0] - y.sum()})")
    log_string(log_obj, f"Predictor variables: {df_input_data.columns.to_list()}")
    return x, y, features


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

