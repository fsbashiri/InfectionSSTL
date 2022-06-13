"""
Model: LSTM/GRU
If you want to use this model, make sure to check the input_shape argument. It is set to a fixed number, as keras-tuner
does not accept two input arguments for the create_model function.
Two variables are hard-coded in this python script: "model_path" and "layers_to_train". These variables are used with
fine tuning a model. They have to be updated in the code accordingly.

Author: Azi Bashiri
Date: April 2021
"""

import os
import numpy as np
# import pandas as pd
import tensorflow as tf
from TL099_utils import log_string, METRICS, PROJ_DIR, LOG_FOUT
from keras import models, layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from kerastuner.engine.hyperparameters import HyperParameters

# trained model path - used for fine tuning
model_path = os.path.join(PROJ_DIR, 'Output', 'log_YYYYMMDD-hhmm', 'best_model.h5')
# which layers to train/freeze. Last layer must always be trained (set to True)
layers_to_train = [False, False] + [True, True]


def create_model(hp):
    """
    Create n-layer LSTM/GRU w/ dropout
    :param hp: keras-tuner hyperparameter object
    :return: a compiled keras model
    """
    # create n-layer LSTM or GRU w/ dropout based on hp
    model = models.Sequential()
    for i in range(hp.Int('num_layers', min_value=1, max_value=3, default=1)):
        cell_type = hp.Choice('cell_type_' + str(i), ['LSTM', 'GRU'], default='LSTM')
        units = hp.Int('units_' + str(i), min_value=20, max_value=500, default=20)
        dropout = hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.9, default=0.1)
        rec_dropout = hp.Float('rec_dropout_' + str(i), min_value=0.0, max_value=0.9, default=0.1)
        activation = hp.Choice('activation_' + str(i), ['tanh', 'sigmoid', 'relu'], default='relu')
        if cell_type == 'LSTM':
            model.add(layers.LSTM(units,
                                  input_shape=[25, 56],
                                  dropout=dropout,
                                  recurrent_dropout=rec_dropout,
                                  activation=activation,
                                  return_sequences=True if ((hp.get('num_layers') > 1) and (
                                              i + 1 < hp.get('num_layers'))) else False))
        elif cell_type == 'GRU':
            model.add(layers.GRU(units,
                                 input_shape=[25, 56],
                                 dropout=dropout,
                                 recurrent_dropout=rec_dropout,
                                 activation=activation,
                                 return_sequences=True if (
                                             (hp.get('num_layers') > 1) and (i + 1 < hp.get('num_layers'))) else False))
        else:
            raise ValueError("unexpected cell type: %r" % cell_type)
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary(print_fn=lambda x: log_string(LOG_FOUT, x))

    # hyper-params related to the optimizer
    opt = hp.Choice('optimizer', ['Adam', 'SGD', 'RMSProp'], default='Adam')
    with hp.conditional_scope('optimizer', ['SGD', 'RMSProp']):
        momentum = hp.Float('momentum', min_value=0.0, max_value=1.0, default=0.0)
    with hp.conditional_scope('optimizer', 'SGD'):
        nesterov = hp.Boolean('nesterov', default=False)
    # define a learning_rate schedule: ExponentialDecay
    # I could also use conditional_scope for lr_decay_nstep and lr_decay_rate conditioning on lr_decay_scheduling
    lr_base = hp.Float('lr', min_value=0.0001, max_value=0.001, default=0.001)
    lr_decay_scheduling = hp.Boolean('lr_decay_scheduling', default=True)
    lr_decay_nstep = hp.Int('lr_decay_nstep', min_value=300, max_value=500, default=300)
    lr_decay_rate = hp.Float('lr_decay_rate', min_value=0.7, max_value=0.99, default=0.95)
    # optimizers accept both a fixed lr value and a lr_schedule as input
    if lr_decay_scheduling is True:
        lr_schedule = ExponentialDecay(lr_base,
                                       decay_steps=lr_decay_nstep,
                                       decay_rate=lr_decay_rate,
                                       staircase=True)
    else:
        lr_schedule = lr_base

    # setting up the optimizer
    if opt == 'Adam':
        optimizer = tf.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5)
    elif opt == 'SGD':
        optimizer = tf.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum, nesterov=nesterov, clipnorm=0.5)
    elif opt == 'RMSProp':
        optimizer = tf.optimizers.RMSprop(learning_rate=lr_schedule, momentum=momentum, clipnorm=0.5)
    else:
        raise ValueError("unexpected optimizer name: %r" % (hp.get('optimizer'),))

    # compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)

    # log HP config
    log_string(LOG_FOUT, f"HP config: {hp.get_config()['values']}\n")
    return model


def log_string_search_space_summary(tuner):
    """
    The built-in search_space_summary method does not return a printable string. I re-wrote it with log_string
    :param tuner: keras-tuner tuner object
    :return: None
    """
    hp = tuner.oracle.get_space()
    log_string(LOG_FOUT, f"Default search space size: {len(hp.space)}")
    for p in hp.space:
        config = p.get_config()
        name = config.pop('name')
        log_string(LOG_FOUT, f"- {name} ({p.__class__.__name__})")
        log_string(LOG_FOUT, f"\t{config}")
    return None


def tune_model(hp):
    """
    This function is used for retraining a pre-trained model with some layers kept frozen.
    :param hp: hyperparameters object
    :return: compiled model
    """
    # load a model
    log_string(LOG_FOUT, '\nLoading a clean copy of the base model')
    model = models.load_model(model_path)
    base_model_lr = model.optimizer.get_config()['learning_rate']  # base model learning_rate config
    if type(base_model_lr) == dict:
        base_model_lr = base_model_lr['config']['initial_learning_rate']
    # index of layers with trainable weights
    trainable_layers = [i for i in range(len(model.layers)) if len(model.layers[i].trainable_weights) != 0]
    log_string(LOG_FOUT, f"Layers to train: {layers_to_train}")

    # drop as many layers as you like. In our experiments, best results were obtained by dropping only the last layer
    model.pop()   # pop last Dense layer (output layer)
    model.add(layers.Dense(units=hp.Int('dense_units', min_value=10, max_value=100, default=30),
                           activation='relu', name="tl_dense"))
    model.add(layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.9, default=0.05), name='tl_dropout'))
    model.add(layers.Dense(units=1, activation='sigmoid', name="output_dense"))
    # freeze some (or none) layers
    for i, layer_id in enumerate(trainable_layers):
        if not layers_to_train[i]:
            model.layers[layer_id].trainable = False
    # log summary after freezing
    model.summary(print_fn=lambda x: log_string(LOG_FOUT, x))

    # setting up the optimizer
    opt = hp.Choice('optimizer', ['Adam', 'SGD', 'RMSProp'], default='Adam')
    with hp.conditional_scope('optimizer', ['SGD', 'RMSProp']):
        momentum = hp.Float('momentum', min_value=0.0, max_value=1.0, default=0.0)
    with hp.conditional_scope('optimizer', 'SGD'):
        nesterov = hp.Boolean('nesterov', default=False)
    # define a learning_rate schedule: ExponentialDecay
    # I could also use conditional_scope for lr_decay_nstep and lr_decay_rate conditioning on lr_decay_scheduling
    lr_base = hp.Float('lr', min_value=base_model_lr/100.0, max_value=base_model_lr*1.0, default=base_model_lr*1.0)
    lr_decay_scheduling = hp.Boolean('lr_decay_scheduling', default=True)
    lr_decay_nstep = hp.Int('lr_decay_nstep', min_value=10, max_value=30, default=20)
    lr_decay_rate = hp.Float('lr_decay_rate', min_value=0.7, max_value=0.99, default=0.95)
    # optimizers accept both a fixed lr value and a lr_schedule as input
    if lr_decay_scheduling is True:
        lr_schedule = ExponentialDecay(lr_base,
                                       decay_steps=lr_decay_nstep,
                                       decay_rate=lr_decay_rate,
                                       staircase=True)
    else:
        lr_schedule = lr_base

    if opt == 'Adam':
        optimizer = tf.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5)
    elif opt == 'SGD':
        optimizer = tf.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum, nesterov=nesterov, clipnorm=0.5)
    elif opt == 'RMSProp':
        optimizer = tf.optimizers.RMSprop(learning_rate=lr_schedule, momentum=momentum, clipnorm=0.5)
    else:
        raise ValueError("unexpected optimizer name: %r" % (hp.get('optimizer'),))

    # compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)

    # log HP config
    log_string(LOG_FOUT, f"HP config: {hp.get_config()['values']}\n")
    return model


def fine_tune(mdl_path='best_model.h5', optimizer=tf.optimizers.Adam(learning_rate=1e-4)):
    """
    This function is used to prepare a model for the fine-tuning process. It loads a pre-trained model, makes sure all
    layers are trainable, then compiles and returns the model.
    :param mdl_path: path to the learned model that will be loaded for fine-tuning
    :param optimizer: optimizer object
    :return: compiled model
    """
    # fine tune
    model = models.load_model(mdl_path)
    model.trainable = True
    log_string(LOG_FOUT, f"Trainable layers: "
                         f"{[i for i in range(len(model.layers)) if len(model.layers[i].trainable_weights) != 0]}")
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=METRICS)
    return model


if __name__ == '__main__':
    x_train = np.empty([None, 25, 56])
    with open(os.path.join(os.path.abspath(''), 'log_train.txt'), 'w') as LOG_FOUT:
        hps = HyperParameters
        my_model = create_model(hps)
