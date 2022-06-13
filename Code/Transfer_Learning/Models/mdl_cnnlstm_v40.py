"""
Model: CNN-LSTM v.4
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
# which trainable layers to train/freeze. Last layer must always be trained (set to True)
layers_to_train = [False, False, False] + [False, True, True]


def create_model(hp):
    """
    Create a model (implementation 4 of CNN_RL)
    n_layer x {num_conv x {Conv2D(filters, 3, activation)} -> MaxPooling2D(2) -> option(dropout)} -> reshape
      -> ConvLSTM2D -> Flatten -> Dense -> dropout -> Softmax
    :param hp: keras-tuner hyperparameter object
    :return: a compiled keras model
    """
    # create n-layer CNN_RL w/ dropout based on hp
    model = models.Sequential()
    # n_layers of {Conv2D -> MaxPooling2D -> Dropout}
    for i in range(hp.Int('num_layers', min_value=1, max_value=2, default=1)):
        # included in hp: n_filters, activation
        # NOT included in hp: kernel_size, padding, pool_size, strides
        for j in range(hp.Int('num_conv_' + str(i), min_value=1, max_value=2, default=1)):
            filters = hp.Int('filters_' + str(i) + '_' + str(j), min_value=20, max_value=200, default=32)
            activation = hp.Choice('activation_' + str(i) + '_' + str(j), ['relu', 'sigmoid', 'tanh'], default='relu')
            model.add(layers.Conv2D(input_shape=[56, 25, 1],  # x_train.shape[1:]
                                    filters=filters,
                                    kernel_size=3,
                                    activation=activation,
                                    ))
        model.add(layers.MaxPooling2D(pool_size=2))
        dropout_on = hp.Boolean('dropout_on_' + str(i), default=True)
        dropout_rate_1 = hp.Float('dropout_rate_1' + str(i), min_value=0.0, max_value=0.9, default=0.05)
        if dropout_on:
            model.add(layers.Dropout(dropout_rate_1))

    # more hyper-parameters
    convlstm_units = hp.Int('convlstm_units', min_value=10, max_value=50, default=32)
    convlstm_kernel = hp.Int('convlstm_kernel', min_value=3, max_value=5, default=3)
    dense_units = hp.Int('dense_units', min_value=10, max_value=200, default=128)
    dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.0, max_value=0.9, default=0.05)

    # Reshape -> ConvLSTM2D
    model.add(layers.Reshape([1] + list(model.output_shape[1:])))
    model.add(layers.ConvLSTM2D(convlstm_units, kernel_size=convlstm_kernel, padding='same', return_sequences=True))

    # Flatten -> Dense -> Dropout -> Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(units=dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate_2))
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
    # model.pop()   # pop dropout before the last dense layer
    # model.pop()   # pop Dense layer before the output layer
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
    x_train = np.empty([None, 56, 25, 1])
    with open(os.path.join(os.path.abspath(''), 'log_train.txt'), 'w') as LOG_FOUT:
        hps = HyperParameters
        my_model = create_model(hps)
