from prediction import create_model, load_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pandas as pd
from parameters import *


# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

# load the data using yahoo finance
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                feature_columns=FEATURE_COLUMNS)

# save the dataframe
data["df"].to_csv(ticker_data_filename)

# construct the model. these are called from parameters.py
# N_STEPS is sequence_length
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
# tensorboard lets you visualize the model performance during the training process
# TensorBoard: https://www.tensorflow.org/tensorboard
# tensorboard can be run either during or after training with tensorboard --logdir="logs"
# on tensorboard, the loss is huber loss or whatever loss you specified in LOSS in the paremters.
# the curve is the validation loss. it should decrease significantly over time. 
# increasing epochs should cause error to keep decreasing to a certain extent
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

# train the model and save the weights whenever we see 
# a new optimal model using ModelCheckpoint
# ModelCheckpoint: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint?version=stable
# ModelCheckpoint saves the model in each epoch during the training process. 
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)
