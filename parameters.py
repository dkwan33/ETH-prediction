import os
import time
from tensorflow.keras.layers import LSTM

# Window size or the historical sequence length
# the number of steps is the number of days of prices that will be used to predict the lookup time step
N_STEPS = 50
# Lookup step, 1 is the next day. 3 is the next 3 days. etc... 
# essentially how far into the future you want to predict.
LOOKUP_STEP = 1

# whether to scale feature columns & output price as well. Will scale from 0 to 1.
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"

# whether to split the training/testing set by date
# false means the data is split randomly into training and testing sets using sklearn's train_test_split() function.
# true means the data is split in date order
# if split by date is false, then the final plot will show the prices of the testing set spread over the whole dataset along with corresponding predicted prices
# if split by date is true, then the testing set will be the last test_size percentage of the total dataset
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

# test ratio size, 0.2 is 20%
# ie: 80% train, 20% test
TEST_SIZE = 0.2

# financial features to use to predict the next price value
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# the date. used for pulling data with yahoo finance
# ie: 2021-04-01
# date_now = time.strftime("%Y-%m-%d") #current date
date_now = "2021-04-01" #setting it to april 1st to keep things consistent.

### model parameters that will be used in create_model

#this is the number of LSTM layers you want in the LSTM stack. or rather RNN layers if you use SimpleRNN or GRU.
N_LAYERS = 2
# LSTM cell. changing this would change the cell type from LSTM to some other type of RNN cell such as SimpleRNN or GRU.
CELL = LSTM
# UNITS is the number of RNN/LSTM cell units.
# 256 = 256 LSTM neurons
UNITS = 256
# the dropout rate after each RNN/LSTM layer. 0.4 would be 40% dropout.
# dropout rate is the probability of not training a given node in a layer. where 0.0 is no dropout at all. 
# dropout rate helps prevent overfitting.
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# LOSS is the type of Loss function to use for the regression. choices are huber loss, mean absolute error, or mean squared error.
# valid LOSS parameters are: "huber_loss", "mae", "mse"
LOSS = "huber_loss"

# OPTIMIZER is the optiimziation algorithm to use. Default is "adam".
OPTIMIZER = "adam"

# Batch Size is the number of data samples to use on each training iteration. 
BATCH_SIZE = 64

# Epochs is the number of times that the learning algorithm will pass through the ENTIRE training dataset. More is better.
EPOCHS = 5

ticker = "ETH-USD"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"
