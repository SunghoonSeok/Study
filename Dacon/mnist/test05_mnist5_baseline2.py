import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf


def get_tr_data(SEED):    
    tr_data = pd.read_csv("./data/train.csv", index_col = 0).values
    tr_data = shuffle(tr_data, random_state = SEED)

    tr_X = tf.convert_to_tensor(tr_data[:, 2:], dtype = tf.float32)
    tr_Y = tf.squeeze(tf.convert_to_tensor(tr_data[:, 0], dtype = tf.int32))

    return tr_X, tr_Y


def get_ts_data():
    ts_data = pd.read_csv("./data/test.csv", index_col = 0).values
    ts_X = tf.convert_to_tensor(ts_data[:, 1:], dtype = tf.float32)

    return ts_X


IMAGE_SIZE = [28, 28]
RESIZED_IMAGE_SIZE = [256, 256]

@tf.function
def _reshape_and_resize_tr(flatten_tensor, label):
    image_tensor = tf.reshape(flatten_tensor, (*IMAGE_SIZE, 1))
    image_tensor = tf.keras.layers.experimental.preprocessing.Resizing(*RESIZED_IMAGE_SIZE)(image_tensor)
    image_tensor = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(image_tensor)
    return image_tensor, label


@tf.function
def _reshape_and_resize_ts(flatten_tensor): # without label
    image_tensor = tf.reshape(flatten_tensor, (*IMAGE_SIZE, 1))
    image_tensor = tf.keras.layers.experimental.preprocessing.Resizing(*RESIZED_IMAGE_SIZE)(image_tensor)
    image_tensor = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(image_tensor)
    return image_tensor

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

def print_shapes(l, l_name, end = False):
    assert type(l) is list

    print(f"len({l_name}) = {len(l)}")
    for i in range(len(l)):
        print(f"{l_name}[{i}]: {l[i]}")
    if not end:
        print()
    

def make_dataset(X, Y, type):
    if type == "train":
        assert X.shape[0] == Y.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)
                            ).map(_reshape_and_resize_tr, num_parallel_calls = AUTO
                            ).cache(
                            ).shuffle(20000, reshuffle_each_iteration = True
                            ).batch(BATCH_SIZE
                            ).prefetch(AUTO)   

    elif type == "validation":
        assert X.shape[0] == Y.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)
                            ).map(_reshape_and_resize_tr, num_parallel_calls = AUTO
                            ).cache(
                            # ).shuffle(20000, reshuffle_each_iteration = True
                            ).batch(BATCH_SIZE
                            ).prefetch(AUTO)

    elif type == "test":
        assert Y is None
        dataset = tf.data.Dataset.from_tensor_slices((X)
                            ).map(_reshape_and_resize_ts, num_parallel_calls = AUTO
                            ).cache(
                            # ).shuffle(20000, reshuffle_each_iteration = True
                            ).batch(BATCH_SIZE
                            ).prefetch(AUTO)
                            
    else:
        raise ValueError(f"Unknown type: {type}")

    return dataset


def make_fold(X, Y, K = 5):
    assert X.shape[0] == Y.shape[0]
    assert len(list(Y.shape)) == 1

    fold_size = [X.shape[0] // K] * (K - 1) + [X.shape[0] - (X.shape[0] // K) * (K - 1)]

    splited_X = tf.split(X, fold_size, 0) # list
    splited_Y = tf.split(Y, fold_size, 0) # list

    tr_list = [(tf.concat(splited_X[:i] + splited_X[i+1:], axis = 0), tf.concat(splited_Y[:i] + splited_Y[i+1:], axis = 0)) for i in range(K)]
    vl_list = [(splited_X[i], splited_Y[i]) for i in range(K)]

    tr_datasets = [make_dataset(tr_X, tr_Y, "train") for tr_X, tr_Y in tr_list]
    vl_datasets = [make_dataset(vl_X, vl_Y, "validation") for vl_X, vl_Y in vl_list]

    return tr_datasets, vl_datasets

