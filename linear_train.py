import tensorflow as tf

#from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
print(dftrain.head())
y_train = dftrain.pop('survived') # popping out the training data from 'survived' column
y_eval = dfeval.pop('survived') # popping out the testing data from 'survived' column
#print(dftrain.loc[0], y_train.loc[0])

#print(dftrain.describe())
#print(dftrain.age.hist(bins=20))

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function(): # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # create tf.data.Dataset object with data and its label data
        if shuffle:
            ds = ds.shuffle(1000)   # randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs)    # split dataset into batches 32 and repeat process for number of epochs times
        return ds # return a batch of the dataset
    return input_function # return a function object for use


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimator by passing the feature column we create earlier

linear_est.train(train_input_fn) # train a model
result = linear_est.evaluate(eval_input_fn) # get model metrics/stats by testing on test data

#clear_output() # clear console output
print(result['accuracy']) # the result variable is simply a dict of stats about our model
print(result)

result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[3])
print(y_eval.loc[3])
print(result[3]['probabilites'][1])