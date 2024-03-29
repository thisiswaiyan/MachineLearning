from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['Sepallength', 'SepalWidth', 'Petallength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

#print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')

#print(train.head())

#print(train.shape)

def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    # Shuffle and repeat if you are in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
#print(my_feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes repectively
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
    
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


def input_fn1(features, batch_size=256):
    # Converts the inputs to a Dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False
    predict[feature] = [float(val)]
    
predictions = classifier.predict(input_fn=lambda: input_fn1(predict))

for pre_dict in predictions:
    print(pre_dict)
    class_id = pre_dict['class_ids'][0]
    probability = pre_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))