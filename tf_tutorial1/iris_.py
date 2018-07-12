from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print('Tensorflow version: {}'.format(tf.VERSION))
print('Eager Execution: {}'.format(tf.executing_eagerly()))


train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))


# input data set
def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)
    features = tf.reshape(parsed_line[:-1],shape=(4,))
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label


# create input dataset
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)
train_dataset = train_dataset.map(parse_csv)
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(32)

feature, label = iter(train_dataset).__next__()
print("example feature: ", feature[0])
print("example label: ",label[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation= "relu", input_shape=(4,)),
    tf.keras.layers.Dense(10, activation= "relu"),
    tf.keras.layers.Dense(3)
])


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, input, target):
    with tf.GradientTape() as tape:
        loss_value = loss(model, input, target)
    return tape.gradient(loss_value, model.variables)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# train our model
train_loss_result = []
train_accuracy_result = []

# number of epoch is 200
for epoch in range(201):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # use batch of 32
    for x, y in train_dataset:
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        epoch_loss_avg(loss(model, x, y))
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_result.append(epoch_loss_avg.result())
    train_accuracy_result.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

# visualize the result over epoches
fig, axes = plt.subplots(2, sharex=True, figsize = (12,8))
axes[0].set_ylabel('loss')
axes[0].plot(train_loss_result)

axes[1].set_ylabel('accuracy')
axes[1].set_xlabel('epoch')
axes[1].plot(train_accuracy_result)

plt.show()

# evaluation of the model efficiency
test_url = "http://download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)             # skip header row
test_dataset = test_dataset.map(parse_csv)      # parse each row with the funcition created earlier
test_dataset = test_dataset.shuffle(1000)       # randomize
test_dataset = test_dataset.batch(32)           # use the same batch size as the training set

test_accuracy = tfe.metrics.Accuracy()

for x, y in test_dataset:
    prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("test set accuracy: {:.3%}".format(test_accuracy.result()))





















