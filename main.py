import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PREFIX = './data/'

ATT_FILE = PREFIX + "MedianHouseValuePreparedCleanAttributes.csv"
LABEL_FILE = PREFIX + "MedianHouseValueOneHotEncodedClasses.csv"

train_rate = 0.8

attributes = pd.read_csv(ATT_FILE)
labels = pd.read_csv(LABEL_FILE)

x_train, x_test, t_train, t_test = train_test_split(attributes, labels, test_size=1 - train_rate, stratify=labels)

x_dev, x_test, t_dev, t_test = train_test_split(x_test, t_test, test_size=(1 - train_rate)/2, stratify=t_test)


print("x_train:", x_train.shape)
print("t_train:", t_train.shape)

print("x_dev:", x_dev.shape)
print("t_dev:", t_dev.shape)

print("x_test:", x_test.shape)
print("t_test:", t_test.shape)

INPUTS = x_train.shape[1]
OUTPUTS = t_train.shape[1]
NUM_TRAINING_EXAMPLES = int(round(x_train.shape[0] / 1))
NUM_DEV_EXAMPLES = int(round(x_dev.shape[0] / 1))
NUM_TEST_EXAMPLES = int(round(x_test.shape[0] / 1))

n_epochs = 20000
learning_rate = 0.1
batch_size = 200
n_neurons_per_layer = [150, 75, 25, 10]

X = tf.placeholder(dtype=tf.float32, shape=(None, INPUTS), name="X")
t = tf.placeholder(dtype=tf.float32, shape=(None, OUTPUTS), name="t")

hidden_layers = [tf.layers.dense(X, n_neurons_per_layer[0], activation=tf.nn.relu)]
for layer in range(1, len(n_neurons_per_layer)): hidden_layers.append(
    tf.layers.dense(hidden_layers[layer - 1], n_neurons_per_layer[layer], activation=tf.nn.relu))
net_out = tf.layers.dense(hidden_layers[len(n_neurons_per_layer) - 1], OUTPUTS)
y = tf.nn.softmax(logits=net_out, name="y")

for layer in range(len(n_neurons_per_layer)): print(hidden_layers[layer])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=net_out)
mean_log_loss = tf.reduce_mean(cross_entropy, name="cost")

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_log_loss)

correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

init = tf.global_variables_initializer()
accuracy_train_history = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in tqdm(range(n_epochs)):
        offset = (epoch * batch_size) % (NUM_TRAINING_EXAMPLES - batch_size)
        sess.run(train_step,
                 feed_dict={X: x_train[offset:(offset + batch_size)], t: t_train[offset:(offset + batch_size)]})
    accuracy_test = accuracy.eval(feed_dict={X: x_test[:NUM_TEST_EXAMPLES], t: t_test[:NUM_TEST_EXAMPLES]})
    test_predictions = y.eval(feed_dict={X: x_test[:NUM_TEST_EXAMPLES]})
    test_correct_preditions = correct_predictions.eval(
        feed_dict={X: x_test[:NUM_TEST_EXAMPLES], t: t_test[:NUM_TEST_EXAMPLES]})

print("Accuracy for the TEST set: " + str(accuracy_test))

test_rounded_predictions = np.round(test_predictions)
print(test_rounded_predictions[:10])
print(t_test[:10])  # target classes

print(test_correct_preditions[:10])
