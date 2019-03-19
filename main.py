import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

PREFIX = './data/'

ATT_FILE = PREFIX + "MedianHouseValuePreparedCleanAttributes.csv"
LABEL_FILE = PREFIX + "MedianHouseValueOneHotEncodedClasses.csv"

train_ratio = 0.8

attributes = pd.read_csv(ATT_FILE)
labels = pd.read_csv(LABEL_FILE)

x_train, x_test, t_train, t_test = train_test_split(attributes, labels, test_size=1 - train_ratio, stratify=labels)

x_dev, x_test, t_dev, t_test = train_test_split(x_test, t_test, test_size=0.5, stratify=t_test)

print("x_train:", x_train.shape)
print("t_train:", t_train.shape)

print("x_dev:", x_dev.shape)
print("t_dev:", t_dev.shape)

print("x_test:", x_test.shape)
print("t_test:", t_test.shape)

INPUTS = x_train.shape[1]
OUTPUTS = t_train.shape[1]
NUM_TRAINING_EXAMPLES = x_train.shape[0]
NUM_DEV_EXAMPLES = x_dev.shape[0]
NUM_TEST_EXAMPLES = x_test.shape[0]

n_epochs = 200
batch_size = 256
n_neurons_per_layer = [256, 128, 64]

X = tf.placeholder(dtype=tf.float32, shape=(None, INPUTS), name="X")
t = tf.placeholder(dtype=tf.float32, shape=(None, OUTPUTS), name="t")
training = tf.placeholder(tf.bool, name='training')
learning_rate = tf.placeholder(tf.float32, name='lr')

hidden_layers = [tf.layers.dense(X, n_neurons_per_layer[0], activation=tf.nn.relu)]
for layer in range(1, len(n_neurons_per_layer)):
    hidden_layers.append(
        tf.layers.dropout(
            hidden_layers[-1],
            rate=0.2,
            noise_shape=None,
            seed=None,
            training=training,
            name=f'dropout_{layer-1}'
        )
    )
    hidden_layers.append(
        tf.layers.dense(
            hidden_layers[-1], n_neurons_per_layer[layer],
            activation=tf.nn.relu,
            name=f'dense_{layer}'
        )
    )
net_out = tf.layers.dense(hidden_layers[len(n_neurons_per_layer) - 1], OUTPUTS)
y = tf.nn.softmax(logits=net_out, name="y")

for layer in hidden_layers:
    print(layer)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=net_out)

# loss definition
loss = tf.reduce_mean(cross_entropy, name="cost")
# l2_loss = tf.losses.get_regularization_loss()
# loss += l2_loss

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


init = tf.global_variables_initializer()
accuracy_train_history = []
minibatch_size = 256
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1, n_epochs + 1):
        # offset = (epoch * batch_size) % (NUM_TRAINING_EXAMPLES - batch_size)
        # sess.run(train_step,
        #          feed_dict={X: x_train[offset:(offset + batch_size)], t: t_train[offset:(offset + batch_size)]})
        losses = []
        for j in range(len(x_train) // minibatch_size):
            sess.run(train_step,
                     feed_dict={X: x_train[j * minibatch_size:(j + 1) * minibatch_size],
                                t: t_train[j * minibatch_size:(j + 1) * minibatch_size],
                                learning_rate: 0.01 if epoch <= 100 else 0.001,
                                training: True})
            val_loss = sess.run(loss, feed_dict={X: x_dev, t: t_dev, training: False})
            losses.append(val_loss)

        if epoch % 10 == 0:
            accuracy_dev = accuracy.eval(feed_dict={X: x_dev, t: t_dev, training: False})
            print("Accuracy for the DEV set: " + str(accuracy_dev))
        else:
            print(f'Epoch {epoch}/{n_epochs} - val_loss: {np.mean(losses)}')

    accuracy_test = accuracy.eval(feed_dict={X: x_test, t: t_test, training: False})
    test_predictions = y.eval(feed_dict={X: x_test, training: False})
    test_correct_predictions = correct_predictions.eval(
        feed_dict={X: x_test, t: t_test, training: False})

print("Accuracy for the TEST set: " + str(accuracy_test))
# test_rounded_predictions = np.round(test_predictions)
# print(test_rounded_predictions[:10])
# print(t_test[:10])
#
# print(test_correct_predictions[:10])
