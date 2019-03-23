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

LEARNING_RATE = 0.05

n_epochs = 200
batch_size = 256
n_neurons_per_layer = [512, 256, 128]

# Defining placeholder for our data and labels.
X = tf.placeholder(dtype=tf.float32, shape=(None, INPUTS), name="X")
t = tf.placeholder(dtype=tf.float32, shape=(None, OUTPUTS), name="t")

training = tf.placeholder(tf.bool, name='training')
learning_rate = tf.placeholder(tf.float32, name='lr')


# 8 attributes
n_inputs = INPUTS
# Define neurons per layer
n_hidden1 = 3
n_hidden2 = 6
n_hidden3 = 3
# 10 different classes
n_outputs = OUTPUTS

# Define deep neural network 
with tf.name_scope("dnn"):
    # We are goingt o use HE INITIALIZATION
    he_init = tf.contrib.layers.variance_scaling_initializer()
    # Utilizamos el método de Tensorflow tf.layers.dense ==> crea una red totalmemte conectada 
    hidden1 = tf.layers.dense(X, n_hidden1,kernel_initializer=he_init, activation=tf.nn.elu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.elu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.elu)
    # Logits es el output de la red neuronal ANTES de pasar por la softmax activation function.
    logits = tf.layers.dense(hidden3, OUTPUTS, name="outputs")
    logits_summary = tf.summary.scalar('logits', logits)

# Define our loss/cost function
with tf.name_scope("cost"):
    # This function is equivalent to applyiong the softmaz activation function and then computing the cross entropy, but it is more efficent, and it properly teakes care of corner cases:
    ## When logits are large, floating-point rounding error may cause the softmax output to be exactly equal to 0 or 1, and in this case
    ## the cross entropy equationwould contain a log(0) term, equal to negative infinity.
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t, logits=logits)
    loss = tf.reduce_mean(xentropy, name="cost")
    cost_summary = tf.summary.scalar('log_loss', loss)

# Define optimizer
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_step = optimizer.minimize(loss)

# Evaluation of the model
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, t, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    correct_summary = tf.summary.scalar('correct', tf.cast(correct, tf.int32))

# Create and initial node and save
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# TENSORBOARD CONFIGURATION
from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

# # regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
# hidden_layers = [tf.layers.dense(X, n_neurons_per_layer[0], activation=tf.nn.relu)]
# for layer in range(1, len(n_neurons_per_layer)):
#     hidden_layers.append(
#         tf.layers.dropout(
#             hidden_layers[-1],
#             rate=0.4 if layer < 2 else 0.2,
#             noise_shape=None,
#             seed=None,
#             training=training,
#             name=f'dropout_{layer-1}'
#         )
#     )
#     hidden_layers.append(
#         tf.layers.dense(
#             hidden_layers[-1], n_neurons_per_layer[layer],
#             activation=tf.nn.relu,
#             # kernel_regularizer=regularizer,
#             name=f'dense_{layer}'
#         )
#     )
# net_out = tf.layers.dense(hidden_layers[len(n_neurons_per_layer) - 1], OUTPUTS)
# y = tf.nn.softmax(logits=net_out, name="y")


# for layer in hidden_layers:
#     print(layer)

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=net_out)

# # loss definition
# loss = tf.reduce_mean(cross_entropy, name="cost")
# # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# # loss += tf.add_n(reg_losses)

# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


# init = tf.global_variables_initializer()
# accuracy_train_history = []
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
                                learning_rate: 0.01 if epoch <= 100 else 0.001 if epoch < 150 else 0.0001,
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
