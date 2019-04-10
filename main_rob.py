import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
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

n_epochs = 500
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

activation = tf.nn.elu
# Define deep neural network 
with tf.name_scope("dnn"):
    # We are goingt o use HE INITIALIZATION
    he_init = tf.contrib.layers.variance_scaling_initializer()
    # Utilizamos el método de Tensorflow tf.layers.dense ==> crea una red totalmemte conectada 
    hidden1 = tf.layers.dense(X, n_hidden1,kernel_initializer=he_init, activation=activation, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=activation)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=activation)
    # Logits es el output de la red neuronal ANTES de pasar por la softmax activation function.
    logits = tf.layers.dense(hidden3, OUTPUTS, name="outputs")
    logits_summary = tf.summary.scalar('logits', logits)

# Define our loss/cost function
with tf.name_scope("cost"):
    # This function is equivalent to applyiong the softmaz activation function and then computing the cross entropy, but it is more efficent, and it properly teakes care of corner cases:
    ## When logits are large, floating-point rounding error may cause the softmax output to be exactly equal to 0 or 1, and in this case
    ## the cross entropy equationwould contain a log(0) term, equal to negative infinity.
    y = tf.nn.softmax(logits=logits, name="y")
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="cost")
    loss_summary = tf.summary.scalar('log_loss', loss)

# Define optimizer
with tf.name_scope("train"):
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               decay_steps, decay_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate) --> 0.3537182
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9) --> 0.34931508
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True) #-->  0.38062623
    # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate) --> 0.31066537
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=0.9, decay=0.9, epsilon=1e-10) --> 0.099804305
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) --> 0.35567516

    train_step = optimizer.minimize(loss, global_step=global_step)

# Evaluation of the model
with tf.name_scope("eval"):
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    correct_summary = tf.summary.scalar('correct', tf.cast(correct_predictions, tf.float32))

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

checkpoint_path = "/tmp/DNN_profiles.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "/Users/robertollopcardenal/Desktop/developer/python/ml/deeplearning/assignment-1-Median house/tensorflow-ml/models/DNN_profiles.ckpt"

best_loss = np.infty
#print(type(best_loss))
epochs_without_progress = 0
max_epochs_without_progress = 5

minibatch_size = 256
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1, n_epochs + 1):
        for j in range(len(x_train) // minibatch_size):
            sess.run(train_step,
                     feed_dict={X: x_train[j * minibatch_size:(j + 1) * minibatch_size],
                                t: t_train[j * minibatch_size:(j + 1) * minibatch_size]
                                })
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: x_dev, t: t_dev})

        if epoch % 10 == 0:
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                  "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            # Si el error es menor que el anterior seguimos entrenando
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                # Sumamos de 5 en 5, porque estamos dentro del IF 
                epochs_without_progress += 5
                # Si después de m"ax_epochs_without_progress", el error no ha disminuido "OVERFITTING" ==> EARLY STOPPING
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

os.remove(checkpoint_epoch_path)

with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    # accuracy_val = accuracy.eval(feed_dict={X: x_test, t: t_test})
    # #correct_val = sess.run([correct], feed_dict={X: X_test, y: y_test})
    # correct_val = correct_predictions.eval(feed_dict={X: x_test, t: t_test})
    # print(correct_val)
    # print(len(correct_val))
    # correct_val = np.array([correct_val]).T
    # y_test_aux = np.array([t_test]).T

    accuracy_test = accuracy.eval(feed_dict={X: x_test, t: t_test})
    test_predictions = y.eval(feed_dict={X: x_test})
    test_correct_predictions = correct_predictions.eval(
        feed_dict={X: x_test, t: t_test})


print("Accuracy for the TEST set: " + str(accuracy_test))