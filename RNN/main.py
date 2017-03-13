# Working example for my blog post at:
# http://danijar.com/variable-sequence-lengths-in-tensorflow/
import functools
import tensorflow as tf
import os

from .data import prepare_test_data, prepare_train_data


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class VariableSequenceClassification:

    CHECKPOINT_FILE_NAME = 'CheckPoints/Training_Checkpoint'

    def __init__(self, data, target, num_hidden=100, num_layers=2):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Recurrent network.
        output, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(self._num_hidden),
            self.data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        last = self._last_relevant(output, self.length)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    def save_progress(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.CHECKPOINT_FILE_NAME)

    def continue_progress(self, sess):
        if os.path.exists(os.path.join(self.CHECKPOINT_FILE_NAME+'.index')):
            saver = tf.train.Saver()
            saver.restore(sess, self.CHECKPOINT_FILE_NAME)


def train_and_test():
    train_input, train_output , max_length_train= prepare_train_data()
    test_input, test_output , max_length_test = prepare_test_data()
    max_length = max([max_length_test, max_length_train])
    num_classes = 2
    data = tf.placeholder(tf.float32, [None, max_length, 1])
    target = tf.placeholder(tf.float32, [None, num_classes])
    model = VariableSequenceClassification(data, target)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 10
    for epoch in range(1000):
        model.continue_progress(sess)
        for _ in range(int(len(train_input)/batch_size)):
            sess.run(model.optimize, {data: train_input[_*batch_size:(_+1)*batch_size], target: train_output[_*batch_size:(_+1)*batch_size]})
        model.save_progress(sess)
        error = sess.run(model.error, {data: test_input, target: test_output})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
