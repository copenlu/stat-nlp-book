"""
         __  _ __
  __  __/ /_(_) /
 / / / / __/ / /
/ /_/ / /_/ / /
\__,_/\__/_/_/ v0.2
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.python.ops import variable_scope as vs
import os

def get_by_index(tensor, index):
    """
    :param tensor: [dim1 x dim2 x dim3] tensor
    :param index: [dim1] tensor of indices for dim2
    :return: [dim1 x dim3] tensor
    """
    dim1, dim2, dim3 = tf.unpack(tf.shape(tensor))
    flat_index = tf.range(0, dim1) * dim2 + (index - 1)
    return tf.gather(tf.reshape(tensor, [-1, dim3]), flat_index)


def get_last(tensor):
    """
    :param tensor: [dim1 x dim2 x dim3] tensor
    :return: [dim1 x dim3] tensor
    """
    shape = tf.shape(tensor)  # [dim1, dim2, dim3]
    slice_size = shape * [1, 0, 1] + [0, 1, 0]  # [dim1, 1 , dim3]
    slice_begin = shape * [0, 1, 0] + [0, -1, 0]  # [1, dim2-1, 1]
    return tf.squeeze(tf.slice(tensor, slice_begin, slice_size), [1])


def mask_for_lengths(lengths, batch_size=None, max_length=None, mask_right=True,
                     value=-1000.0):
    """
    Creates a [batch_size x max_length] mask.
    :param lengths: int64 1-dim tensor of batch_size lengths
    :param batch_size: int32 0-dim tensor or python int
    :param max_length: int32 0-dim tensor or python int
    :param mask_right: if True, everything before "lengths" becomes zero and the
        rest "value", else vice versa
    :param value: value for the mask
    :return: [batch_size x max_length] mask of zeros and "value"s
    """
    if max_length is None:
        max_length = tf.cast(tf.reduce_max(lengths), tf.int32)
    if batch_size is None:
        batch_size = tf.shape(lengths)[0]
    # [batch_size x max_length]
    mask = tf.reshape(tf.tile(tf.range(0, max_length), [batch_size]), tf.pack([batch_size, -1]))
    if mask_right:
        mask = tf.greater_equal(tf.cast(mask, tf.int64), tf.expand_dims(lengths, 1))
    else:
        mask = tf.less(tf.cast(mask, tf.int64), tf.expand_dims(lengths, 1))
    mask = tf.cast(mask, tf.float32) * value
    return mask


def tfrun(tensor):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        return sess.run(tensor)


def tfrunprint(tensor, suffix="", prefix=""):
    if prefix == "":
        print(tfrun(tensor), suffix)
    else:
        print(prefix, tfrun(tensor), suffix)


def tfrunprintshape(tensor, suffix="", prefix=""):
    tfrunprint(tf.shape(tensor), suffix, prefix)


def tfprint(tensor, fun=None, prefix=""):
    if fun is None:
        fun = lambda x: x
    return tf.Print(tensor, [fun(tensor)], prefix)


def tfprints(tensors, fun=None, prefix=""):
    if fun is None:
        fun = lambda x: x
    prints = []
    for i in range(0, len(tensors)):
        prints.append(tf.Print(tensors[i], [fun(tensors[i])], prefix))
    return prints


def tfprintshapes(tensors, prefix=""):
    return tfprints(tensors, lambda x: tf.shape(x), prefix)


def tfprintshape(tensor, prefix=""):
    return tfprint(tensor, lambda x: tf.shape(x), prefix)


def gather_in_dim(params, indices, dim, name=None):
    """
    Gathers slices in a defined dimension. If dim == 0 this is doing the same
      thing as tf.gather.
    """
    if dim == 0:
        return tf.gather(params, indices, name)
    else:
        dims = [i for i in range(0, len(params.get_shape()))]
        to_dims = list(dims)
        to_dims[0] = dim
        to_dims[dim] = 0

        transposed = tf.transpose(params, to_dims)
        gathered = tf.gather(transposed, indices)
        reverted = tf.transpose(gathered, to_dims)

        return reverted


def unit_length(tensor):
    l2norm_sq = tf.reduce_sum(tensor * tensor, 1, keep_dims=True)
    l2norm = tf.rsqrt(l2norm_sq)
    return tensor * l2norm



class BatchBucketSampler:
    """
        Samples batches from a list of data points
    """
    def __init__(self, data, batch_size=1, buckets=None, test=False):
        """
        :param data: a list of higher order tensors where the first dimension
        corresponds to the number of examples which needs to be the same for
        all tensors
        :param batch_size: desired batch size
        :param buckets: a list of bucket boundaries
        :return:
        """
        self.data = data
        self.num_examples = len(self.data[0])
        self.batch_size = batch_size
        self.buckets = buckets
        self.to_sample = list(range(0, self.num_examples))
        if test==False:
            np.random.shuffle(self.to_sample)
        self.counter = 0

    def __reset(self):
        self.to_sample = list(range(0, self.num_examples))
        np.random.shuffle(self.to_sample)
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_examples - self.counter <= self.batch_size:
            self.__reset()
            raise StopIteration
        return self.get_batch(self.batch_size)

    def get_batch(self, batch_size):
        if self.num_examples == self.counter:
            self.__reset()
            return self.get_batch(batch_size)
        else:
            num_to_sample = batch_size
            batch_indices = []
            if len(self.to_sample) < num_to_sample:
                batch_indices += self.to_sample
                num_to_sample -= len(self.to_sample)
                self.__reset()
            self.counter += batch_size
            batch_indices += self.to_sample[0:num_to_sample]
            self.to_sample = self.to_sample[num_to_sample:]
            return [x[batch_indices] for x in self.data]


LOSS_TRACE_TAG = "Loss"

class Hook(object):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, sess, epoch, iteration, model, loss):
        raise NotImplementedError


class TraceHook(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer

    def __call__(self, sess, epoch, iteration, model, loss):
        raise NotImplementedError

    def update_summary(self, sess, current_step, title, value):
        cur_summary = tf.scalar_summary(title, value)
        merged_summary_op = tf.merge_summary([cur_summary])  # if you are using some summaries, merge them
        summary_str = sess.run(merged_summary_op)
        self.summary_writer.add_summary(summary_str, current_step)


class LossHook(TraceHook):
    def __init__(self, summary_writer, iteration_interval):
        super().__init__(summary_writer)
        self.iteration_interval = iteration_interval
        self.acc_loss = 0

    def __call__(self, sess, epoch, iteration, model, loss):
        self.acc_loss += loss
        if not iteration == 0 and iteration % self.iteration_interval == 0:
            loss = self.acc_loss / self.iteration_interval
            print("Epoch " + str(epoch) +
                  "\tIter " + str(iteration) +
                  "\tLoss " + str(loss))
            self.update_summary(sess, iteration, LOSS_TRACE_TAG, loss)
            self.acc_loss = 0



class Trainer(object):
    """
    Object representing a TensorFlow trainer.
    """

    def __init__(self, optimizer, max_epochs):
        self.loss = None
        self.optimizer = optimizer
        self.max_epochs = max_epochs

    def __call__(self, batcher, placeholders, loss, session=None):
        self.loss = loss
        minimization_op = self.optimizer.minimize(loss)

        if session is None:
            session = tf.Session()

        init = tf.initialize_all_variables()
        session.run(init)
        epoch = 1
        while epoch < self.max_epochs:
            iteration = 1
            total_loss = 0
            total = 0
            for values in batcher:
                iteration += 1
                total += len(values[-1])
                feed_dict = {}
                for i in range(0, len(placeholders)):
                    feed_dict[placeholders[i]] = values[i]
                _, current_loss = session.run([minimization_op, loss], feed_dict=feed_dict)
                total_loss += current_loss

            print("Epoch ", str(epoch), "\tLoss ", total_loss)
            epoch += 1

        return session


    def test(self, batcher, placeholders, model=None, session=None):
        """
        Test using a trained Tensorflow model
        """

        predicted_all = []
        total = 0
        for values in batcher:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                feed_dict[placeholders[i]] = values[i]
            scores = session.run(model, feed_dict=feed_dict)
            predicted_all.extend(scores)

        session.close()
        return predicted_all


def load_model(sess, path, modelname):
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, os.path.join(path, modelname))