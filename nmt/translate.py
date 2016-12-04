#################################################################################
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.                   #
# Modified by Pragaash Ponnusamy (2016) under the Apache License.               #
#                                                                               #
# Licensed under the Apache License, Version 2.0 (the "License");               #
# you may not use this file except in compliance with the License.              #
# You may obtain a copy of the License at                                       #
#                                                                               #
#     http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                               #
# Unless required by applicable law or agreed to in writing, software           #
# distributed under the License is distributed on an "AS IS" BASIS,             #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.      #
# See the License for the specific language governing permissions and           #
# limitations under the License.                                                #
#################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os
import sys
import random
from bisect import bisect_right as br
from time import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
from tensorflow.python.ops import variable_scope as vs

from translate.model import Model

tf.app.flags.DEFINE_string('data_dir', '/tmp', 'Data directory.')
tf.app.flags.DEFINE_string('output_dir', '/tmp', 'Output directory.')
tf.app.flags.DEFINE_string('train_dir', '/tmp', 'Training directory.')
tf.app.flags.DEFINE_boolean('dual_train', True, 'Set true to train two models at unison.')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 200, 'Steps per checkpoint.')

# Global Constants.
_EN_DATA = 'giga-fren.release2.fixed.ids50000.en'
_FR_DATA = 'giga-fren.release2.fixed.ids50000.fr'
_EN_VOCAB = 50000
_FR_VOCAB = 50000
_FLAGS = tf.app.flags.FLAGS
_CONFIG = tf.ConfigProto(allow_soft_placement=True)
_GPU = map(lambda x: x.name, filter(lambda d: d.device_type == 'GPU', list_local_devices()))
_BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]
_MAX_EN = 40
_MAX_FR = 50


def get_int_seq(line):
    """
    Convert a string of space-separated values (SSV) into a list
    of integers.
    """
    return map(int, line.strip().split())


def read_into_buckets(en_ids_path, fr_ids_path, print_every=100000):
    """
    Generate dataset by reading data into buckets.
    """
    data_set = map(lambda x: [], _BUCKETS)
    counter = 0
    en_buckets, fr_buckets = zip(*_BUCKETS)
    with open(en_ids_path, 'r') as en:
        with open(fr_ids_path, 'r') as fr:
            en_line, fr_line = en.readline(), fr.readline()
            while en_line and fr_line:
                counter += 1
                if counter % print_every == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                en_sequence, fr_sequence = get_int_seq(en_line), get_int_seq(fr_line)
                en_seq_len, fr_seq_len = len(en_sequence), len(fr_sequence)
                if en_seq_len < _MAX_EN and fr_seq_len < _MAX_FR:
                    b_id = max(br(en_buckets, en_seq_len), br(fr_buckets, fr_seq_len))
                    data_set[b_id].append([en_sequence, fr_sequence])
                en_line, fr_line = en.readline(), fr.readline()
    return data_set


def get_data(binary_path=None, en_ids_path=None, fr_ids_path=None):
    """
    Load data from binary if exists, otherwise call subroutine to read tokenized
    data into buckets.
    """
    data_set = None
    if binary_path is not None and os.path.exists(binary_path):
        with open(binary_path, 'rb') as binary:
            data_set = pickle.load(binary)
    else:
        assert en_ids_path is not None and fr_ids_path is not None, 'Tokenized File Not Found!'
        data_set = read_into_buckets(en_ids_path, fr_ids_path)
    return data_set


def get_weights(data):
    std = np.sqrt(np.asarray(map(len, data)))
    return std.cumsum() / std.sum()


def linebreak():
    return '-' * 50 + '\n'


def train():
    with open(os.path.join(_FLAGS.output_dir, 'train.out'), 'w') as f:
        # Graph Creation.
        graph = tf.Graph()
        t = time()
        with graph.as_default():
            with tf.device(_GPU[0]), vs.variable_scope('graph_one') as gscope_1:
                model_one = Model(source_vocab_size=_EN_VOCAB,
                                  target_vocab_size=_FR_VOCAB,
                                  buckets=_BUCKETS,
                                  size=512,
                                  num_layers=3,
                                  learning_rate=None,
                                  batch_size=256,
                                  use_lstm=True,
                                  use_local=True,
                                  optim='adam',
                                  scope=gscope_1.name,
                                  num_samples=None)
            with tf.device(_GPU[1]), vs.variable_scope('graph_two') as gscope_2:
                model_two = Model(source_vocab_size=_EN_VOCAB,
                                  target_vocab_size=_FR_VOCAB,
                                  buckets=_BUCKETS,
                                  size=512,
                                  num_layers=3,
                                  learning_rate=None,
                                  batch_size=256,
                                  use_lstm=True,
                                  use_local=True,
                                  optim='sgd',
                                  scope=gscope_2.name,
                                  num_samples=None)
        print('Initializing Graphs took %.3f s' % (time() - t))
        print(linebreak())
        sys.stdout.flush()

        with tf.Session(graph=graph, config=_CONFIG) as sess:

            # Initializations.
            t = time()
            m1_path = os.path.join(_FLAGS.train_dir, 'model_one')
            m2_path = os.path.join(_FLAGS.train_dir, 'model_two')
            m1_checkpoint = tf.train.get_checkpoint_state(m1_path)
            m2_checkpoint = tf.train.get_checkpoint_state(m2_path)
            conditions = [bool(m1_checkpoint), bool(m2_checkpoint)]
            if all(conditions):
                conditions.append(tf.gfile.Exists(m1_checkpoint.model_checkpoint_path))
                conditions.append(tf.gfile.Exists(m2_checkpoint.model_checkpoint_path))
            if all(conditions):
                model_one.saver.restore(sess, m1_checkpoint.model_checkpoint_path)
                model_two.saver.restore(sess, m2_checkpoint.model_checkpoint_path)
            else:
                sess.run(tf.initialize_all_variables())
            print('Initializing Variables took %.3f s' % (time() - t))
            print(linebreak())
            sys.stdout.flush()
            
            # Gather Data.
            dataset = get_data(en_ids_path=os.path.join(_FLAGS.data_dir, _EN_DATA),
                       fr_ids_path=os.path.join(_FLAGS.data_dir, _FR_DATA))
            intervals = get_weights(dataset)

            # Book-keeping.
            step_time = 0.0
            batch_loss = [0.0, 0.0]
            current_step = 0
            previous_losses = [[], []]

            # Training.
            while True:
                bucket_id = np.abs(np.random.rand() - intervals).argmin()
                start_time = time()
                encoder_inputs, decoder_inputs, target_weights = model_one.get_batch(dataset, bucket_id)

                of_one, if_one = model_one.step(sess,
                                                encoder_inputs,
                                                decoder_inputs,
                                                target_weights,
                                                bucket_id,
                                                forward_only=False,
                                                delayed=True)
                of_two, if_two = model_two.step(sess,
                                                encoder_inputs,
                                                decoder_inputs,
                                                target_weights,
                                                bucket_id,
                                                forward_only=False,
                                                delayed=True)
                output_feed = of_one + of_two

                input_feed = {}
                input_feed.update(if_one)
                input_feed.update(if_two)

                loss_one, _, loss_two, _ = sess.run(output_feed, input_feed)

                step_time += (time() - start_time) / _FLAGS.steps_per_checkpoint
                batch_loss[0] += loss_one / _FLAGS.steps_per_checkpoint
                batch_loss[1] += loss_two / _FLAGS.steps_per_checkpoint
                current_step += 1

                if current_step % _FLAGS.steps_per_checkpoint == 0:
                    perplexity_one = np.exp(loss_one)
                    perplexity_two = np.exp(loss_two)
                    f.write('%f\t%f\n' % (perplexity_one, perplexity_two))

                    if current_step > 2 * _FLAGS.steps_per_checkpoint:
                        if batch_loss[0] > max(previous_losses[0][-3:]):
                            sess.run(model_one.learning_rate_decay_op)
                        if batch_loss[1] > max(previous_losses[1][-3:]):
                            sess.run(model_two.learning_rate_decay_op)

                    previous_losses[0].append(batch_loss[0])
                    previous_losses[1].append(batch_loss[1])
                    
                    print('current-step: %d step-time: %.3f' %(current_step, step_time))

                    step_time = 0.0
                    batch_loss = [0.0, 0.0]

                    model_one.save(sess, m1_path)
                    model_two.save(sess, m2_path)
                    sys.stdout.flush()

def train_one():
    dataset = get_data(en_ids_path=os.path.join(_FLAGS.data_dir, _EN_DATA),
                       fr_ids_path=os.path.join(_FLAGS.data_dir, _FR_DATA))
    intervals = get_weights(dataset)

    step_time = 0.0
    batch_loss = 0.0
    current_step = 0
    previous_losses = []

    with open(os.path.join(_FLAGS.output_dir, 'train_one.out'), 'w') as f:
        graph = tf.Graph()
        t = time()
        with graph.as_default():
            model = Model(source_vocab_size=_EN_VOCAB,
                          target_vocab_size=_FR_VOCAB,
                          buckets=_BUCKETS,
                          size=512,
                          num_layers=3,
                          learning_rate=None,
                          batch_size=256,
                          use_lstm=True,
                          use_local=True,
                          optim='adam',
                          scope=gscope_1.name,
                          num_samples=None)
        print('Initializing Graphs took %.3f s\n' % (time() - t))
        print(linebreak())
        sys.stdout.flush()

        with tf.Session(graph=graph, config=_CONFIG) as sess:

            # Initializations.
            t = time()
            save_path = os.path.join(_FLAGS.train_dir, 'model')
            model.load(sess, save_path)
            print('Initializing Variables took %.3f s\n' % (time() - t))
            print(linebreak())
            sys.stdout.flush()

            # Graph Creation.
            while True:
                bucket_id = np.abs(np.random.rand() - intervals).argmin()
                start_time = time()
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(dataset, bucket_id)
                loss, _ = model.step(sess,
                                     encoder_inputs,
                                     decoder_inputs,
                                     target_weights,
                                     bucket_id,
                                     forward_only=False)
                step_time += (time() - start_time) / _FLAGS.steps_per_checkpoint
                batch_loss += loss / _FLAGS.steps_per_checkpoint
                current_step += 1

                if current_step % _FLAGS.steps_per_checkpoint == 0:
                    perplexity = np.exp(loss)
                    f.write('%f\t%f\n' % perplexity)

                    if current_step > 2 * _FLAGS.steps_per_checkpoint:
                        if batch_loss > max(previous_losses[-3:]):
                            sess.run(model.learning_rate_decay_op)

                    previous_losses.append(batch_loss)
                    
                    print('current-step: %d step-time: %.3f' %(current_step, step_time))

                    step_time = 0.0
                    batch_loss = 0.0

                    model.save(sess, save_path)
                    sys.stdout.flush()

def self_test_model():
    """
    Test the translation model.
    """

    print("Self-test for neural translation model.")
    linebreak()

    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/cpu:0'):
            t = time()
            # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
            model = Model(source_vocab_size=10,
                          target_vocab_size=10,
                          buckets=[(3, 3), (6, 6)],
                          size=32,
                          num_layers=2,
                          learning_rate=None,
                          max_gradient_norm=5.0,
                          batch_size=32,
                          use_lstm=True,
                          optim='adam',
                          num_samples=None)

            print("Initializing Model took %.6fs" % (time() - t))
            linebreak()

    with tf.Session(graph=graph) as sess:
        t = time()
        sess.run(tf.initialize_all_variables())
        print("Initializing Variables took %.6fs" % (time() - t))
        linebreak()

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        num_iter = 20

        print('Using Learning Rate: %.2f' % (model.learning_rate.eval()))
        linebreak()

        t = time()
        # Train the fake model for 5 steps.
        for _ in xrange(num_iter):
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
            loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            print('Perplexity: %f' % (np.exp(loss)))
        linebreak()
        print("Average training time: %.6fs/iter" % ((time() - t) / num_iter))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
