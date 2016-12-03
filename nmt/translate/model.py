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

"""
Sequence-to-sequence model with Manning, et. al. attention mechanism 
and full softmax support.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from six.moves import zip

import numpy as np
import os
import tensorflow as tf
import random

from translate.attn_lib import embedding_attention_s2s as s2s
from translate.model_utils import full_sequence_loss as fsl
from translate.model_utils import sampled_sequence_loss as ssl
from translate.model_utils import model_with_buckets

from translate import data_utils


class Model(object):
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers=3,
                 max_gradient_norm=5.0,
                 batch_size=512,
                 learning_rate=None,
                 learning_rate_decay_factor=0.99,
                 optim='adam',
                 use_lstm=True,
                 use_lstm_peepholes=False,
                 use_local=False,
                 num_samples=None,
                 forward_only=False,
                 scope=None,
                 dtype=tf.float32):

        """
        Initialize the multi-bucket graph model.
        """

        # Book-keeping.
        self.src_vocab_size = source_vocab_size
        self.tgt_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size

        # Non-trainable TensorFlow variables.
        learning_rate = self._get_learning_rate(optim) if learning_rate is None else learning_rate
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor
        )
        self.global_step = tf.Variable(0, trainable=False)

        # Output Projection Weights and Biases.
        proj_w_t = tf.Variable(tf.random_normal([target_vocab_size, size]) / tf.sqrt(float(size)),
                               name='proj_w')
        proj_w = tf.transpose(proj_w_t)
        proj_b = tf.Variable(tf.random_normal([target_vocab_size]),
                             name='proj_b')
        output_proj = (proj_w, proj_b)

        # Creating the RNN cells.
        cell = None
        if use_lstm:
            cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=use_lstm_peepholes)
        else:
            cell = tf.nn.rnn_cell.GRUCell(size)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        # Define the Sequence-to-Sequence function with Input Embedding and Attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return s2s(encoder_inputs,
                       decoder_inputs,
                       cell,
                       source_vocab_size,
                       target_vocab_size,
                       size,
                       output_projection=output_proj,
                       feed_previous=do_decode, 
                       use_lstm=use_lstm, 
                       local_p=use_local)

        # Define naming function.
        name_f = lambda name, key: '{0}{1}'.format(name, key)

        # Define the input and weight placeholders.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32,
                                                      shape=[None],
                                                      name=name_f('encoder', i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32,
                                                      shape=[None],
                                                      name=name_f('decoder', i)))
            self.target_weights.append(tf.placeholder(dtype,
                                                      shape=[None],
                                                      name=name_f('weight', i)))

        targets = self.decoder_inputs[1:]

        # Training outputs and losses.
        if forward_only:
            self.outputs, _ = model_with_buckets(self.encoder_inputs,
                                                 self.decoder_inputs,
                                                 lambda x, y: seq2seq_f(x, y, True),
                                                 buckets)
            self.outputs = [tf.pack([tf.argmax(tf.matmul(output, proj_w) + proj_b, 1) 
                                     for output in self.outputs[b]]) 
                            for b in xrange(len(buckets))]
        else:
            loss_f = self._get_loss(num_samples, target_vocab_size, proj_w_t, proj_w, proj_b)
            self.outputs, self.losses = model_with_buckets(self.encoder_inputs,
                                                           self.decoder_inputs,
                                                           lambda x, y: seq2seq_f(x, y, False),
                                                           buckets,
                                                           targets=targets,
                                                           weights=self.target_weights,
                                                           loss_f=loss_f)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables() if scope is None else tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            optimizer = self._get_optimizer(self.learning_rate, optimizer=optim)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_grads, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(optimizer.apply_gradients(zip(clipped_grads, params),
                                                              global_step=self.global_step))

        # Saving Model Parameters Operation.
        trainable_vars = tf.all_variables() if scope is None else tf.get_collection(
            tf.GraphKeys.VARIABLES, scope=scope)
        self.saver = tf.train.Saver(trainable_vars)

    def _get_loss(self, num_samples, target_vocab_size, proj_w_t, proj_w, proj_b):
        """
        Internal method to retrieve loss function.
        """
        if num_samples is not None and (0 < num_samples < target_vocab_size):
            # Sampled Softmax Loss.
            return lambda x, y, z: ssl(x, y, z, proj_w_t, proj_b, target_vocab_size, num_samples)
        else:
            # Full Softmax Loss.
            return lambda x, y, z: fsl(x, y, z, proj_w, proj_b)
        
    def _get_learning_rate(self, optimizer='sgd'):
        return {
            'sgd': 0.5,
            'adam': 0.01,
            'rmsprop': 0.1,
            'adagrad': 0.1
        }[optimizer]

    def _get_optimizer(self, learning_rate, optimizer='sgd'):
        """
        Internal method to retrieve optimizer.
        """
        opts = {
            'sgd': tf.train.GradientDescentOptimizer,
            'adagrad': tf.train.AdagradOptimizer,
            'adam': lambda x: tf.train.AdamOptimizer(learning_rate=x, epsilon=1e-10),
            'rmsprop': lambda x: tf.train.RMSPropOptimizer(learning_rate=x, momentum=0.5)
        }

        if optimizer not in opts:
            raise KeyError('Invalid optimizer. Must be one of sgd, rmsprop or adam.')

        return opts[optimizer](learning_rate)
    
    def save(self, session, checkpoint_dir):
        """
        Saves a snapshot of the current trainable parameters of this model
        to a translate.ckpt file in the <checkpoint_path>.
        """
        checkpoint_path = os.path.join(checkpoint_dir, 'translate.ckpt')
        self.saver.save(session, checkpoint_path, global_step=self.global_step)
        
    def load(self, session, checkpoint_dir):
        """
        Loads a snapshot model from <checkpoint_dir> or creates a new model 
        and initializes its variables.
        """
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
            self.saver.restore(session, checkpoint.model_checkpoint_path)
        else:
            session.run(tf.initialize_all_variables())

    def step(self,
             session,
             encoder_inputs,
             decoder_inputs,
             target_weights,
             bucket_id,
             forward_only=False, 
             delayed=False):
        """
        Run a step of the model given the inputs for a specific bucket.
        """

        k = bucket_id
        encoder_size, decoder_size = self.buckets[k]

        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket, %d != %d."
                             % (len(encoder_inputs), encoder_size))

        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket, %d != %d."
                             % (len(decoder_inputs), decoder_size))

        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket, %d != %d."
                             % (len(target_weights), decoder_size))

        input_feed = {}
        input_feed.update({self.encoder_inputs[i].name: encoder_inputs[i]
                           for i in xrange(encoder_size)})
        input_feed.update({self.decoder_inputs[i].name: decoder_inputs[i]
                           for i in xrange(decoder_size)})
        input_feed.update({self.target_weights[i].name: target_weights[i]
                           for i in xrange(decoder_size)})

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = self.outputs[k] if forward_only else [self.losses[k], self.updates[k]]

        return session.run(output_feed, input_feed) if not delayed else (output_feed, input_feed)

    def get_batch(self, data, bucket_id):
        """
        Get a batch from a specific bucket <bucket_id> in the dataset.
        """

        # Initializations.
        enc_size, dec_size = self.buckets[bucket_id]
        batch_enc = np.zeros((enc_size, self.batch_size), dtype=np.int32)
        batch_dec = np.zeros((dec_size, self.batch_size), dtype=np.int32)
        batch_weights = np.ones((dec_size, self.batch_size), dtype=np.float32)
        batch_dec[0, :] = data_utils._GO_ID

        # Gather Batch Data.
        for i in xrange(self.batch_size):
            enc_data, dec_data = random.choice(data[bucket_id])
            batch_enc[-len(enc_data):, i] = list(reversed(enc_data))
            batch_dec[1:len(dec_data) + 1, i] = dec_data

        # Set the Batch Weights.
        batch_weights[:-1, :] *= batch_dec[1:, :] > 0
        batch_weights[-1, :] = 0.0

        return batch_enc, batch_dec, batch_weights
