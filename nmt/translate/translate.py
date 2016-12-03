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

import os
import random
import tensorflow as tf
import cPickle as pickle
from time import time
from bisect import bisect_right as br
from translate import data_utils

# Global Constants.
_BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]
_BUCKETS_ALT = [(10, 10), (20, 20), (25, 30), (30, 40), (40, 50)]
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

def self_test_model():
    """
    Test the translation model.
    """
    def linebreak():
        print('-' * 50)
    
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
            
            print("Initializing Model took %.6fs" %(time() - t))
            linebreak()
    
    with tf.Session(graph=graph) as sess:
        
        t = time()
        sess.run(tf.initialize_all_variables())
        print("Initializing Variables took %.6fs" %(time() - t))
        linebreak()

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        num_iter = 20
        
        print('Using Learning Rate: %.2f' %(model.learning_rate.eval()))
        linebreak()
        
        t = time()
        # Train the fake model for 5 steps.
        for _ in xrange(num_iter):
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
            loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            print('Perplexity: %f' %(np.exp(loss)))
        linebreak()
        print("Average training time: %.6fs/iter" %((time() - t)/num_iter))