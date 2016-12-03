from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from six.moves import zip

from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops as em_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs


def _get_embedder(name, shape, inputs):
    embedder = tf.get_variable(name, initializer=tf.random_normal(shape))
    return embedder, [em_ops.embedding_lookup(embedder, i) for i in inputs]


def _get_weights(name, shape):
    return tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.1))


def _get_hs(state, use_lstm=False):
    return state[-1][1] if use_lstm else state[-1]


def embedding_attention_s2s(encoder_inputs,
                            decoder_inputs,
                            cell,
                            num_encoder_symbols,
                            num_decoder_symbols,
                            size,
                            output_projection=None,
                            feed_previous=False,
                            use_lstm=False,
                            local_p=False,
                            dtype=tf.float32):
    """
    Embedding with Manning, et. al. global attention model.
    """
    reuse_f = lambda k: True if k > 0 else None
    S = len(encoder_inputs)
    
    with vs.variable_scope('embedding_attention_s2s') as outer_scope:
        with vs.variable_scope('encoder') as enc_scope:
            embedder_en, emb_en_input = _get_embedder('embedding_en', 
                                                      [num_encoder_symbols, size], 
                                                      encoder_inputs)
            
            encoder_states, prev_state = [], None
            for i, enc_in in enumerate(emb_en_input):
                with vs.variable_scope(enc_scope, reuse=reuse_f(i)):
                    _, state = tf.nn.rnn(cell, [enc_in], initial_state=prev_state, dtype=dtype)
                    prev_state = state
                    encoder_states.append(state)
            
            h_s_bar = tf.transpose(tf.pack([_get_hs(es, use_lstm) for es in encoder_states]), 
                                   perm=[1, 0, 2])
            
        with vs.variable_scope('decoder') as dec_scope:
            embedding_de, emb_de_input = _get_embedder('embedding_de',
                                                       [num_decoder_symbols, size],
                                                       decoder_inputs)
            
            W_a = _get_weights('W_a', [size, size])
            W_c = _get_weights('W_c', [2*size, size])
            
            if local_p:
                W_p = _get_weights('W_p', [size, size])
                v_p = _get_weights('v_p', [size, 1])
            
            state = encoder_states[-1]
            partial_scores = [tf.reshape(tf.matmul(_get_hs(es, use_lstm), W_a), [-1, 1, size])
                              for es in encoder_states]
            
            output, outputs = None, []
            
            for i, dec_in in enumerate(emb_de_input):
                with vs.variable_scope(dec_scope, reuse=reuse_f(i)):
                    
                    # Loop function at test time
                    if feed_previous and output is not None:
                        w, b = output_projection
                        prev = nn_ops.xw_plus_b(output, w, b)
                        prev_symbol = math_ops.argmax(prev, 1)
                        dec_in = em_ops.embedding_lookup(embedding_de, prev_symbol)
                        
                    _, state = cell(dec_in, state)
                    h_t = _get_hs(state, use_lstm)
                    batch_h_t = tf.reshape(h_t, [-1, size, 1])
                    
                    if local_p:
                        align = tf.matmul(v_p, tf.tanh(tf.matmul(W_p, h_t, transpose_b=True)), 
                                          transpose_a=True)
                        p_t = S * tf.sigmoid(align)
                        scale = tf.concat(0, [tf.exp(-4.5 * tf.square((p_t - s)/S)) for s in xrange(S)])
                    
                    scores = tf.nn.softmax(tf.pack([tf.reshape(tf.batch_matmul(ps, batch_h_t), [-1]) 
                                                    for ps in partial_scores]), dim=0)
                    
                    if local_p:
                        scores *= scale
                        scores /= tf.reduce_sum(scores, 0)
                    
                    scores = tf.reshape(scores, [-1, 1, S])
                    
                    c_t = tf.reshape(tf.batch_matmul(scores, h_s_bar), [-1, size])
                    h_bar_t = tf.concat(1, [c_t, _get_hs(state, use_lstm)])
                    output = tf.tanh(tf.matmul(h_bar_t, W_c))
                    outputs.append(output)
                    
        return outputs, state