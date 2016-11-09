from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow as tf
import pprint

def embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, source_vocab_size, target_vocab_size, size, output_projection, feed_previous=False):
  """Embedding RNN sequence-to-sequence model. """
  with variable_scope.variable_scope("seq2seq") as scope:
    dtype = tf.float32
    
    # Encoder
    with variable_scope.variable_scope("encoding") as scope:
      embedding_en = tf.get_variable("embedding_en", initializer=tf.truncated_normal([source_vocab_size, size], stddev=1))
      emb_enc_input = [embedding_ops.embedding_lookup(embedding_en, i) for i in encoder_inputs]
      for i, inp in enumerate(emb_enc_input):
        if i > 0:
          scope.reuse_variables()
          _, encoder_state = tf.nn.rnn(cell, [inp], initial_state=encoder_state)
        else:
          _, encoder_state = tf.nn.rnn(cell, [inp], dtype=dtype)

    # Decoder
    with variable_scope.variable_scope("decoder") as scope2:

      embedding_de = tf.get_variable("embedding_de", initializer=tf.truncated_normal([target_vocab_size, size], stddev=1))
      emb_dec_input = [embedding_ops.embedding_lookup(embedding_de, i) for i in decoder_inputs]
      state = encoder_state
      outputs = []
      output = None
      for i, inp in enumerate(emb_dec_input):
        if feed_previous and output is not None:
          prev = nn_ops.xw_plus_b(output, output_projection[0], output_projection[1])
          prev_symbol = math_ops.argmax(prev, 1)           
          inp = embedding_ops.embedding_lookup(embedding_de, prev_symbol)
        if i > 0:
          variable_scope.get_variable_scope().reuse_variables()
        output, state = cell(inp, state)
        outputs.append(output)

    return outputs, state

def attn_seq2seq(encoder_inputs, decoder_inputs, cell, source_vocab_size, target_vocab_size, size, output_projection, feed_previous=False):
  """Embedding RNN sequence-to-sequence model. """
  with variable_scope.variable_scope("seq2seq") as scope:
    dtype = tf.float32
    
    # Encoder.
    with variable_scope.variable_scope("encoding") as scope:
      embedding_en = tf.get_variable("embedding_en", initializer=tf.random_normal([source_vocab_size, size], stddev=1.0))
      emb_enc_input = [embedding_ops.embedding_lookup(embedding_en, i) for i in encoder_inputs]
      encoder_states = []
      for i, inp in enumerate(emb_enc_input):
        if i > 0:
          scope.reuse_variables()
          _, state = tf.nn.rnn(cell, [inp], initial_state=encoder_states[-1])
        else:
          _, state = tf.nn.rnn(cell, [inp], dtype=dtype)
        encoder_states.append(state)

    full_encodes = tf.transpose(tf.pack([ec[-1][1] for ec in encoder_states]), perm=[1, 0, 2])
    # Decoder
    with variable_scope.variable_scope("decoder") as scope2:

      embedding_de = tf.get_variable("embedding_de", initializer=tf.random_normal([target_vocab_size, size], stddev=1.0))
      W_a = tf.get_variable("w_a", initializer=tf.random_normal([size, size], stddev=0.1))
      W_c = tf.get_variable("w_c", initializer=tf.random_normal([2*size, size], stddev=0.1))

      emb_dec_input = [embedding_ops.embedding_lookup(embedding_de, i) for i in decoder_inputs]
      state = encoder_states[-1]

      outputs = []
      output = None
      precomps = []
      for ec in encoder_states:
        h_s = ec[-1][1]
        precomp = tf.reshape(tf.matmul(h_s, W_a), [-1, 1, size])
        precomps.append(precomp)

      for t, inp in enumerate(emb_dec_input):
        if feed_previous and output is not None: # Loop function at test time
          prev = nn_ops.xw_plus_b(output, output_projection[0], output_projection[1])
          prev_symbol = math_ops.argmax(prev, 1)
          inp = embedding_ops.embedding_lookup(embedding_de, prev_symbol)
        if t > 0:
          variable_scope.get_variable_scope().reuse_variables()
        _, state = cell(inp, state)
        orig_h_t = state[-1][1]
        h_t = tf.reshape(orig_h_t, [-1, size, 1])
        scores = []; c_ts = []
        for pc in precomps:
          scores.append(tf.reshape(tf.batch_matmul(pc, h_t), [-1]))

        scores = tf.nn.softmax(tf.transpose(tf.pack(scores)))
        scores = tf.transpose(scores)

        scores = tf.reshape(scores, [-1, 1, len(encoder_inputs)])

        c_t = tf.reshape(tf.batch_matmul(scores, full_encodes), [-1, size])

        h_bar_t = tf.concat(1, [c_t, state[-1][1]])
        output = tf.tanh(tf.matmul(h_bar_t, W_c))
        outputs.append(output)

    return outputs, state