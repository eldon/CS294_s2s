from __future__ import absolute_import, division, print_function
from tensorflow.python.ops import variable_scope
import myseq2seq, random, numpy as np, tensorflow as tf
from sequence_loss import *


class Seq2SeqModel(object):
  def __init__(self,source_vocab_size,target_vocab_size,buckets,size,num_layers,max_gradient_norm,learning_rate,num_samples=256,dtype=tf.float32):
    """source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      learning_rate: learning rate to start with.
      num_samples: number of samples for sampled softmax.
      dtype: the data type to use to store internal variables.
    """

    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype) # learning_rate
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.99) # This is a function we run when needed

    self.global_step = tf.Variable(0, trainable=False) # global_step

    proj_w_t = tf.Variable(tf.truncated_normal([target_vocab_size, size], stddev=0.1), name='proj_w')
    proj_w = tf.transpose(proj_w_t)
    proj_b = tf.Variable(tf.truncated_normal([target_vocab_size], stddev=0.1), name='proj_b')

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

    def seq2seq_f(encoder_inputs, decoder_inputs, feed_previous):
      return myseq2seq.attn_seq2seq(encoder_inputs, decoder_inputs, cell, source_vocab_size, target_vocab_size, size, output_projection=(proj_w, proj_b), feed_previous=feed_previous)
      # return myseq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, source_vocab_size, target_vocab_size, size, output_projection=(proj_w, proj_b), feed_previous=feed_previous)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []

    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    self.outputs = []; self.full_outputs = []; self.test_outputs = [];
    self.losses = []; self.full_losses = [];
    for i, b in enumerate(self.buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True if i > 0 else None):

        output, _ = seq2seq_f(self.encoder_inputs[:b[0]], self.decoder_inputs[:b[1]], False)
        self.outputs.append(output)
        tf.get_variable_scope().reuse_variables()
        self.losses.append(sampled_sequence_loss(output, targets[:b[1]], self.target_weights[:b[1]], proj_w_t, proj_b, num_samples, target_vocab_size))

        test_output, _ = seq2seq_f(self.encoder_inputs[:b[0]], self.decoder_inputs[:b[1]], True)
        test_output = [tf.matmul(out, proj_w) + proj_b for out in test_output] # Extra last layer
        self.test_outputs.append(test_output)

        full_output = [tf.matmul(out, proj_w) + proj_b for out in output] # Extra last layer
        self.full_outputs.append(full_output)
        self.full_losses.append(sequence_loss(full_output, targets[:b[1]], self.target_weights[:b[1]]))
    
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    params = tf.trainable_variables()
    self.full_gradient_norms = []; self.full_updates = [];
    self.gradient_norms = []; self.updates = [];
    # Gradients and SGD update operation for training the model.
    for i, b in enumerate(self.buckets):
      # Full Softmax model
      full_gradients = tf.gradients(self.full_losses[i], params)
      clipped_gradients, norm = tf.clip_by_global_norm(full_gradients, max_gradient_norm)
      self.full_gradient_norms.append(norm)
      self.full_updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

      # Sampled Softmax model
      gradients = tf.gradients(self.losses[i], params)
      clipped_gradients2, norm2 = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.gradient_norms.append(norm2)
      self.updates.append(opt.apply_gradients(zip(clipped_gradients2, params), global_step=self.global_step))

  def build_input_feed(self, encoder_inputs, decoder_inputs, target_weights, bucket_id):
      encoder_size, decoder_size = self.buckets[bucket_id]
      input_feed = {}
      for l in xrange(encoder_size):
        input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      for l in xrange(decoder_size):
        input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        if target_weights != None:
          input_feed[self.target_weights[l].name] = target_weights[l]
      # last_target = self.decoder_inputs[decoder_size].name
      # input_feed[last_target] = np.zeros([len(encoder_inputs[0])], dtype=np.int32)
      return input_feed

  def train_step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, sampled=True):
    input_feed = self.build_input_feed(encoder_inputs, decoder_inputs, target_weights, bucket_id)
    if sampled:
      output_feed = [self.losses[bucket_id], self.updates[bucket_id]]
    else:
      output_feed = [self.full_losses[bucket_id], self.full_updates[bucket_id]]

    outputs = session.run(output_feed, input_feed)

    return outputs[0], outputs[1]  # Gradient norm, loss

  def test_step(self, session, encoder_inputs, bucket_id):
    decoder_inputs = np.zeros((self.buckets[bucket_id][1], len(encoder_inputs[0])),dtype=np.int32)
    decoder_inputs[0, :] = 1

    input_feed = self.build_input_feed(encoder_inputs, decoder_inputs, None, bucket_id)

    outputs = session.run(self.test_outputs[bucket_id], input_feed)
    return outputs