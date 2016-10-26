from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random, numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

class Seq2SeqModel(object):
  def __init__(self,source_vocab_size,target_vocab_size,buckets,size,num_layers,max_gradient_norm,batch_size,learning_rate,num_samples=512,forward_only=False,dtype=tf.float32):
    """source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: size of batches. Can be changed later
      learning_rate: learning rate to start with.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size

    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.99) # This is a function we run when needed

    self.global_step = tf.Variable(0, trainable=False) # This is in case we want to use exponential decay on LR

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        return tf.cast(tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels, num_samples, self.target_vocab_size), dtype)
      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, feed_previous):
      return tf.nn.seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, num_encoder_symbols=source_vocab_size,
      num_decoder_symbols=target_vocab_size, embedding_size=size, output_projection=output_projection, feed_previous=feed_previous, dtype=dtype)

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
    self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, targets,
        self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, forward_only), softmax_loss_function=softmax_loss_function)

    if forward_only and output_projection is not None:
      # If we use output projection, we need to project outputs for decoding.
        for b in xrange(len(buckets)):
          self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs[b]] # Extra last layer

    # Gradients and SGD update operation for training the model.
    if not forward_only:
      params = tf.trainable_variables()
      self.gradient_norms = []; self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
    """session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.losses[bucket_id], self.gradient_norms[bucket_id]] # , self.updates[bucket_id]
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[0], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.