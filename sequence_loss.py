from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

import tensorflow as tf

def sequence_loss(logits, targets, weights, name=None):
	"""Weighted cross-entropy loss for a sequence of logits, batch-collapsed.
	Args:
		logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
		targets: List of 1D batch-sized int32 Tensors of the same length as logits.
		weights: List of 1D batch-sized float-Tensors of the same length as logits.
	Returns:
		A scalar float Tensor: The average log-perplexity per symbol (weighted).
	"""
	with ops.name_scope(name, "sequence_loss", logits + targets + weights):
		log_perp_list = []
		for logit, target, weight in zip(logits, targets, weights):
			crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logit, target)
			log_perp_list.append(crossent * weight)
		log_perps = math_ops.add_n(log_perp_list)
		total_size = math_ops.add_n(weights) + 1e-12 # Just to avoid division by 0 for all-0 weights.
		log_perps /= total_size

		return math_ops.reduce_mean(log_perps)

def sampled_sequence_loss(inputs, targets, weights, w_t, b, num_samples, target_vocab_size, name="sampled_sequence_loss"):
	# We need to compute the sampled_softmax_loss using 32bit floats to avoid numerical instabilities.
	def sampled_loss(inputs, labels, w_t, b):
		labels = tf.reshape(labels, [-1, 1])
		# We need to compute the sampled_softmax_loss using 32bit floats to
		# avoid numerical instabilities.
		local_w_t = tf.cast(w_t, tf.float32)
		local_b = tf.cast(b, tf.float32)
		local_inputs = tf.cast(inputs, tf.float32)
		return tf.cast(tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels, num_samples, target_vocab_size), tf.float32)

	with ops.name_scope(name, "sequence_loss", inputs + targets + weights):
		local_w_t = tf.cast(w_t, tf.float32)
		local_b = tf.cast(b, tf.float32)
		log_perp_list = []
		for inp, target, weight in zip(inputs, targets, weights):
			target = tf.reshape(target, [-1, 1])
			local_input = tf.cast(inp, tf.float32)
			crossent = tf.cast(tf.nn.sampled_softmax_loss(local_w_t, local_b, local_input, target, num_samples, target_vocab_size), tf.float32)
			log_perp_list.append(crossent * weight)
		log_perps = math_ops.add_n(log_perp_list)
		total_size = math_ops.add_n(weights) + 1e-12 # Just to avoid division by 0 for all-0 weights.
		log_perps /= total_size

		return math_ops.reduce_mean(log_perps)