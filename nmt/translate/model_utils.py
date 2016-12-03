from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from six.moves import zip

from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs


def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       seq2seq_f,
                       buckets,
                       targets=None,
                       weights=None,
                       loss_f=None,
                       name=None):
    """
    Create sequence-to-sequence models for each bucket.
    """

    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of last bucket (%d)."
                         % (len(encoder_inputs), buckets[-1][0]))

    if len(decoder_inputs) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last bucket (%d)."
                         % (len(targets), buckets[-1][1]))

    if weights is not None and len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last bucket (%d)."
                         % (len(weights), buckets[-1][1]))

    all_inputs = encoder_inputs + decoder_inputs
    all_inputs += targets if targets is not None else []
    all_inputs += weights if weights is not None else []
    losses = []
    outputs = []
    reuse_f = lambda k: True if k > 0 else None

    with ops.name_scope(name, 'model_with_buckets', all_inputs):
        for i, bucket in enumerate(buckets):
            with vs.variable_scope(vs.get_variable_scope(), reuse=reuse_f(i)):
                bucket_outputs, _ = seq2seq_f(encoder_inputs[:bucket[0]],
                                              decoder_inputs[:bucket[1]])
                outputs.append(bucket_outputs)
                if loss_f is not None:
                    losses.append(loss_f(bucket_outputs,
                                         targets[:bucket[1]],
                                         weights[:bucket[1]]))
    return outputs, losses


def _compute_log_perps(log_perps, weights):
    return math_ops.reduce_mean(math_ops.add_n(log_perps) / (math_ops.add_n(weights) + 1e-12))


def _sampled_loss(output,
                  target,
                  proj_w_t,
                  proj_b,
                  target_vocab_size,
                  num_samples=512):
    """
    Helper function for sampled softmax loss.
    """

    target = tf.reshape(target, [-1, 1])
    return tf.nn.sampled_softmax_loss(proj_w_t, proj_b, output, target, num_samples, target_vocab_size)


def sampled_sequence_loss(outputs,
                          targets,
                          weights,
                          proj_w_t,
                          proj_b,
                          target_vocab_size,
                          num_samples=512,
                          name=None,
                          dtype=tf.float32):
    """
    Compute the sampled softmax loss for the outputs.
    """
    with ops.name_scope(name, 'sequence_loss', outputs + targets + weights):
        cast_f = lambda x: tf.cast(x, dtype)
        local_w_t = cast_f(proj_w_t)
        local_b = cast_f(proj_b)
        log_perps = []
        for output, target, weight in zip(outputs, targets, weights):
            cross_ent = cast_f(_sampled_loss(cast_f(output),
                                             target,
                                             local_w_t,
                                             local_b,
                                             target_vocab_size,
                                             num_samples))
            log_perps.append(cross_ent * weight)

        return _compute_log_perps(log_perps, weights)


def full_sequence_loss(outputs,
                       targets,
                       weights,
                       proj_w,
                       proj_b,
                       name=None):
    """
    Compute the full softmax loss for the outputs.
    """
    with ops.name_scope(name, 'sequence_loss', outputs + targets + weights):
        outputs = [tf.matmul(output, proj_w) + proj_b for output in outputs]
        log_perps = []
        for output, target, weight in zip(outputs, targets, weights):
            cross_ent = nn_ops.sparse_softmax_cross_entropy_with_logits(output, target)
            log_perps.append(cross_ent * weight)

        return _compute_log_perps(log_perps, weights)
