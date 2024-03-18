import tensorflow as tf
from typing import Dict
from tensorflow.keras import backend as K


def weighted_binary_cross_entropy(weights: Dict[int, float]):
  """Return a function to calculate weighted binary xent with multi-hot labels.

  Due to @menrfa
  (https://stackoverflow.com/questions/46009619/
    keras-weighted-binary-crossentropy)

  Returns:
    A function to calculate (weighted) binary cross entropy.
  """
  if 0 not in weights or 1 not in weights:
    raise NotImplementedError

  def weighted_cross_entropy_fn(y_true, y_pred, from_logits=False):
    tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
    tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

    weight_1 = tf.cast(weights[1], dtype=y_pred.dtype)
    weight_0 = tf.cast(weights[0], dtype=y_pred.dtype)
    weights_v = tf.where(tf.equal(tf_y_true, 1), weight_1, weight_0)
    ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
    loss = K.mean(tf.multiply(ce, weights_v), axis=-1)
    return loss

  return weighted_cross_entropy_fn
