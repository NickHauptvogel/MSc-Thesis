# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default priors for Bayesian neural networks.

Prior factories can create suitable priors given Keras layers as input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import prior
layers = tf.keras.layers


class PriorFactory(object):
  """Prior factory base class.

  The prior factory is a helper class that makes the task of adding proper
  prior distributions to Keras models easy.

  Examples:
    The following code instantiates a prior factory object and shows how to
    wrap a newly created layer in order to add proper default priors to all
    parameters of the layer.

    >>> pfac = DefaultPriorFactory(weight=1.0/total_train_size)
    >>> dense = pfac(tf.keras.layers.Dense(32))

  """

  def __init__(self, weight=0.0, prior_dict=None):
    """Construct a new PriorFactory object.

    Args:
      weight: prior weight, typically 1.0/total_train_sample_size for Bayesian
        neural networks.  Must be >0.0.
      prior_dict: dict, containing as keys layer.name values and as value a dict
        describing the regularizers to add to the respective layer.
        The prior_dict can be used to override choices made by other
        PriorFactory classes, i.e. it always takes precedence in determining
        priors.

    Raises:
      ValueError: invalid value for weight keyword argument.
    """
    # The weight parameter is critical so we force the user to set a value
    if weight <= 0.0:
      raise ValueError('You must provide a "weight" argument to the prior '
                       'factory.  Typically weight=1.0/total_train_size, '
                       'where total_train_size is the number of iid training '
                       'instances.')

    self.weight = weight
    self.prior_dict = prior_dict

  def _replace(self, config, rdict):
    """Replace a key in a tf.keras.layers.Layer config dictionary.

    This method replaces a regularizer key in a layer.get_config() dictionary
    with specified elements from the rdict regularization dictionary.

    Examples:
      >>> embedding = tf.keras.layers.Embedding(5000, 512)
      >>> config = embedding.get_config()
      >>> pfac = DefaultPriorFactory(weight=1.0/50000.0)
      >>> rdict = {'embeddings_regularizer': {
              'class_name': 'NormalRegularizer',
              'config': {'stddev': 0.1, 'weight': 1.0/50000.0} } }
      >>> pfac._replace(config, rdict)

    Args:
      config: dict, containing the layer.get_config() dictionary to modify.
      rdict: dict, regularizer keys/values to put into config dictionary.
    """
    # If fixed prior is used, replace rdict using prior dictionary
    layer_name = config['name']
    if (self.prior_dict is not None) and (layer_name in self.prior_dict):
      logging.info('Using regularizer for layer "%s" from prior_dict',
                   layer_name)
      rdict = self.prior_dict[layer_name]

    if rdict is None:
      return

    for name in rdict:
      if config[name] is not None:
        logging.warn('Warning: Overriding regularizer from layer "%s"s %s',
                     layer_name, name)

      config[name] = rdict[name]

  def _update_prior(self, layer, config):
    """Update the config dictionary for the given layer.

    This abstract method must be overridden by concrete implementations in
    derived classes.

    The method's job is to select for a given 'layer' the corresponding priors
    that are suitable.

    Args:
      layer: tf.keras.layers.Layer class.
      config: the layer.get_config() dictionary.  This argument must be
        modified.  The modified dictionary will then be used to reconstruct the
        layer.
    """
    raise NotImplementedError('Users must override the _update_prior method '
                              'of PriorFactory')

  def __call__(self, layer):
    """Add a prior to the newly constructed input layer.

    Args:
      layer: tf.keras.layers.Layer that has just been constructed (not built, no
        graph).

    Returns:
      layer_out: the layer with a suitable prior added.
    """
    if not layer.trainable:
      return layer

    # Obtain serialized layer representation and replace priors
    config = layer.get_config()
    self._update_prior(layer, config)

    # Reconstruct prior from updated serialized representation
    with bnn_scope():
      layer_out = type(layer).from_config(config)

    return layer_out


DEFAULT_GAUSSIAN_PFAC_STDDEV = 1.0


class GaussianPriorFactory(PriorFactory):
  """Gaussian prior factory for Bayesian neural networks.

  This prior was used in [Zhang et al., 2019].
  """

  def __init__(self, prior_stddev=DEFAULT_GAUSSIAN_PFAC_STDDEV, **kwargs):
    super(GaussianPriorFactory, self).__init__(**kwargs)
    self.prior_stddev = prior_stddev

  def normal(self, _):
    normal_dict = {
        'class_name': 'NormalRegularizer',
        'config': {'stddev': self.prior_stddev, 'weight': self.weight},
    }
    return normal_dict

  def _update_prior(self, layer, config):

    if isinstance(layer, layers.Dense) or isinstance(layer, layers.Conv1D) or \
        isinstance(layer, layers.Conv2D):
      self._replace(config, {
          'kernel_regularizer': self.normal(layer),
          'bias_regularizer': self.normal(layer),
      })
    elif isinstance(layer, layers.Embedding):
      self._replace(config, {
          'embeddings_regularizer': self.normal(layer),
      })
    elif isinstance(layer, layers.LSTM):
      self._replace(config, {
          'kernel_regularizer': self.normal(layer),
          'recurrent_regularizer': self.normal(layer),
          'bias_regularizer': self.normal(layer),
      })
    else:
      logging.warning('Layer type "%s" not found', type(layer))


def bnn_scope():
  """Create a scope that is aware of BNN library objects.

  Returns:
    scope: tf.keras.utils.CustomObjectScope with object/class mapping.
  """
  scope_dict = {
      'NormalRegularizer': prior.NormalRegularizer,
      'StretchedNormalRegularizer': prior.StretchedNormalRegularizer,
      'HeNormalRegularizer': prior.HeNormalRegularizer,
      'GlorotNormalRegularizer': prior.GlorotNormalRegularizer,
      'LaplaceRegularizer': prior.LaplaceRegularizer,
      'CauchyRegularizer': prior.CauchyRegularizer,
      'SpikeAndSlabRegularizer': prior.SpikeAndSlabRegularizer,
      'EmpiricalBayesNormal': prior.EmpiricalBayesNormal,
      'HeNormalEBRegularizer': prior.HeNormalEBRegularizer,
      'ShiftedNormalRegularizer': prior.ShiftedNormalRegularizer,
  }
  scope_dict.update(tf.keras.utils.get_custom_objects())
  scope = tf.keras.utils.CustomObjectScope(scope_dict)

  return scope
