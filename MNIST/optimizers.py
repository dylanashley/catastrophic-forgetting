# -*- coding: utf-8 -*-

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.keras.optimizers import Optimizer


@tf_export(v1=['keras.optimizers.NMOM'])
class NMOM(Optimizer):
  """Stochastic gradient descent optimizer.

  Includes support for momentum,
  learning rate decay, and Nesterov momentum.

  Arguments:
      lr: float >= 0. Learning rate.
      momentum: float >= 0. Parameter that accelerates NMOM
          in the relevant direction and dampens oscillations.
      decay: float >= 0. Learning rate decay over each update.
  """

  def __init__(self, lr=0.01, momentum=0., decay=0., **kwargs):
    super(NMOM, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.lr = K.variable(lr, name='lr')
      self.momentum = K.variable(momentum, name='momentum')
      self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))
    # momentum
    shapes = [K.int_shape(p) for p in params]
    moments = [K.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + moments
    for p, g, m in zip(params, grads, moments):
      v = - self.momentum * math_ops.abs(m) - lr * g  # velocity
      self.updates.append(state_ops.assign(m, v))
      new_p = p - v

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'momentum': float(K.get_value(self.momentum)),
        'decay': float(K.get_value(self.decay))
    }
    base_config = super(NMOM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
