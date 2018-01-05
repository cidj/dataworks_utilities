#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:57:44 2017

@author: Tao Su
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx


def batch_norm(x, is_training, axes, decay=0.99, epsilon=1e-3,scope='bn', reuse=None):
    """
    Performs a batch normalization layer. For so-called "global normalization",
    used with convolutional filters with shape [batch, height, width, depth],
    pass axes=[0, 1, 2].For simple batch normalization pass axes=[0] (batch only).

    Parameters:
        x: Input tensor.
        is_training: tf.bool type value, tensor or variable.
        axes: Array of ints. Axes along which to compute mean and variance.
        decay: The moving average decay.
        epsilon: The variance epsilon - a small float number to avoid dividing by 0.
        scope: Scope name.
        reuse:if True, we go into reuse mode for this scope as well as all sub-scopes;
        if None, we just inherit the parent scope reuse.

    Returns:
        Batch normalization layer or maps.
    """

    with tf.variable_scope(scope,reuse=reuse):

#         beta = tf.Variable(tf.constant(0.0, shape=[x.get_shape()[-1]]),name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[x.get_shape()[-1]]),name='gamma', trainable=True)
        beta = tf.get_variable("beta", x.get_shape()[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        gamma = tf.get_variable("gamma", x.get_shape()[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
    return normed
