#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:57:44 2017

@author: Tao Su
"""
import numpy as np
import pandas as pd
import tensorflow as tf


#Input functions generators

def csv_input_fn(data_file,label,num_epochs, shuffle, batch_size,header='infer',names=None):
    """ Generate a input function from a csv file."""
    
    _DEFAULT_DTYPE_DICT={np.dtype('int64'):[0],np.dtype('float64'):[0.0], np.dtype('O'):[""]}
    
    dfdata=pd.read_csv(data_file,header=header,names=names)
    
    _CSV_COLUMNS = dfdata.columns.tolist()
    _CSV_COLUMN_DEFAULTS=dfdata.dtypes.map(_DEFAULT_DTYPE_DICT).tolist()
        
    if names:
        skip=1
    else:
        skip=0
   
    def parse_csv(value,label_col):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop(label_col)
        return features, labels
        
    def input_fn():
    
        dataset = tf.data.TextLineDataset(data_file).skip(skip)
    
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(dfdata))
    
        dataset = dataset.map(lambda x: parse_csv(x,label),num_parallel_calls=5)
    
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        
        iterator = dataset.make_one_shot_iterator()
      
        return iterator.get_next()
    
    return input_fn


def df_input_fn(features,label,num_epochs, shuffle, batch_size):
    """ Generate a input function from a pandas DataFrame."""
            
    def input_fn():
    
        dataset = tf.data.Dataset.from_tensor_slices((dict(features),label))
    
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features))
    
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        
        iterator = dataset.make_one_shot_iterator()
      
        return iterator.get_next()
    
    return input_fn


#Write to and read from tfrecords file.
    
def df_to_tfrecord(df,names_file, tfrecord_file):
    """Write a pandas DataFrame to a tfrecords file and a csv for column names. 
    Here strings are encoded using utf-8"""
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))    
    def _string_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))    
#    def _bytes_feature(value):
#        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    _DTYPE_FUNCTION_DICT={np.dtype('int64'):_int64_feature,
                          np.dtype('float64'):_float_feature,
                          np.dtype('O'):_string_feature}
    
    f={i:_DTYPE_FUNCTION_DICT[df.dtypes[i]] for i in df.columns}
    df.dtypes.to_csv(names_file)
    
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(0,len(df)):
            x=df.iloc[i] 
            example = tf.train.Example(features=tf.train.Features(feature={
                    i:f[i](x[i]) for i in df.columns}))
            writer.write(example.SerializeToString())
            

def get_tfrecord_parse_function(names_file,use_default_value=False):
    """Read the csv file generated by df_to_tfrecord and creat a parse function
    for tf.dataset API."""
    
    _TYPE_TF_DICT={'int64':tf.int64,'float64':tf.float64,'object':tf.string}
    _TYPE_TF_DEFAULT={'int64':0,'float64':0.0,'object':""}
    
    bb=pd.read_csv(names_file,header=None,index_col=0,squeeze=True)
    cc=bb.map(_TYPE_TF_DICT)
    dd=bb.map(_TYPE_TF_DEFAULT)
    
    if use_default_value:
        hh={i:tf.FixedLenFeature([],cc[i],default_value=dd[i]) for i in bb.index}
    else:
        hh={i:tf.FixedLenFeature([],cc[i]) for i in bb.index}
    
    def _parse_function(example_proto):
      parsed_features = tf.parse_single_example(example_proto, hh)
      return parsed_features
  
    return _parse_function    


def read_tfdataset_to_df(dataset, read_len, start_ind=0):
    """Read tensorflow dataset with specific length and start index and convert
    it to pandas DataFrame.
    """
    iter1=dataset.make_one_shot_iterator()
    datf=pd.DataFrame()
    sess=tf.Session()
    end_ind=start_ind+read_len
    for i in range(0,end_ind):
        try:
            rec=sess.run(iter1.get_next())
            if i in range(start_ind,end_ind):
                datf=datf.append(pd.DataFrame(rec,index=[i]))
        except tf.errors.OutOfRangeError:
            print("Dataset out of range: ", i, " records in total.")   
    sess.close() 
    return datf



#some pattern.

def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    
    input_layer = tf.feature_column.input_layer(features, params['feature_columns'])

    ...#creat model use input layer.

    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = model(..., training=True)
        inferences=tf.argmax(logits, axis=1)
        loss = ...
        accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=inferences)
        tf.summary.scalar('train_accuracy', accuracy[1])
        optimizer = ...
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(loss,
                                            tf.train.get_or_create_global_step()))
    else:
        logits = model(..., training=False)
        inferences=tf.argmax(logits, axis=1)
        if mode == tf.estimator.ModeKeys.EVAL:            
            loss = ...
            accuracy = tf.metrics.accuracy(
                    labels=labels,
                    predictions=inferences)
            return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.EVAL,
                    loss=loss,
                    eval_metric_ops={
                            'eval_accuracy':accuracy})
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                    'classes': inferences,
                    'probabilities': tf.nn.softmax(logits),}
            return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.PREDICT,
                    predictions=predictions,
                    export_outputs={
                            'classify': tf.estimator.export.PredictOutput(predictions)})
#        else:
#            raise ValueError("Mode doesn't exist: (TRAIN/EVAL/PREDICT).")
            

# A Framework for tensorflow estimator.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
        
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', default=1000, type=int,
                        help='number of training steps')
    args = parser.parse_args(argv[1:])    

    classifier = tf.estimator.Estimator(
            model_fn=...,
            params={
                    'feature_columns': ...,
                    'n_classes': ...,
                    },
            model_dir=...)

    classifier.train(
            input_fn=...,
            steps=args.train_steps)

    eval_result = classifier.evaluate(
            input_fn=...)
    
    predictions = classifier.predict(
            input_fn=...)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

#Optimization trick snippets.
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







################################################
#######Codes below are from internet.
#Haven't tested yet.
"""adapted from https://github.com/OlavHN/bnlstm to store separate population statistics per state"""

RNNCell = tf.nn.rnn_cell.RNNCell

class BNLSTMCell(RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''
    def __init__(self, num_units, is_training_tensor, max_bn_steps, initial_scale=0.1, activation=tf.tanh, decay=0.95):
        """
        * max bn steps is the maximum number of steps for which to store separate population stats
        """
        self._num_units = num_units
        self._training = is_training_tensor
        self._max_bn_steps = max_bn_steps
        self._activation = activation
        self._decay = decay
        self._initial_scale = 0.1

    @property
    def state_size(self):
        return (self._num_units, self._num_units, 1)

    @property
    def output_size(self):
        return self._num_units

    def _batch_norm(self, x, name_scope, step, epsilon=1e-5, no_offset=False, set_forget_gate_bias=False):
        '''Assume 2d [batch, values] tensor'''

        with tf.variable_scope(name_scope):
            size = x.get_shape().as_list()[1]

            scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(self._initial_scale))
            if no_offset:
                offset = 0
            elif set_forget_gate_bias:
                offset = tf.get_variable('offset', [size], initializer=offset_initializer())
            else:
                offset = tf.get_variable('offset', [size], initializer=tf.zeros_initializer)

            pop_mean_all_steps = tf.get_variable('pop_mean', [self._max_bn_steps, size], initializer=tf.zeros_initializer, trainable=False)
            pop_var_all_steps = tf.get_variable('pop_var', [self._max_bn_steps, size], initializer=tf.ones_initializer(), trainable=False)

            step = tf.minimum(step, self._max_bn_steps - 1)

            pop_mean = pop_mean_all_steps[step]
            pop_var = pop_var_all_steps[step]

            batch_mean, batch_var = tf.nn.moments(x, [0])

            def batch_statistics():
                pop_mean_new = pop_mean * self._decay + batch_mean * (1 - self._decay)
                pop_var_new = pop_var * self._decay + batch_var * (1 - self._decay)
                with tf.control_dependencies([pop_mean.assign(pop_mean_new), pop_var.assign(pop_var_new)]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

            def population_statistics():
                return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

            return tf.cond(self._training, batch_statistics, population_statistics)

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h, step = state
            _step = tf.squeeze(tf.gather(tf.cast(step, tf.int32), 0))

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self._num_units],
                initializer=orthogonal_lstm_initializer())
            W_hh = tf.get_variable('W_hh',
                [self._num_units, 4 * self._num_units],
                initializer=orthogonal_lstm_initializer())

            hh = tf.matmul(h, W_hh)
            xh = tf.matmul(x, W_xh)

            bn_hh = self._batch_norm(hh, 'hh', _step, set_forget_gate_bias=True)
            bn_xh = self._batch_norm(xh, 'xh', _step, no_offset=True)

            hidden = bn_xh + bn_hh

            f, i, o, j = tf.split(1, 4, hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * self._activation(j)
            bn_new_c = self._batch_norm(new_c, 'c', _step)

            new_h = self._activation(bn_new_c) * tf.sigmoid(o)
            return new_h, (new_c, new_h, step+1)

def orthogonal_lstm_initializer():
    def orthogonal(shape, dtype=tf.float32, partition_info=None):
        # taken from https://github.com/cooijmanstim/recurrent-batch-normalization
        # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
        """ benanne lasagne ortho init (faster than qr approach)"""
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return tf.constant(q[:shape[0], :shape[1]], dtype)
    return orthogonal

def offset_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        size = shape[0]
        assert size % 4 == 0
        size = size // 4
        res = [np.ones((size)), np.zeros((size*3))]
        return tf.constant(np.concatenate(res, axis=0), dtype)
    return _initializer




#Useful codes snippets
    
import matplotlib.pyplot as plt
import networkx as nx


def children(op):
  return set(op for out in op.outputs for op in out.consumers())

def get_graph():
  """Creates dictionary {node: {child1, child2, ..},..} for current
  TensorFlow graph. Result is compatible with networkx/toposort"""

  ops = tf.get_default_graph().get_operations()
  return {op: children(op) for op in ops}

def plot_graph(G):
    '''Plot a DAG using NetworkX'''        
    def mapping(node):
        return node.name
    G = nx.DiGraph(G)
    nx.relabel_nodes(G, mapping, copy=False)
    nx.draw(G, cmap = plt.get_cmap('jet'), with_labels = True)
    plt.show()


#x = tf.Variable(0, name='x')
#model = tf.global_variables_initializer()
#with tf.Session() as session:
#    for i in range(5):
#        session.run(model)
#        x = x + 1
#        print(session.run(x))
#
#        plot_graph(get_graph())
