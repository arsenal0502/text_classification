# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import get_data,get_data_dynamic
import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

from tensorflow.contrib import learn
FLAGS = None

MAX_DOCUMENT_LENGTH = 22
EMBEDDING_SIZE = 15
n_words = 0
max_epochs=50
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def last_relevant(output, length):#找到最后的结尾向量
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def dynamic_rnn_model(features, target):
  """RNN model to predict from sequence of words to a class."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].  



  word_vectors = tf.contrib.layers.embed_sequence(
      features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')
  sequence_length=length(word_vectors)
  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = tf.unstack(word_vectors, axis=1)

  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
  print(sequence_length)
  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  #_, encoding = tf.nn.rnn(cell, word_list, dtype=tf.float32)
  output,encoding=tf.nn.dynamic_rnn(cell,word_vectors, dtype=tf.float32,sequence_length=sequence_length,)
  output=tf.reduce_mean(output,reduction_indices=1)
  # Given encoding of RNN, take encoding of last step (e.g hidden size of the
  # neural network of last step) and pass it as features for logistic
  # regression over output classes.
  target = tf.one_hot(target, 24, 1, 0)
  logits = tf.contrib.layers.fully_connected(output, 24, activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

  # Create a training op.
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)

  return (
      {'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)},
      loss, train_op)



if __name__ == '__main__':
  global n_words
  # Prepare training and testing data
  train_x,test_x,train_y,test_y,n_words=get_data_dynamic()
  train_x=np.array(train_x)
  train_y=np.array(train_y)
  train_y=np.reshape(train_y,[-1,1])
  print('Total words: %d' % n_words)
  
  train_x=iter(train_x)
  train_y=iter(train_y)
  test_x=iter(test_x)
  #print(train_x.shape(),train_y.shape())
  # Build model
  # Switch between rnn_model and bag_of_words_model to test different models.
  model_fn = dynamic_rnn_model
  classifier = learn.Estimator(model_fn=model_fn)
  # Train and predict
  classifier.fit(train_x, train_y, steps=3000)
  predicted_y=[
      p['class'] for p in classifier.predict(test_x, as_iterable=True)]
  score = metrics.accuracy_score(test_y, predicted_y)
  print('Accuracy: {0:f}'.format(score))


