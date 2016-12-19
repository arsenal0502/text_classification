from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import get_data
import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

from tensorflow.contrib import learn

FLAGS = None

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 15
n_words = 0


def bag_of_words_model(features, target):
  """A bag-of-words model. Note it disregards the word order in the text."""
  target = tf.one_hot(target, 24, 1, 0)
  features = tf.contrib.layers.bow_encoder(
      features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
  logits = tf.contrib.layers.fully_connected(features, 24, activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)
  return (
      {'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)},
      loss, train_op)

if __name__ == '__main__':
  global n_words
  # Prepare training and testing data
  train_x,test_x,train_y,test_y,n_words=get_data()
  print('Total words: %d' % n_words)
  train_x=iter(train_x)
  train_y=iter(train_y)
  test_x=iter(test_x)
  # Build model
  model_fn = bag_of_words_model
  classifier = learn.Estimator(model_fn=model_fn)
  # Train and predict
  classifier.fit(train_x, train_y, steps=6000)
  predicted_y=[
      p['class'] for p in classifier.predict(test_x, as_iterable=True)]
  score = metrics.accuracy_score(test_y, predicted_y)
  print('Accuracy: {0:f}'.format(score))

