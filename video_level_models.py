# Copyright 2017 Xinyi Wang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")


class LogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
   
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):

    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class ThreeLayerNN(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_hidden_units=2048, l2_penalty=1e-7, prefix='', **unused_params):

    # Initialize weights for projection
    w_s = tf.Variable(tf.random_normal(shape=[1024, 2048], stddev=0.01))
    input_projected = tf.matmul(model_input, w_s)

    hidden1 = tf.layers.dense( 
        inputs=model_input, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_1')


    bn1 = tf.contrib.layers.batch_norm(inputs=hidden1, decay=0.99, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn1')

    relu1 = tf.nn.relu(hidden1, name=prefix+'relu1' )

    dropout1 = tf.layers.dropout(inputs=relu1, rate=0.5, name=prefix+"dropout1")


    hidden2 = tf.layers.dense(
        inputs=dropout1, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_2')

    bn2 = tf.contrib.layers.batch_norm(inputs=hidden2, decay=0.99, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn2')


    relu2 = tf.nn.relu(hidden2, name=prefix+'relu2' )

    dropout2 = tf.layers.dropout(inputs=relu2, rate=0.5, name=prefix+"dropout2")

    input_projected_plus_h2 = tf.add(input_projected, dropout2)

    output = slim.fully_connected(
        input_projected_plus_h2, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_3')


    weights_norm = tf.add_n(tf.losses.get_regularization_losses())
    
    return {"predictions": output, "regularization_loss": weights_norm}
    #return {"predictions": output}
