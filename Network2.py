import numpy as np
import math
from keras.initializers import normal, identity
from keras.initializers import VarianceScaling 	
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
import tensorflow.contrib.slim as slim
import keras.backend as K
import sys

hidden1_units = 128
hidden2_units = 128
hidden3_units = 32

class VanillaPolicyGradientNetwork(object):



    def __init__(self, lr, s_size, a_size, h_size, h2_size):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden0 = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden1 = slim.fully_connected(hidden0, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden2_1 = slim.fully_connected(hidden1, h2_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden2_2 = slim.fully_connected(hidden1, h2_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden2_3 = slim.fully_connected(hidden1, h2_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden2_4 = slim.fully_connected(hidden1, h2_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden3_1 = slim.fully_connected(hidden2_1, a_size, biases_initializer=None, activation_fn=tf.nn.softmax)
        hidden3_2 = slim.fully_connected(hidden2_2, a_size, biases_initializer=None, activation_fn=tf.nn.softmax)
        hidden3_3 = slim.fully_connected(hidden2_3, a_size, biases_initializer=None, activation_fn=tf.nn.softmax)
        hidden3_4 = slim.fully_connected(hidden2_4, a_size, biases_initializer=None, activation_fn=tf.nn.softmax)
        self.output = Concatenate()([hidden3_1, hidden3_2, hidden3_3, hidden3_4])
        #self.output = slim.fully_connected(hidden3, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        #self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.action_holder2 = tf.reshape(self.action_holder, [-1]) 

 
        self.aaa = tf.range(0, tf.shape(self.action_holder2)[0]*5, 20)

        self.a_c1 = self.aaa + tf.gather(self.action_holder2, tf.range(0, tf.shape(self.action_holder2)[0] , 4))
        self.a_c2 = self.aaa + tf.gather(self.action_holder2, tf.range(1,  tf.shape(self.action_holder2)[0], 4))
        self.a_c3 = self.aaa + tf.gather(self.action_holder2, tf.range(2,  tf.shape(self.action_holder2)[0], 4))
        self.a_c4 = self.aaa + tf.gather(self.action_holder2, tf.range(3,  tf.shape(self.action_holder2)[0], 4))

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] #+ self.action_holder2

        self.responsible_outputs = []
        #self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.responsible_outputs2 = np.zeros_like(self.action_holder)

        self.responsible_outputs01=tf.reshape(self.output, [-1])

        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.responsible_outputs_ok = []
        self.responsible_outputs_ok.append(tf.gather(tf.reshape(self.output, [-1]), self.a_c1))
        self.responsible_outputs_ok.append(tf.gather(tf.reshape(self.output, [-1]), self.a_c2))
        self.responsible_outputs_ok.append(tf.gather(tf.reshape(self.output, [-1]), self.a_c3))
        self.responsible_outputs_ok.append(tf.gather(tf.reshape(self.output, [-1]), self.a_c4))

        self.loss0 = tf.log(self.responsible_outputs) * self.reward_holder
        self.loss00 = tf.log(self.responsible_outputs[0]) * self.reward_holder
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs_ok) * self.reward_holder)

        tvars = tf.trainable_variables()
        print("t   v   a   r   s")
        print(tvars)
        print("endoftvars")

        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))







 
    def train(self, feed_dict, sess, gradBuffer, states, i):
        print(self.action_holder)

        print("                 i n d e x e s")
        """print(sess.run(self.a_c1, feed_dict = feed_dict))
        print(tf.shape(self.output))
        print(tf.shape(self.output)[1])
        print(tf.reshape(self.output, [-1]))
        print(self.responsible_outputs)"""
        grads = sess.run(self.gradients, feed_dict=feed_dict)
        for idx, grad in enumerate(grads):
            gradBuffer[idx] += grad

        """print("A")
        print_op = tf.print(self.indexes)
        print(sess.run(self.indexes, feed_dict = states))
        print(sess.run(self.output, feed_dict = states))
        print("B")
        print(sess.run(self.responsible_outputs_ok, feed_dict = states))
        print("C")
        print("C")
        print("C")
        print(sess.run(self.action_holder2, feed_dict = feed_dict))
        print("          r   e   s   h   a   p   e   d   ")
        print(sess.run(self.responsible_outputs01, feed_dict = feed_dict))

        print("C")
        print("C")
        print(sess.run(self.loss00, feed_dict = feed_dict))
        print("C")
        print(sess.run(self.reward_holder, feed_dict = feed_dict))
        print("C")
        print("C")"""


        


        if i % 5 == 0 and i != 0:

            feed_dict = dictionary = dict(zip(self.gradient_holders, gradBuffer))
            _ = sess.run(self.update_batch, feed_dict=feed_dict)
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0



        
    def _compute_gradients(self, tensor, var_list):
      grads = tf.gradients(tensor, var_list)
      return [grad if grad is not None else tf.zeros_like(var)
              for var, grad in zip(var_list, grads)]


    def logits_to_actions(self, last_layer, actionsize):
        x = tf.placeholder(tf.float32,[None, actionsize])
        y = x
        self.sess.run(y, feed_dict = {x: last_layer})
        return tf.nn.softmax(y)


    def loss_counter(self, logits, actions, discounted_episode_rewards):
        neg_prob_log = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = actions)
        return tf.reduce_mean(neg_prob_log * discounted_episode_reward)
