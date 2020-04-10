import glob
import os
import sys

import time
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json

#from ReplayBuffer import ReplayBuffer
from Network2 import VanillaPolicyGradientNetwork
#from CriticNetwork import CriticNetwork
#from OU import OU
import timeit
import csv



try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass



import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from carla_env2 import CarlaEnv

try:
    print(sys.path)
    sys.path.append(glob.glob('../'))
    print(sys.path)
except IndexError:
    pass



#def discount_rewards():

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main(train_indicator =1):


    tf.reset_default_graph()  # Clear the Tensorflow graph.
    agent_network = VanillaPolicyGradientNetwork(0.95, 8, 5, 128, 16,)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        #sess.run( tf.local_variables_initializer())
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        main2(sess, init, agent_network)



def main2(sess, init, agent_networkk):
# ==============================================================================
# -- Neural Network Setup ----------------------------------------------------
# ==============================================================================
    sess.run(init)




    state_dim = 8
    agents_number = 4
    action_dim_single  = 5
    
    gamma = 0.95
    LR = 0.1
    BATCH_SIZE = 10000
    TAU = 0.01
    episode_count = 1000
    step_count = 1000
    

    agent_network = agent_networkk



    options = np.hstack((-1.0, -0.5, 0.0, 0.5, 1.0))
    env = CarlaEnv()
    env.start2()


    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

# ==============================================================================
# -- Episodes and steps ----------------------------------------------------
# ==============================================================================

    for i in range(episode_count):
        if i != 0:
            env.kill()
        env.setup_car()
        s_t1 = (0,0,0,0,0,0,0,0)

        rewards_sum = 0
        feed_dict = {}
        s_t = np.hstack((0,0,0,0,0,0,0,0))

        episode_buff = []
        a_t = np.hstack((0.0,0.0,0.0,0.0))     
        for j in range(step_count):
             

            print(j)
            predicts_logits = sess.run(agent_network.output, feed_dict ={agent_network.state_in: [s_t]} )
            predicts = predicts_logits.reshape([4,5])
            p_t = []


            for k in range(predicts.shape[0]):

                maxq = np.random.choice(predicts[k], p=predicts[k])
                maxq = np.argmax(predicts[k] == maxq)
                m_i = maxq + k*5
                p_t.append(m_i)
                a_t[k] = options[maxq]

      
            s_t1, r_t, done = env.step(a_t, j)


            episode_buff.append([s_t, a_t, r_t, done, s_t1, p_t])


            s_t = s_t1

            if done:
                print("DONE")
                states2 = np.vstack([e[0] for e in episode_buff])
                actions = np.asarray([e[1] for e in episode_buff])
                rewards = np.asarray([e[2] for e in episode_buff])
                dones = np.asarray([e[3] for e in episode_buff])
                action_indexes = np.asarray([e[5] for e in episode_buff])
                disc_rew = discount_rewards(rewards, gamma)                
                feed_dict = {agent_network.reward_holder: disc_rew, agent_network.action_holder: action_indexes, agent_network.state_in: states2}
                total_rewards = np.sum(rewards)


                feed_dict2 = {agent_network.state_in: states2}
                agent_network.train(feed_dict,sess, gradBuffer, feed_dict, i)
                break







#if __name__ == '__main__':

main(1)
