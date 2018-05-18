"""
ship trainer using DDPG
ship actions are:
state: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot]
"""
import gym

# keras stuff here:
import pickle
import numpy as np
import random
import tensorflow as tf
from sklearn.utils import shuffle
# Initial Setup for Keras
import numpy as np
from keras import initializers
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Input, concatenate, Concatenate, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import json


import time
from helpers import *
import copy
from scipy.stats import truncnorm
from collections import deque

import logging
logging.basicConfig(filename='debugging.log',level=logging.DEBUG)
logging.info('tensorflow session configured')

#Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)


MINIBATCH_SIZE         = 256 
N_REPLAY_MEMORY_LENGTH = 100000
NUM_EPISODES           = 200
DISCOUNT_FACTOR        = 0.99
LEARNING_RATE          = 0.00025 # CURRENTLY USING ADAM OPTIMIZER
REPLAY_START_SIZE      = 100
SAVE_SIZE              = 20
render                 = 1

# Fixed game parameters:
STATE_LENGTH    = 3
in_shape        = [STATE_LENGTH]
N_ACTIONS       = 1

# initialize networks
# actor and its target
actor, actor_state_layer, actor_weights                         = create_actor(in_shape)
actor_target, actor_target_weights, actor_target_state_layer    = create_actor(in_shape)
action_gradient   = tf.placeholder(tf.float32,[None, N_ACTIONS])                            # This is a placeholder for gradients from critic
params_grad = tf.gradients(actor.output, actor_weights, -action_gradient)
grads = zip(params_grad, actor_weights)
actor_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).apply_gradients(grads)


# critic and its target
critic, critic_actions_layer, critic_state_layer = create_critic(in_shape, N_ACTIONS)
critic_target, dont_care1, dont_care2 = create_critic(in_shape, N_ACTIONS)
critic_action_gradients = tf.gradients(critic.output, critic_actions_layer)



# load net weights
actor.load_weights("actor.h5")
critic.load_weights("critic.h5")
actor_target.load_weights("actor.h5")
critic_target.load_weights("critic.h5")
print("loaded all weights successfully")


# initialize some variables:
##############################################
state             = np.zeros((1, STATE_LENGTH))
state_batch       = np.zeros((MINIBATCH_SIZE, STATE_LENGTH))
next_state_batch  = np.zeros((MINIBATCH_SIZE, STATE_LENGTH))
action_batch      = np.zeros((MINIBATCH_SIZE, N_ACTIONS))
reward_batch      = np.zeros((MINIBATCH_SIZE, 1))
is_end_batch      = np.zeros((MINIBATCH_SIZE, 1))
y_batch           = np.zeros((MINIBATCH_SIZE, 1))


env = env = gym.make('Pendulum-v0')

# Training:
#################################
#################################
episode_rewards = 0
ep = 0
loss = 0
is_end = False
sum_rewards = 0
while ep < 100:
    # initialize new episode variables
    print("New episode: " + str(ep))
    episode_rewards = 0
    loss = 0
    observation = env.reset()
    print("observation: ", observation)
    state = np.array(observation).reshape((1,3))
    #logging.info("Initial state: " + str(state))
    step_count = 0
    is_end = False
    end_reward = 0
    while not is_end:
        ## Render
        if render == 1:
            env.render()

        ## Find actions and add noise then bound
        actions  = np.zeros((1, N_ACTIONS))
        actions  = 2*actor.predict(state) 
        #print(actions)
        actions = bound_actions(actions, N_ACTIONS)

        ## Find new state, reward, is_end, then compute episode rewards
        new_observation, reward, is_end, info = env.step(actions[0])
        episode_rewards += (DISCOUNT_FACTOR**step_count) *reward
        state = np.array(new_observation).reshape((1,3))


            # print(step_count)
        step_count += 1

    # print and log some data
    print("Episode ended")
    print("loss: ", loss)
    print("total rewards: ", episode_rewards)
    print("end reward is: ", end_reward)
    print(".........................")
    print()
    ep += 1
    sum_rewards += episode_rewards/100

print(sum_rewards)