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
render                 = 0

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

pickle_in = open("experience_replay_dict.pickle","rb")
D = deque(pickle.load(pickle_in))
len_d = len(D)


''' state vector: X
x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, (x-x_m), (y-y_m), (z-z_m), (psi-psi_m)
0, 1, 2,   3 ,  4 ,   5 ,   6 ,   7 ,   8 ,   9  ,     10   ,   11  ,    12  ,    13   ,   14  ,      15 
where _m denotes a command signal.
'''
'''cost (reward) function:
cost = integral(x'Qx + u'Ru + c)
reward = -cost
'''
'''actions vector
thrust for each motor
'''

# initialize some variables:
##############################################
state             = np.zeros((1, STATE_LENGTH))
state_batch       = np.zeros((MINIBATCH_SIZE, STATE_LENGTH))
next_state_batch  = np.zeros((MINIBATCH_SIZE, STATE_LENGTH))
action_batch      = np.zeros((MINIBATCH_SIZE, N_ACTIONS))
reward_batch      = np.zeros((MINIBATCH_SIZE, 1))
is_end_batch      = np.zeros((MINIBATCH_SIZE, 1))
y_batch           = np.zeros((MINIBATCH_SIZE, 1))




# Noise function
###############################################

def UONoise():
    theta = 0.15
    sigma = 0.2
    state = 0
    while True:
        yield state
        state += -theta*state+sigma*np.random.randn()


def noise_decay(ep_count):
    decay = ep_count/150
    if decay > 1:
        decay = 1
    scale = 1 - decay
    return scale
env = env = gym.make('Pendulum-v0')
#env.reset()
#O-H noise process:


OU = OU()
OUn = UONoise()
# Training:
#################################
#################################
episode_rewards = 0
ep = 0
loss = 0
is_end = False
while ep < NUM_EPISODES:
    # initialize new episode variables
    print("New episode: " + str(ep))
    episode_rewards = 0
    loss = 0
    OUn = UONoise()
    noise_scale = noise_decay(ep)
    print("noise scale: ", noise_scale)
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
        for i in range(N_ACTIONS):
            #actions[0][i] += max(noise_scale, 0)*OU.function(actions[0][i], 0, 0.3, 0.3)
            c = max(noise_scale, 0)
            actions[0][i] = actions[0][i] + next(OUn)
        actions = bound_actions(actions, N_ACTIONS)


        ## Find new state, reward, is_end, then compute episode rewards
        new_observation, reward, is_end, info = env.step(actions[0])
        new_state = np.array(new_observation).reshape((1,3))
        episode_rewards += (DISCOUNT_FACTOR**step_count) *reward

        ## Append data to experience replay, update current state(to be used next iter)
        D.append((state, actions/2, reward, new_state, is_end))
        state = np.copy(new_state)


        ## Maintain the length of the buffer
        while len(D) > N_REPLAY_MEMORY_LENGTH:
                    D.popleft()

        # Experience replay
        if len(D) > 2000:#10*MINIBATCH_SIZE:
            minibatch = np.array(random.sample(D, MINIBATCH_SIZE))
            for i in range(MINIBATCH_SIZE):
                state_batch[i]        = minibatch[i][0]
                action_batch[i]       = minibatch[i][1]
                reward_batch[i]       = minibatch[i][2]
                next_state_batch[i]   = minibatch[i][3]
                is_end_batch[i]       = minibatch[i][4]

            next_state_actions_batch        = actor_target.predict(next_state_batch, batch_size=MINIBATCH_SIZE)
            future_reward_predictions_batch = critic_target.predict([next_state_batch, next_state_actions_batch], batch_size=MINIBATCH_SIZE)

            '''
            Batch                                :    Dimensions
            state_batch                          :   (MINIBATCH_SIZE, STATE_LENGTH)
            action_batch                         :   (MINIBATCH_SIZE, N_ACTIONS)
            nest_state_batch                     :   (MINIBATCH_SIZE, STATE_LENGTH)
            is_end_batch                         :   (MINIBATCH_SIZE, 1)
            next_state_actions_batch             :   (MINIBATCH_SIZE, N_ACTIONS)
            future_reward_predictions_batch      :   (MINIBATCH_SIZE, N_ACTIONS)
            '''
            for i in range(MINIBATCH_SIZE):
                # for the particular action taken, use the obtained reward to correct prediction function
                if is_end_batch[i]:
                    y_batch[i] = reward_batch[i]
                else:
                    y_batch[i] = reward_batch[i] + DISCOUNT_FACTOR * future_reward_predictions_batch[i]

            
            ## Training steps:
            loss += critic.train_on_batch([state_batch, action_batch], y_batch)
            a_for_grads = actor.predict(state_batch)      # this step is different from ben yoa's blog
            critic_action_gradients_batch = sess.run(critic_action_gradients, feed_dict={
                                                    critic_state_layer: state_batch,
                                                    critic_actions_layer: a_for_grads#action_batch
                                                    })[0]
            # print(critic_action_gradients_batch)
            # print("new gradients")
            # train actor network:
            sess.run(actor_optimizer, feed_dict={
                        actor_state_layer: state_batch,
                        action_gradient: critic_action_gradients_batch
                        })
            # update target networks:
            target_train(critic, critic_target, tau=0.001)
            target_train(actor, actor_target, tau=0.001)

            # print(step_count)
        step_count += 1

    # print and log some data
    print("Episode ended")
    print("loss: ", loss)
    print("total rewards: ", episode_rewards)
    print("end reward is: ", end_reward)
    print("length of experience replay buffer: ", len(D))
    print(".........................")
    print()
    ep += 1
    # print(ep)
    if ep%SAVE_SIZE == 0:
        pickle_out = open("experience_replay_dict.pickle","wb")
        pickle.dump(D, pickle_out)
        critic.save_weights("critic.h5", overwrite=True)
        critic_target.save_weights("critic_target.h5", overwrite=True)
        actor.save_weights("actor.h5", overwrite=True)
        actor_target.save_weights("actor_target.h5", overwrite=True)
