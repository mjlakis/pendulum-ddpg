"""
ship trainer using DDPG
ship actions are:
state: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot]
"""
from scipy.integrate import ode
import models# , trajectories, controllers

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


MINIBATCH_SIZE         = 64 
N_REPLAY_MEMORY_LENGTH = 100000
NUM_EPISODES           = 2000
DISCOUNT_FACTOR        = 0.99
LEARNING_RATE          = 0.00025 # CURRENTLY USING ADAM OPTIMIZER
REPLAY_START_SIZE      = 100
SAVE_SIZE              = 20

# Fixed game parameters:
STATE_LENGTH    = 16
in_shape        = [STATE_LENGTH]
N_ACTIONS       = 4

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
state             = np.zeros((1, STATE_LENGTH))
state_batch       = np.zeros((MINIBATCH_SIZE, STATE_LENGTH))
next_state_batch  = np.zeros((MINIBATCH_SIZE, STATE_LENGTH))
action_batch      = np.zeros((MINIBATCH_SIZE, N_ACTIONS))
reward_batch      = np.zeros((MINIBATCH_SIZE, 1))
is_end_batch      = np.zeros((MINIBATCH_SIZE, 1))
y_batch           = np.zeros((MINIBATCH_SIZE, 1))

# cost matrices:
#            x     y     z                 theta    t_dot      x-x_m
# taken from my previous adaptive control porject, probably should tune again
#Q = 10*np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 ,1 ,1])
Q = np.diag([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, 1, 1, 1, 1, 1, 2, 2 ,2 ,2])
R = 0.01*np.diag([1,1,1,1])
# cost related to time
c = 0.1 


def rand_pos():
    x     = np.random.uniform(-5,5,1)
    y     = np.random.uniform(-5,5,1)
    z     = np.random.uniform(-5,5,1)
    phi   = np.random.uniform(-np.pi/10,np.pi/10,1)
    theta = np.random.uniform(-np.pi/10,np.pi/10,1)
    psi   = np.random.uniform(-np.pi/10,np.pi/10,1)
    return (x,y,z,phi,theta,psi)

def noise_decay(ep_count):
    decay = ep_count/50
    if decay > 1:
        decay = 1
    scale = 1.05 - decay
    return scale
env = []
#O-H noise process:


OU = OU()
 
episode_rewards = 0
ep = 0
loss = 0
is_end = False
while ep < NUM_EPISODES:
    #logging.info(".............................................")
    print("New episode: " + str(ep))
    #logging.info("New episode: " + str(ep))
    episode_rewards = 0
    loss = 0
    # restart environment with new initial position and orientation
    # x, y, z, phi, theta psi
    rand_r_init = (0,0,-1,0,0,0) # rand_pos()
    #sign = -1*(np.random.randn(1) > 0) +1*(np.random.randn(1) <= 0)

    noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(4), sigma=0.2*np.ones(4), x0=0.3)
    #logging.info("Initial position: " + str(rand_r_init))
    env = quad_env(init_r=rand_r_init) 
    state = env.get_state()
    #logging.info("Initial state: " + str(state))
    step_count = 0
    is_end = False
    end_reward = 0
    while not is_end:
        #logging.info("Raw_actions, noise, noisy actions")
        actions  = np.zeros((1, N_ACTIONS))
        
        OU_noise = np.zeros((1, N_ACTIONS))
        OU_noise = noise()
        actions  = actor.predict(state) #+ noise_decay(ep)*noise()
        # print(actions)
        #logging.info(actions)
        for i in range(N_ACTIONS):
            #OU_noise[0][i] = noise_decay(ep)*noise()[i]#OU.function(actions[0][i], 0, 1, 0.1)# noise_decay(ep)*truncnorm.rvs(-0.4, 0.4, size=1)
            actions[0][i] += OU_noise[i]
        #print(OU_noise)
        #logging.info(OU_noise)
        #logging.info(actions)

        u = compute_desired_u(env.model, actions)
        # print(state)
        # print(u)
        #logging.info("Action for new state: " + str(u))
        new_state = env.step(u)
        #logging.info("New state: " + str(new_state))
        is_end, end_reward = env.check_end(new_state)
        reward  = compute_rewards(new_state, np.array(u), Q, R, c) + end_reward
        #logging.info("Reward obtained: " + str(reward))
        # logg rewards:
        episode_rewards += (DISCOUNT_FACTOR**step_count) *reward
        # append data to experience replay
        D.append((state, actions, reward, new_state, is_end))
        state = np.copy(new_state)

        while len(D) > N_REPLAY_MEMORY_LENGTH:
                    D.popleft()

        # experience replay
        if len(D) > MINIBATCH_SIZE:
            minibatch = np.array(random.sample(D, MINIBATCH_SIZE))
            for i in range(MINIBATCH_SIZE):
                state_batch[i]        = minibatch[i][0]
                #print(is_end_batch[i])
                action_batch[i]       = minibatch[i][1]
                reward_batch[i]       = minibatch[i][2]
                next_state_batch[i]   = minibatch[i][3]
                is_end_batch[i]       = minibatch[i][4]
                #print(is_end_batch.shape)
        
            #print(state_batch.shape)
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
                    # print(reward_batch[i])
                else:
                    y_batch[i] = reward_batch[i] + DISCOUNT_FACTOR * future_reward_predictions_batch[i]
                    # print(future_reward_predictions_batch[i])
                    # print("...")
            
            # training steps:
            loss += critic.train_on_batch([state_batch, action_batch], y_batch)
            a_for_grads = actor.predict(state_batch)      # this step is different from ben yoa's blog
            #print(a_for_grads.shape)
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
    print(env.integrator.t)
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
