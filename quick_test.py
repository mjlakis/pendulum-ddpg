"""
ship trainer using DDPG
ship actions are:
    0. angle            : angle in degrees, between 0 and 360
    1. speed
    2. dock
    3. undock
state: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot]
"""
from scipy.integrate import ode
import models#, trajectories, controllers

# keras stuff here:
import pickle
import numpy as np
import random
import collections
import helpers
import tensorflow as tf
from sklearn.utils import shuffle
# Initial Setup for Keras
import tensorflow as tf
import numpy as np
from keras import initializers
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Input, concatenate, Concatenate, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K


import time
from helpers import *
import copy
from scipy.stats import truncnorm
from collections import deque

MINIBATCH_SIZE         = 32
N_REPLAY_MEMORY_LENGTH = 50000
NUM_EPISODES           = 200
DISCOUNT_FACTOR        = 0.99
LEARNING_RATE          = 0.00025 # CURRENTLY USING ADAM OPTIMIZER
REPLAY_START_SIZE      = 100
SAVE_SIZE              = 50


# Fixed game parameters:
SHIP_RADIUS     = 0.5       # NOT USED
STATE_LENGTH    = 16
in_shape        = [STATE_LENGTH]
N_ACTIONS       = 4

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

#Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)


# initialize networks
# actor and its target
actor, actor_state_layer, actor_weights                         = create_actor(in_shape)
actor_target, actor_target_weights, actor_target_state_layer    = create_actor(in_shape)
action_gradient   = tf.placeholder(tf.float32,[None, N_ACTIONS])                            # This is a placeholder for gradients from critic
params_grad = tf.gradients(actor.output, actor_weights, -action_gradient)
grads = zip(params_grad, actor_weights)
actor_optimizer = tf.train.AdamOptimizer(learning_rate = 0.00025).apply_gradients(grads)


# critic and its target
critic, critic_actions_layer, critic_state_layer = create_critic(in_shape, N_ACTIONS)
critic_target, dont_care1, dont_care2 = create_critic(in_shape, N_ACTIONS)
critic_action_gradients = tf.gradients(critic.output, critic_actions_layer)


# load net weights
# actor.load_weights("actor.h5")
# critic.load_weights("critic.h5")
# actor_target.load_weights("actor_target.h5")
# critic_target.load_weights("critic_target.h5")
# print("loaded all weights successfully")

# load experience buffer
D = deque() 
pickle_out = open("experience_replay_dict.pickle","wb")
pickle.dump(D, pickle_out)



# initialize some variables:
default_state     = np.zeros((1, STATE_LENGTH))
state_batch       = np.zeros((MINIBATCH_SIZE, STATE_LENGTH))
nest_state_batch  = np.zeros((MINIBATCH_SIZE, STATE_LENGTH))
action_batch      = np.zeros((MINIBATCH_SIZE, N_ACTIONS))
reward_batch      = np.zeros((MINIBATCH_SIZE, 1))
is_end_batch      = np.zeros((MINIBATCH_SIZE, 1))
y_batch           = np.zeros((MINIBATCH_SIZE, 1))



sess.run(tf.global_variables_initializer())
# actor._make_train_function()
default_state     = np.zeros((1, STATE_LENGTH))
actions = np.array([0, 0, 0, 0]).reshape((1,4))

print(sess.run(critic_action_gradients, feed_dict={
        critic_state_layer: default_state,
        critic_actions_layer: actions
    })[0])
print(actor.predict(default_state, batch_size=1))

experience_replay_dict = deque() 
pickle_out = open("experience_replay_dict.pickle","wb")
pickle.dump(experience_replay_dict, pickle_out)


print("Now we save model")

actor_json = actor.to_json()
with open("actor.json", "w") as outfile:
    outfile.write(actor_json)
actor.save_weights("actor.h5", overwrite=True)

actor_target_json = actor_target.to_json()
with open("actor_target.json", "w") as outfile:
    outfile.write(actor_target_json)
actor_target.save_weights("actor_target.h5", overwrite=True)

critic_json = critic.to_json()
with open("critic.json", "w") as outfile:
    outfile.write(critic_json)
critic.save_weights("critic.h5", overwrite=True)

critic_target_json = critic_target.to_json()
with open("critic_target.json", "w") as outfile:
    outfile.write(critic_target_json)
critic_target.save_weights("critic_target.h5", overwrite=True)



'''
from helpers import OrnsteinUhlenbeckActionNoise

noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(4), x0=0.4)

for i in range(300):
    delta = 0.4
    print(truncnorm.rvs(-delta, delta, size=1))

'''

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

