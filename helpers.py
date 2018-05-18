# main file to train actor and critic
#import keras stuff here
import tensorflow as tf
import numpy as np
from keras import initializers
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Input, concatenate, Concatenate, merge, Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
K.set_learning_phase(1)

import logging
logging.basicConfig(filename='debugging.log',level=logging.DEBUG)



def state_from_model(model):
    # returns np array for state vector:
    return state
def bound_actions(actions, n_actions):
    for i in range(n_actions):
        if actions[0][i] > 2:
            actions[0][i] = 2
        elif actions[0][i] < -2:
            actions[0][i] = -2
    return actions


def compute_rewards(state, actions, Q, R, c):
    '''
    state  : (1, STATE_LENGTH)
    actions: (1, N_ACTIONS)
    Q      : (STATE_LENGTH, STATE_LENGTH)
    R      : (N_ACTIONS, N_ACTIONS)
    '''
    r1     = np.dot(np.dot(state, Q), np.transpose(state))
    #logging.info(r1)
    r2     = np.dot(np.dot(actions, R), np.transpose(actions))
    reward = - r1 -r2 - c
    # print("reward is:")
    # print(reward)
    return reward/1000

def create_critic(in_shape, n_actions=1):
    #initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    weight_initializer = initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
    bias_initializer   = initializers.RandomUniform(minval=-0.0003, maxval=0.0003, seed=None)

    critic_state = Sequential()
    #critic_state.add(BatchNormalization(input_shape=in_shape, axis=1))
    
    critic_state.add(Dense(32,input_shape = in_shape, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, name='L1'))
    #critic_state.add(BatchNormalization(axis=-1, name='L2'))
    critic_state.add(Activation("elu", name='3'))

    ''' Removing a layer
    critic_state.add(Dense(64, input_shape=in_shape, kernel_initializer=initializer, bias_initializer="zeros", name='L4'))
    #critic_state.add(BatchNormalization(axis=1, name='L5'))
    critic_state.add(Activation("relu", name='L6'))
    '''

    critic_action = Sequential()
    #critic_action.add(BatchNormalization(input_shape=[n_actions], axis=1))
    critic_action.add(Dense(32, input_shape = [n_actions], kernel_initializer=weight_initializer, bias_initializer=bias_initializer, name='L7'))
    #critic_action.add(BatchNormalization(axis=-1, name='L8'))
    critic_action.add(Activation("elu", name='L9'))
    
    # concatinate models:
    merged = Merge([critic_state, critic_action], mode='concat' , name='L10')

    # critic = Sequential(Concatenate([critic_conv, critic_action]))
    critic_full = Sequential()
    critic_full.add(merged)
    critic_full.add(Dense(128, activation="elu", kernel_initializer=weight_initializer, bias_initializer=bias_initializer, name='L11'))
    #critic_full.add(BatchNormalization(axis=-1, name='L12'))
    critic_full.add(Dense(1, activation="linear", kernel_initializer=weight_initializer, bias_initializer=bias_initializer, name='output'))

    critic = Model(input = [critic_state.input, critic_action.input], output= critic_full.output)
    critic.compile(loss='mse', optimizer=Adam(lr=0.001))
    critic.summary()
    return critic, critic_action.input, critic_state.input

def create_actor(in_shape, n_actions=1):
    # custom_objects=None
    '''
    actor = Sequential()
    # actor.add(Input(shape=in_shape))
    initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    actor.add(BatchNormalization(input_shape=in_shape, axis=-1))
    actor.add(Dense(200, kernel_initializer=initializer, bias_initializer="zeros", name='L1'))
    actor.add(BatchNormalization(axis=-1, name='L2'))
    actor.add(Activation("relu", name='L3'))
    actor.add(Dense(200, kernel_initializer=initializer, bias_initializer="zeros", name='L4'))
    actor.add(BatchNormalization(axis=-1, name='L5'))
    actor.add(Activation("relu", name='L6'))
    actor.add(Dense(80, kernel_initializer=initializer, bias_initializer="zeros", name='L7'))
    actor.add(BatchNormalization(axis=-1, name='L8'))
    actor.add(Activation("relu", name='L9'))  
    actor.add(Dense(n_actions, activation="sigmoid", kernel_initializer=initializer, bias_initializer="zeros", name='L10'))
    # actor.summary()
    return actor, actor.input, actor.trainable_weights
    '''
    actor = Sequential()
    # actor.add(Input(shape=in_shape))
    #initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    weight_initializer = initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
    bias_initializer   = initializers.RandomUniform(minval=-0.0003, maxval=0.0003, seed=None)

    #actor.add(BatchNormalization(input_shape=in_shape, axis=1))
    actor.add(Dense(32, input_shape = in_shape, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, name='L1'))
    #actor.add(BatchNormalization(axis=-1, name='L2'))
    actor.add(Activation("elu", name='L3'))
    actor.add(Dense(64, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, name='L4'))
    #actor.add(BatchNormalization(axis=-1, name='L5'))
    actor.add(Activation("elu", name='L6'))
    actor.add(Dense(n_actions, activation="tanh", kernel_initializer=weight_initializer, bias_initializer=bias_initializer, name='L10'))
    actor.summary()
    return actor, actor.input, actor.trainable_weights

def train_actor(sess, actor, actor_optimizer, actor_state_layer, action_gradient, state_batch, action_grads_batch):
     sess.run(actor_optimizer, feed_dict={
            actor_state_layer: state_batch,
            action_gradient: action_grads_batch
        })

def target_train(model, target_model, tau=0.001):
    model_weights = model.get_weights()
    target_model_weights = target_model.get_weights()
    for i in range(len(model_weights)):
        target_model_weights[i] = tau* model_weights[i] + (1-tau)*target_model_weights[i]
    target_model.set_weights(target_model_weights)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def epsilon(i, final_step=100000, final_exploration=0):
    eps = 1 - i/100000
    return eps
'''
def make_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable
        '''



# from yanpanlau github
class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)