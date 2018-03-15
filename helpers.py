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

from scipy.integrate import ode
import models# , trajectories
import logging
logging.basicConfig(filename='debugging.log',level=logging.DEBUG)



def state_from_model(model):
    # returns np array for state vector:
    return state
def bound_actions(actions):
    for i in range(4):
        if actions[0][i] > 1:
            actions[0][i] = 1
        elif actions[0][i] < 0:
            actions[0][i] = 0
    return actions
def compute_desired_u(quad_model, actions):
    # actions are commands to each rotor, commands are between 0 and 1
    # scale them to 1100 to 8600 rpm, then convert motor speeds to torques
    omegas = np.zeros((1, 4))
    # bound the noisy actions:
    bounded_actions = bound_actions(actions)
    for i in range(4):
        omegas[0][i] = 1100 + 7000*bounded_actions[0][i]
    # compute torques and moments from 
    # print(omegas)
    u1 = quad_model.ct * (omegas[0][0]**2 + omegas[0][1]**2 +omegas[0][2]**2 +omegas[0][3]**2)
    u2 = quad_model.ct * quad_model.l*(omegas[0][3]**2 - omegas[0][1]**2)
    u3 = quad_model.ct * quad_model.l*(omegas[0][2]**2 - omegas[0][0]**2)
    u4 = quad_model.cq * (omegas[0][0] - omegas[0][1] + omegas[0][2] - omegas[0][3])
    return [u1, u2, u3, u4]

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

def create_critic(in_shape, n_actions=4):
    initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    critic_state = Sequential()
    critic_state.add(BatchNormalization(input_shape=in_shape, axis=1))
    critic_state.add(Dense(400, kernel_initializer=initializer, bias_initializer="zeros", name='L1'))
    critic_state.add(BatchNormalization(axis=1, name='L2'))
    critic_state.add(Activation("relu", name='3'))
    critic_state.add(Dense(150, input_shape=in_shape, kernel_initializer=initializer, bias_initializer="zeros", name='L4'))
    critic_state.add(BatchNormalization(axis=1, name='L5'))
    critic_state.add(Activation("relu", name='L6'))

    critic_action = Sequential()
    critic_action.add(BatchNormalization(input_shape=[n_actions], axis=1))
    critic_action.add(Dense(150, kernel_initializer=initializer, bias_initializer="zeros", name='L7'))
    critic_action.add(BatchNormalization(axis=1, name='L8'))
    critic_action.add(Activation("relu", name='L9'))
    
    # concatinate models:
    merged = Merge([critic_state, critic_action], mode='concat' , name='L10')

    # critic = Sequential(Concatenate([critic_conv, critic_action]))
    critic_full = Sequential()
    critic_full.add(merged)
    critic_full.add(Dense(1, activation="linear", kernel_initializer=initializer, bias_initializer="zeros", name='output'))

    critic = Model(input = [critic_state.input, critic_action.input], output= critic_full.output)
    critic.compile(loss='mse', optimizer=Adam(lr=0.001))
    # critic.summary()
    return critic, critic_action.input, critic_state.input

def create_actor(in_shape, n_actions=4):
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
    initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    actor.add(BatchNormalization(input_shape=in_shape, axis=1))
    actor.add(Dense(400, kernel_initializer=initializer, bias_initializer="zeros", name='L1'))
    actor.add(BatchNormalization(axis=1, name='L2'))
    actor.add(Activation("relu", name='L3'))
    actor.add(Dense(300, kernel_initializer=initializer, bias_initializer="zeros", name='L4'))
    actor.add(BatchNormalization(axis=1, name='L5'))
    actor.add(Activation("relu", name='L6'))
    actor.add(Dense(n_actions, activation="sigmoid", kernel_initializer=initializer, bias_initializer="zeros", name='L10'))
    # actor.summary()
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
    def __init__(self, mu=0, sigma=0.2, theta=1.0, dt=1e-2, x0=None):
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

class quad_env():
    def __init__(self, init_r=(0.0,0.0,0.0,0.0,0.0,0.0), dt=0.01, tol=(0.05,0.05,0.05,0.04), r_des=(0.0,0.0,0.0,0.0), t1=10):
        # 
        #self.post = q

        # time settings
        self.t0 = 0             # initial time
        self.t1 = t1            # final time
        self.dt = dt            # time step for integration
        self.STATE_LENGTH = 16
        self.r_des = r_des
        self.tol = tol
        # dynamical model (from baldr forked repo)
        self.model = models.NonLinear3()

        # initial values
        x0     = init_r[0];   dx0     = 0.0
        y0     = init_r[1];   dy0     = 0.0
        z0     = init_r[2];   dz0     = 0.0
        phi0   = init_r[3];   dphi0   = 0.0
        theta0 = init_r[4];   dtheta0 = 0.0
        psi0   = init_r[5];   dpsi0   = 0.0
        self.init_state = np.array([    x0,     dx0, 
                        y0,     dy0, 
                        z0,     dz0,
                        phi0,       dphi0,
                        theta0,     dtheta0,
                        phi0,       dphi0       ])


        self.integrator = ode(self.model.integration_loop)
        # https://itp.tugraz.at/~ert//blog/2014/06/02/equivalent-ode-integrators-in-matlab-and-scipy/
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.integrate.ode.html
        # set integrator equivalent to ode45 of matlab
        self.integrator.set_integrator('dopri5')
        self.integrator.set_initial_value(self.init_state, self.t0)

        # logging data in a dictionary:
        self.data = dict(   x=[],       y=[],       z=[], 
                    xr=[],      yr=[],      zr=[],      psir=[],
                    phi=[],     theta=[],   psi=[],     t=[],
                    u1=[],      u2=[],      u3=[],      u4=[],
                    u1r=[],     u2r=[],     u3r=[],     u4r=[],
                    omega1=[],  omega2=[],  omega3=[],  omega4=[]   )   


    def set_integrator(self):
        self.integrator = ode(self.model.integration_loop)
        # https://itp.tugraz.at/~ert//blog/2014/06/02/equivalent-ode-integrators-in-matlab-and-scipy/
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.integrate.ode.html
        # set integrator equivalent to ode45 of matlab
        self.integrator.set_integrator('dopri5')
        self.integrator.set_initial_value(self.init_state, self.t0)

    def step(self, u):
        self.model.update(u)
        self.integrator.integrate(self.integrator.t + self.dt)
        # omega = self.model.get_omega()
        # return new state:
        state = self.get_state()
        return state


    def get_state(self):
        ''' state vector: X
        x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, (x-x_m), (y-y_m), (z-z_m), (psi-psi_m)
        0, 1, 2,   3 ,  4 ,   5 ,   6 ,   7 ,   8 ,   9  ,     10   ,   11  ,    12  ,    13   ,   14  ,      15 
        where _m denotes a command signal.
        '''
        state = np.zeros((1, self.STATE_LENGTH))
        state[0][0]  = self.integrator.y[0]
        state[0][1]  = self.integrator.y[2]
        state[0][2]  = self.integrator.y[4]
        state[0][3]  = self.integrator.y[1]
        state[0][4]  = self.integrator.y[3]
        state[0][5]  = self.integrator.y[5]
        state[0][6]  = self.integrator.y[6]#%(2*np.pi)
        state[0][7]  = self.integrator.y[8]#%(2*np.pi)
        state[0][8]  = self.integrator.y[10]#%(2*np.pi)
        state[0][9]  = self.integrator.y[7]
        state[0][10] = self.integrator.y[9]
        state[0][11] = self.integrator.y[11]
        state[0][12] = self.integrator.y[0] - self.r_des[0]
        state[0][13] = self.integrator.y[2] - self.r_des[1]
        state[0][14] = self.integrator.y[4] - self.r_des[2]
        state[0][15] = self.integrator.y[10] - self.r_des[3]

        return state

    def check_end(self, state):
        # check if the quad left the allowed area of navigation (a cube centered at 0,0,0, with width=16)
        for i in range(3):
            if abs(state[0][i]) > 8:
                print("state " + str(i) +" failed first")
                print(state[0][0:3])
                return True, -1000
        if self.integrator.t >= self.t1:
            return True, -500

        for i in range(4):
            if abs(state[0][12+i]) > self.tol[i]:
                return False, 0
        return True, 100



# from yanpanlau github
class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)