import numpy as np

import gym
env = gym.make('Pendulum-v0')

'''
oservation: [cart_pos, cart_vel, pole_angle, pole_vel]
actions   : {left: 0, right: 1}
reward	  : fixed=1. should compute our own rewards
start     : random position withing +- 0.05
terminal  : angle > 12, pos > 2.4, episode length > 200
'''
env.reset()
for _ in range(1000):
    env.render()
    print(env.action_space)
    observation, reward, done, info = env.step([1]) # take a random action
    print(reward)
    if done:
    	break
