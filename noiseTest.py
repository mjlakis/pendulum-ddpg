import numpy as np




def UONoise():
    theta = 0.15
    sigma = 0.2
    state = 0
    while True:
        yield state
        state += -theta*state+sigma*np.random.randn()


OUn = UONoise()
for i in range(100):
    print("OU(): ", next(OUn))