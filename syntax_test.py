import numpy as np

a = np.array([1, 2 ,3])
a.shape = (3,1)
print(a)
Q = np.matrix('1 0 0; 0 1 0; 0 0 1')
print(Q)
x = np.transpose(a)
print(x.shape, Q.shape, a.shape)
y = np.dot(np.dot(x, Q), a)
y.shape = (1,)
print(y.shape)
print(y)


Q = np.diag([1, 2, 3])
print(Q)