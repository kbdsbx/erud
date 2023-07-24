import numpy as np

import conv2d

x = np.array([(1. * (i + 1)) for i in range(144)], dtype=np.float32)
w = np.array([(-1. * (i + 1)) for i in range(24)], dtype=np.float32)
z = np.array([0. for i in range(60)], dtype=np.float32)
cache_x = np.array([0. for i in range(360)], dtype=np.float32)
dz = np.array([(0.01 * (i + 1)) for i in range(60)], dtype=np.float32)
dw = np.array([0 for i in range(24)], dtype=np.float32)
dx = np.array([0 for i in range(144)], dtype=np.float32)

conv2d.conv2d_fprop(x, w, z, cache_x, 2, 4, 6, 3, 3, 5, 2, 2, 2, 1)
print('Z :' + str(z))
conv2d.conv2d_bprop(cache_x, w, dz, dx, dw, 2, 4, 6, 3, 3, 5, 2, 2, 2, 1)
print('dX :' + str(dx))
print('dW :' + str(dw))

import max_pool

x = np.array([(1. * (i + 1)) for i in range(144)], dtype=np.float32)
z = np.array([0. for i in range(36)], dtype=np.float32)
cache_x = np.array([0 for i in range(144)], dtype=np.int32)
dz = np.array([(0.01 * (i + 1)) for i in range(36)], dtype=np.float32)
dx = np.array([0 for i in range(144)], dtype=np.float32)

max_pool.max_pool_fprop(x, z, cache_x, 2, 4, 6, 2, 3, 2, 2, 3, 2)
print('Z: ' + str(z))

max_pool.max_pool_bprop(cache_x, dz, dx, 2, 4, 6, 2, 3, 2, 2, 3, 2)
print('dX: ' + str(dx))
