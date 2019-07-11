import numpy as np

a = np.array([1, 1])
b = a
c = np.array([1, 2])
print(a == b)
print(a == c)

print((a == b).all())
print((a == c).all())


import logging

logging.basicConfig(level=logging.INFO)
logging.info('hello' * 3)
