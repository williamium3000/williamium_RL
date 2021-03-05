import numpy as np
states = {1:1, 2:1, 3:1}
a, b = np.array(list(zip(*states.items())))
print(a)
print(b)