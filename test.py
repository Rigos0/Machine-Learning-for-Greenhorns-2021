import numpy as np
from copy import deepcopy
betas = np.array([0,0])
alfa = deepcopy(betas)
betas[0] = 1
print(betas, alfa)