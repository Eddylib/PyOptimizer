import numpy as np
gth = np.load('__solveLinearSystemByMatrixInverse_h.npy')
gtb = np.load('__solveLinearSystemByMatrixInverse_b.npy')


outh = np.load('__solveLinearSystemByReconstructedMatrixInverse_h.npy')
outb = np.load('__solveLinearSystemByReconstructedMatrixInverse_b.npy')

import matplotlib.pyplot as plt
plt.imshow(np.abs(outh-gth)>0.00001, vmin=0,vmax=1)
plt.show()
print(np.max(np.abs(gth-outh)))

print(np.sum(np.abs(outb-gtb)))