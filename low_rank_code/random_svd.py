import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import time
def rSVD(X,r,q,p):
    # Step 1: Sample coloumn space of X with P matrix
    ny = X.shape[1]
    P = np.random.randn(ny,r + q)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)
    
    Q, R = np.linalg.qr(Z, mode='reduced')
    
    # Step 2: Compute SVD on projected Y = Q.T @ X
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    return U, S, VT

A = imread(r'image1.jpg')
X = np.mean(A, axis=2)

start = time.time()
U, S, VT = rSVD(X, 100, 0, 0)
sigma = S * np.identity(100) 
print(time.time() - start)
X_hat = (U @ sigma) @ VT

plt.imshow(X)
plt.show()
plt.imshow(X_hat)

start = time.time()
U0, S0, VT0 = np.linalg.svd(X, full_matrices=0)
print(time.time() - start)

sigma0 = S0 * np.identity(S0.shape[0]) 
plt.imshow((U0 @ sigma0) @ VT0)

X0 = (U0 @ sigma0) @ VT0
np.sum(X - X_hat)


print('Original size', X.shape)
print('Approximate size U', U.shape)
print('Approximate size Sigma', sigma.shape)
print('Approximate size V.T', VT.shape)





