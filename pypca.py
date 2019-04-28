'''
Name: pynb
By: CDoyle
Implementation of PCA
'''

import numpy as np

def pca(X, n_comp = 50):
    '''
    Returns eigenvalues S, eigenvectors U of the covariance matrix sigma
    and the projected data
    '''
    # normalise patches
    X_norm = (X-np.mean(X))/np.std(X)

    # obtain eigenvalues and eigenvectors of sigma
        # U is a matrix of eigenvectors
        # S is a vector of eigenvalues
    [S,U] = np.linalg.eig(np.cov(X_norm))
    p = np.size(U,axis=1)

    # sort eigenvalues and eigenvectors
    idx = np.argsort(-S)
    S = S[idx]
    U = U[:,idx]

    # take only the n_components which account for most variance
    if n_comp < p and n_comp >=0:
        #S = S[:n_components]
        U = U[:,range(n_comp)].real

    # projection into lower dimensional space
    Xpc = np.dot(U.T,X_norm)
    return Xpc,U,S

def whiten(X,epsilon):
    '''
    Returns a PCA whitened image
    '''
    # normalise patches
    X_norm = (X-np.mean(X.T,axis=1)).T
    # obtain eigenvalues and eigenvectors of sigma
        # U: is a matrix of eigenvectors
        # S: is a vector of eigenvalues
    [S,U] = np.linalg.eig(np.cov(X_norm))
    # Compute PCA matrix
    PCAMatrix = np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)
    return np.dot(PCAMatrix, X_norm), U, S

def reconstruct(U,Xpc,X):
    X_norm = (X-np.mean(X.T,axis=1)).T
    return np.dot(U,Xpc).T + np.mean(X,axis=0)

def pca_dataset(dataset,components):
    data = []
    for i in dataset:
        Xpc,U,S = pca(i, components)
        data.append(Xpc.flatten())
    return np.array(data)
