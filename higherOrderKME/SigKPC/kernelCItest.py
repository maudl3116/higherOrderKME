
import numpy as np
import torch 
from higherOrderKME import sigkernel
from sklearn.metrics import pairwise_distances

def hsicclust( X, Y, Z, p=0, numCluster=10, eps=0.1, dyadic_order=0, static='rbf', sigma=1.):
    """Input: 
                    - X: torch tensor of shape (batch, length_X, dim_X),
                    - Y: torch tensor of shape (batch, length_Y, dim_Y)
                    - Z: torch tensor of shape (batch, length_Z, dim_Z)
                    - p: number of permutation of 1-alpha level test
                    - numCluster: number of clusters if we use kpc cluster permutation
                    - eps: normalization parameter
    """

    X = torch.tensor(X,dtype=torch.float64)
    Y = torch.tensor(Y,dtype=torch.float64)
    Z = torch.tensor(Z,dtype=torch.float64)

    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        X = X.cuda()
        Y = Y.cuda()
        Z = Z.cuda()

    assert p==0, "p>0 not implemented"

    n = X.shape[0] # number of samples
    H = torch.eye(n,device=X.device) - (1./n)*torch.ones((n,n),device=X.device)
    pval_perm = torch.zeros(p, device=X.device)


    H = torch.eye(n,dtype=torch.float64,device=X.device) - (1./n)*torch.ones((n,n),dtype=torch.float64, device=X.device)

    # create signature kernel gram matrices
    if static=='rbf':
        static_kernel = sigkernel.RBFKernel(sigma=sigma) #,add_time = X.shape[1]-1)
    else:
        static_kernel = sigkernel.LinearKernel() #add_time = X.shape[1]-1)

    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    K_X = signature_kernel.compute_Gram(X,X,sym=True)
    K_Y = signature_kernel.compute_Gram(Y,Y,sym=True)
    K_Z = signature_kernel.compute_Gram(Z,Z,sym=True)

    K = torch.matmul(H, torch.matmul(K_X, H))
    L = torch.matmul(H, torch.matmul(K_Y, H))
    M = torch.matmul(H, torch.matmul(K_Z, H))


    hsic_ = HSIC(K,L,M,eps)

    # TODO: permutation test

    # for i in range(p):  
        # TODO
        # pval_perm[i]  = HSIC(K,L,M,eps)

    if p > 0:
        pval = np.mean(np.append(pval_perm,hsic_)>=hsic_)  # the more there are pval_terms bigger than hsic, the higher the p-value
    else:
        pval = hsic_ # not a pvalue
    # print('pval',pval)
    return pval

def hsicclust_baseline( X, Y, Z, p=0, numCluster=10, eps=0.1, baseline='rbf',gamma=-1):
    """Input: 
                    - X: torch tensor of shape (batch, length_X, dim_X),
                    - Y: torch tensor of shape (batch, length_Y, dim_Y)
                    - Z: torch tensor of shape (batch, length_Z, dim_Z)
                    - p: number of permutation of 1-alpha level test
                    - numCluster: number of clusters if we use kpc cluster permutation
                    - eps: normalization parameter
    """
    nobs = X.shape[1]
    X = np.reshape(X,(X.shape[0],-1))
    Y = np.reshape(Y,(Y.shape[0],-1))
    Z = np.reshape(Z,(Z.shape[0],-1))

    assert p==0, "p>0 not implemented"

    n = X.shape[0] # number of samples

    if baseline=='rbf':
        K_X = RBF(X,nobs,gamma=gamma)
        K_Y = RBF(Y,nobs,gamma=gamma)
        K_Z = RBF(Z,nobs,gamma=gamma)
    elif baseline=='matern':
        K_X = Matern(X,nobs,gamma=gamma)
        K_Y = Matern(Y,nobs,gamma=gamma)
        K_Z = Matern(Z,nobs,gamma=gamma)

    K_X = torch.tensor(K_X,dtype=torch.float64)
    K_Y = torch.tensor(K_Y,dtype=torch.float64)
    K_Z = torch.tensor(K_Z,dtype=torch.float64)
    X = torch.tensor(X,dtype=torch.float64)
    Y = torch.tensor(Y,dtype=torch.float64)
    Z = torch.tensor(Z,dtype=torch.float64)

    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        X = X.cuda()
        Y = Y.cuda()
        Z = Z.cuda()
        K_X = K_X.cuda()
        K_Y = K_Y.cuda()
        K_Z = K_Z.cuda()
    
    pval_perm = torch.zeros(p, device=X.device)

    H = torch.eye(n,dtype=torch.float64,device=X.device) - (1./n)*torch.ones((n,n),dtype=torch.float64, device=X.device)
    K = torch.matmul(H, torch.matmul(K_X, H))
    L = torch.matmul(H, torch.matmul(K_Y, H))
    M = torch.matmul(H, torch.matmul(K_Z, H))


    hsic_ = HSIC(K,L,M,eps)

    # TODO: permutation test

    # for i in range(p):  
        # TODO
        # pval_perm[i]  = HSIC(K,L,M,eps)

    if p > 0:
        pval = np.mean(np.append(pval_perm,hsic_)>=hsic_)  # the more there are pval_terms bigger than hsic, the higher the p-value
    else:
        pval = hsic_ # not a pvalue
    # print('pval',pval)
    return pval

def HSIC(K,L,M, eps):

    n = K.shape[0]
    M_eps = torch.cholesky_inverse(M + n*eps*torch.eye(n,device=K.device))  
    M_eps_2 = torch.matmul(M_eps, M_eps)

    term_1 = torch.matmul(K, L)
    KM = torch.matmul(K,M)
    ML = torch.matmul(M,L)
    MLM = torch.matmul(M,torch.matmul(L,M))
    term_2 = torch.matmul(KM,torch.matmul(M_eps_2,ML))
    term_3 = torch.matmul( KM, torch.matmul( M_eps_2, torch.matmul( MLM,torch.matmul(M_eps_2,M) ) ) )

    return torch.trace(term_1 -2*term_2 + term_3)

def RBF(X,n_obs,gamma=-1):
    """
    Forms the kernel matrix K using the SE-T kernel with bandwidth gamma
    where T is the identity operator
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the distribution 
    gamma - bandwidth for the kernel, if -1 then median heuristic is used to pick gamma
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the distributions
    """
    
    dist_mat = (1/np.sqrt(n_obs))*pairwise_distances(X, metric='euclidean')
    if gamma == -1:
        gamma = np.median(dist_mat[dist_mat > 0])
   
    K = np.exp(-0.5*(1/gamma**2)*(dist_mat**2))
    return K


def Matern(X,n_obs,gamma=-1):
    """
    Forms the kernel matrix K using the SE-T kernel with bandwidth gamma
    where T is the identity operator
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the distribution 
    gamma - bandwidth for the kernel, if -1 then median heuristic is used to pick gamma
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the distributions
    """
    sqrt3 = np.sqrt(3.0)

    dist_mat = (1/np.sqrt(n_obs))*pairwise_distances(X, metric='euclidean')
    if gamma == -1:
        gamma = np.median(dist_mat[dist_mat > 0])
        
    # K = np.exp(-0.5*(1/gamma**2)*(dist_mat**2))

    K = (1.0 + sqrt3 * (1./gamma) * dist_mat) * np.exp(-sqrt3 * (1./gamma) * dist_mat)
    
    return K