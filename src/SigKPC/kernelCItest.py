
import numpy as np
import torch 
import sigkernel
def hsicclust( X, Y, Z, p=0, numCluster=10,eps=0.1, dyadic_order=0,static='rbf', sigma=1.):

    X = torch.tensor(X,dtype=torch.float64)
    Y = torch.tensor(Y,dtype=torch.float64)
    Z = torch.tensor(Z,dtype=torch.float64)

    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        X = X.cuda()
        Y = Y.cuda()
        Z = Z.cuda()

    """Input: 
                    - X: torch tensor of shape (batch, length_X, dim_X),
                    - Y: torch tensor of shape (batch, length_Y, dim_Y)
                    - Z: torch tensor of shape (batch, length_Z, dim_Z)
                    - p: number of permutation of 1-alpha level test
                    - numCluster: number of clusters if we use kpc cluster permutation
                    - eps: normalization parameter
    """
    assert p==0, "p>0 not implemented"

    n = X.shape[0] # number of samples
    H = torch.eye(n,device=X.device) - (1./n)*torch.ones((n,n),device=X.device)
    pval_perm = torch.zeros(p, device=X.device)


    H = torch.eye(n,dtype=torch.float64,device=X.device) - (1./n)*torch.ones((n,n),dtype=torch.float64, device=X.device)

    # create signature kernel gram matrices
    if static=='rbf':
        static_kernel = sigkernel.RBFKernel(sigma=sigma)
    else:
        static_kernel = sigkernel.LinearKernel()

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
        
        # t = kmeans(z,numCluster)

        # perm = np.arange(n)

        # for j in range(numCluster):

        #     perm[] = ...

        # pval_perm[i]  = HSIC(K,L,M,eps)

    if p > 0:
        pval = np.mean(np.append(pval_perm,hsic_)>=hsic_)  # the more there are pval_terms bigger than hsic, the higher the p-value
    else:
        pval = hsic_ # not a pvalue
    # print('pval',pval)
    return pval



def HSIC(K,L,M, eps):
    # TODO: check what needs to be recomputed for the permutation test  
    n = K.shape[0]
    M_eps = torch.cholesky_inverse(M + n*eps*torch.eye(n,device=K.device))   # TODO: test with linalg.inv 
    M_eps_2 = torch.matmul(M_eps, M_eps)

    term_1 = torch.matmul(K, L)
    KM = torch.matmul(K,M)
    ML = torch.matmul(M,L)
    MLM = torch.matmul(M,torch.matmul(L,M))
    term_2 = torch.matmul(KM,torch.matmul(M_eps_2,ML))
    term_3 = torch.matmul( KM, torch.matmul( M_eps_2, torch.matmul( MLM,torch.matmul(M_eps_2,M) ) ) )

    return torch.trace(term_1 -2*term_2 + term_3)