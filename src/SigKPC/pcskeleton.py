# adapted from the pcalg package in R

import numpy as np
import math
from .kernelCItest import hsicclust, hsicclust_baseline

def skeleton(data, alpha, test, p, maxi=1, fixedGaps=None, eps=0.01, dyadic_order=0, static='rbf', sigma=1.,baseline=None,gamma=1):
                         
    """
    Performs undirected part of PC-Algorithm (skeleton phase)

    Input: 
                    - data: torch tensor of shape (n_sto_pro, n_repeat , L, dim),
                    - alpha: threshold to decide whether to reject edge or not 

    TODO: implement permutation test
    """

    
    if fixedGaps is None:
        # start from the fully connected graph
        G = np.ones((p,p)) 
        np.fill_diagonal(G, 0)

    ## Order-independent version: Compute the adjacency sets for any vertex
    ## Then don't update when edges are deleted
    G_memo = G.copy()

    pval = np.nan
    pMAX = -math.inf*np.ones((p,p))
    hsicMIN = math.inf*np.ones((p,p))
    np.fill_diagonal(pMAX, 1)

    done = False
    L = 1      

 
    while(not done and np.sum(G)!=0 and L<=maxi):
        done = True
        
        # find pairs of nodes which are connected 
        ind = np.argwhere(G==1)

        remainEdges = ind.shape[0]

        for edge_ind in range(remainEdges):

            i = ind[edge_ind,0]
            j = ind[edge_ind,1]

            if G[j,i]:
                # use memo adjacency matrix to know which nodes are neighbors or not
                neighborsBool = G_memo[:,i].copy()
                # say that node j is not a neighbor 
                neighborsBool[j] = 0 
                # find the neighbors
                neighbors = np.arange(len(neighborsBool))[neighborsBool==1]
                nbNeighbors = len(neighbors)

                if nbNeighbors >= L:
                    if nbNeighbors > L:
                        done = False
                    
                    # set the first set of L neighbors to condition on 

                    S = np.arange(L)
                    while True:
                        set_nei = np.concatenate(data[neighbors[S]],axis=2)
                        if baseline is None:
                            pval = hsicclust(data[i],data[j], set_nei, eps=eps, dyadic_order=dyadic_order,static=static, sigma=sigma) 
                        else:
                            pval = hsicclust_baseline(data[i],data[j], set_nei, eps=eps, baseline=baseline, gamma=gamma)
                        if pMAX[i,j] < pval:
                            pMAX[i,j] = pval
                        if hsicMIN[i,j] > pval:
                            hsicMIN[i,j] = pval
                        if test:
                            if pval >= alpha: 
                                # disconnect i and j
                                G[i,j] = 0 
                                G[j,i] = 0
                                break
                            else:
                                # get the next set of L neighbors to condition on 
                                nextSet, wasLast = getNextSet(nbNeighbors,L,S)
                                if wasLast:
                                    break
                                S = nextSet
                        else:
                            if pval <= alpha: 
                                # disconnect i and j
                                G[i,j] = 0 
                                G[j,i] = 0
                                break
                            else:
                                # get the next set of L neighbors to condition on 
                                nextSet, wasLast = getNextSet(nbNeighbors,L,S)
                                if wasLast:
                                    break
                                S = nextSet

        # end for remaining edge
        L+=1

    ## end while   

    for i in range(p-1):
        for j in range(1,p):
            pMAX[i,j] = max(pMAX[i,j],pMAX[j,i])
            pMAX[j,i] = pMAX[i,j]
    for i in range(p-1):
        for j in range(1,p):
            hsicMIN[i,j] = min(hsicMIN[i,j],hsicMIN[j,i])
            hsicMIN[j,i] = hsicMIN[i,j]

    return G, pMAX, hsicMIN

def getNextSet(n,k,set_):
    ## Purpose: Generate the next set in a list of all possible sets of size
    ##          k out of 1:n;
    ##  Also returns a boolean whether this set was the last in the list.
    ## ----------------------------------------------------------------------
    ## Arguments:
    ## - n,k: Choose a set of size k out of numbers 1:n
    ## - set_: previous set in list
    ## ----------------------------------------------------------------------

    ## chInd := changing Index
    zeros = sum((np.arange(n-k,n)-set_) == 0)
    
    chInd = k - zeros

    wasLast = (chInd == 0)
    if not wasLast:
        set_[chInd-1] = set_[chInd-1] + 1
        if chInd < k:
            set_[chInd:k] = np.arange(set_[chInd-1] + 1, set_[chInd-1] + 1 + zeros)

    return set_, wasLast 