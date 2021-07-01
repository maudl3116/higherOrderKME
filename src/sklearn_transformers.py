import numpy as np
import copy
import random
import doctest
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array
import torch
import ksig 

class AddTime(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to add time as an extra dimension of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    def __init__(self, init_time=0., total_time=1.):
        self.init_time = init_time
        self.total_time = total_time

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        t = np.linspace(self.init_time, self.init_time + 1, len(X))
        return np.c_[t, X]

    def transform(self, X, y=None):
        return [[self.transform_instance(x) for x in bag] for bag in X]


class LeadLag(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to compute the Lead-Lag transform of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    def __init__(self, dimensions_to_lag):
        if not isinstance(dimensions_to_lag, list):
            raise NameError('dimensions_to_lag must be a list')
        self.dimensions_to_lag = dimensions_to_lag

    def fit(self, X, y=None):
        return self

    def transform_instance_1D(self, x):

        lag = []
        lead = []

        for val_lag, val_lead in zip(x[:-1], x[1:]):
            lag.append(val_lag)
            lead.append(val_lag)
            lag.append(val_lag)
            lead.append(val_lead)

        lag.append(x[-1])
        lead.append(x[-1])

        return lead, lag

    def transform_instance_multiD(self, X):
        if not all(i < X.shape[1] and isinstance(i, int) for i in self.dimensions_to_lag):
            error_message = 'the input list "dimensions_to_lag" must contain integers which must be' \
                            ' < than the number of dimensions of the original feature space'
            raise NameError(error_message)

        lead_components = []
        lag_components = []

        for dim in range(X.shape[1]):
            lead, lag = self.transform_instance_1D(X[:, dim])
            lead_components.append(lead)
            if dim in self.dimensions_to_lag:
                lag_components.append(lag)

        return np.c_[lead_components + lag_components].T

    def transform(self, X, y=None):
        return [[self.transform_instance_multiD(x) for x in bag] for bag in X]


class ExpectedSignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # get the lengths of all time series (across items across bags)
        lengths = [item.shape[0] for bag in X for item in bag]
        if len(list(set(lengths))) == 1:
            # if all time series have the same length, the signatures can be computed in batch
            X = [iisignature.sig(bag, self.order) for bag in X]
        else:
            X = [np.array([iisignature.sig(item, self.order) for item in bag]) for bag in X]
        return [x.mean(0) for x in X]


class SketchExpectedSignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order, ncompo, rbf=False, lengthscale=1):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order
        self.ncompo = ncompo
        if rbf:
            static_kernel = ksig.static.kernels.RBFKernel(lengthscale=lengthscale) 
        else:
            static_kernel = ksig.static.kernels.LinearKernel() 
        static_feat = ksig.static.features.NystroemFeatures(static_kernel, n_components=ncompo)
        proj = ksig.projections.CountSketchRandomProjection(n_components=ncompo)
        self.lr_sig_kernel = ksig.kernels.LowRankSignatureKernel(n_levels=order, static_features=static_feat, projection=proj)

    def fit(self, X, y=None):
        self.lr_sig_kernel.fit(np.array(X[0]))
        return self

    def transform(self, X, y=None):
        X = np.array(X)
        try:
            X_ = X.reshape((-1,X.shape[2],X.shape[3]))   #(NM, L, D)
            feat = self.lr_sig_kernel.transform(X_)   #(NM, D)
            ES = feat.reshape((X.shape[0],X.shape[1],feat.shape[1]))  # (M,N,D)
            ES = np.concatenate([x.mean(0)[None,:] for x in ES]) 
        except:
            ES = np.array([self.lr_sig_kernel.transform(bag) for bag in X])   #(M, N, D)
            ES = np.concatenate([x.mean(0)[None,:] for x in ES]) 

        print(ES.shape)
        return ES 


class pathwiseExpectedSignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pwES = []
        for bag in X:
            # get the lengths of all time series in the bag
            lengths = [item.shape[0] for item in bag]
            if len(list(set(lengths))) == 1:
                # if all time series have the same length, the (pathwise) signatures can be computed in batch
                pwES.append(iisignature.sig(bag, self.order, 2))
            else:
                error_message = 'All time series in a bag must have the same length'
                raise NameError(error_message)

        return [x.mean(0) for x in pwES]

class pathwiseSketchExpectedSignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order, ncompo, rbf=False, lengthscale=1):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order
        self.ncompo = ncompo
        if rbf:
            static_kernel = ksig.static.kernels.RBFKernel(lengthscale=lengthscale) 
        else:
            static_kernel = ksig.static.kernels.LinearKernel() 
        static_feat = ksig.static.features.NystroemFeatures(static_kernel, n_components=ncompo)
        proj = ksig.projections.CountSketchRandomProjection(n_components=ncompo)
        self.lr_sig_kernel = ksig.kernels.LowRankSignatureKernel(n_levels=order, static_features=static_feat, projection=proj)

    def fit(self, X, y=None):

        self.lr_sig_kernel.fit(np.array(X[0]))
        return self

    def transform(self, X, y=None): 
        try:
            X = np.array(X)
            X_ = X.reshape((-1,X.shape[2],X.shape[3]))   #(NM, L, D)
            feat = [self.lr_sig_kernel.transform(X_[:,:i,:])[:,None,:] for i in range(2,X_.shape[1])] # list of (NMx1xD) features
            feat = np.concatenate(feat,axis=1) # (NMxLxD)
            pwES = feat.reshape((X.shape[0],X.shape[1],feat.shape[1],feat.shape[2]))  # (M,N,L,D)
            pwES = np.concatenate([x.mean(0)[None,:,:] for x in pwES]) 
        except:
            pwES = []
            for bag in X:
                feat = [self.lr_sig_kernel.transform(np.array(bag)[:,:i,:])[:,None,:] for i in range(2,bag[0].shape[0])]   # list of (Nx1xD) features 
                pwES.append(np.concatenate(feat,axis=1))   # list of (NxLxD) tensors
            pwES = np.concatenate([x.mean(0)[None,:,:] for x in pwES]) 
        print(pwES.shape)
        return pwES 


class SignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # get the lengths of all pathwise expected signatures
        lengths = [pwES.shape[0] for pwES in X]
        if len(list(set(lengths))) == 1:
            # if all pathwise expected signatures have the same length, the signatures can be computed in batch
            return iisignature.sig(X, self.order)
        else:
            return [iisignature.sig(item, self.order) for item in X]


class SketchSignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order, ncompo):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order
        self.ncompo = ncompo
        # static_kernel = ksig.static.kernels.RBFKernel() 
        static_kernel = ksig.static.kernels.LinearKernel() 
        static_feat = ksig.static.features.NystroemFeatures(static_kernel, n_components=ncompo)
        proj = ksig.projections.CountSketchRandomProjection(n_components=ncompo)
        self.lr_sig_kernel = ksig.kernels.LowRankSignatureKernel(n_levels=order, static_features=static_feat, projection=proj)

    def fit(self, X, y=None):
        self.lr_sig_kernel.fit(X)
        return self

    def transform(self, X, y=None):
        # get the lengths of all pathwise expected signatures
        lengths = [pwES.shape[0] for pwES in X]
        if len(list(set(lengths))) == 1:
            # if all pathwise expected signatures have the same length, the signatures can be computed in batch
            return self.lr_sig_kernel.transform(X)
        else:
            return [self.lr_sig_kernel.transform(item[None,:,:]) for item in X]

class SketchpwCKMETransform(BaseEstimator, TransformerMixin):

    def __init__(self, order, ncompo, rbf=False, lengthscale=1, lambda_=1):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order
        self.ncompo = ncompo
        self.lambda_ = lambda_
        if rbf:
            static_kernel = ksig.static.kernels.RBFKernel(lengthscale=lengthscale) 
        else:
            static_kernel = ksig.static.kernels.LinearKernel() 
        static_feat = ksig.static.features.NystroemFeatures(static_kernel, n_components=ncompo)
        proj = ksig.projections.CountSketchRandomProjection(n_components=ncompo)
        self.lr_sig_kernel = ksig.kernels.LowRankSignatureKernelPathwise(n_levels=order, static_features=static_feat, projection=proj)

    def fit(self, X, y=None):

        self.lr_sig_kernel.fit(np.array(X[0]))
        return self

    def transform(self, X, y=None):
        try:
            X = np.array(X)
            X_ = X.reshape((-1,X.shape[2],X.shape[3]))   #(NM, L, D)
            # feat = [self.lr_sig_kernel.transform(X_[:,:i,:])[:,None,:] for i in range(2,X_.shape[1])] # list of (NMx1xD) features
            # feat = np.concatenate(feat,axis=1) # (NMxLxD)
            feat = self.lr_sig_kernel.transform(X_) #(NM, L, D)
            pwS = feat.reshape((X.shape[0],X.shape[1],feat.shape[1],feat.shape[2]))  # (M,N,L,D)

        except:
            pwS = []
            for bag in X:
                # feat = [self.lr_sig_kernel.transform(np.array(bag)[:,:i,:])[:,None,:] for i in range(2,bag[0].shape[0])]   # list of (Nx1xD) features 
                # pwS.append(np.concatenate(feat,axis=1))   # list of (NxLxD) tensors
                feat = self.lr_sig_kernel.transform(np.array(bag))
                pwS.append(feat)
            pwS = np.array(pwS) #(MxNxLxD) tensor
        
        #(pathwise) multitask ridge regression
        # clf = Ridge(alpha=self.lambda_)
        pwCKMEs = []


        
        T = torch.tensor(pwS).cuda()  #(M,N,L,D)


        for i in range(len(X)):
            to_inv = torch.zeros((T.shape[2], T.shape[3], T.shape[3]), dtype=T.dtype, device=T.device)
            for p in range(pwS.shape[2]):
                to_inv[p,:,:] = torch.matmul(T[i,:,p,:].t(),T[i,:,p,:])
            to_inv+= self.lambda_*torch.eye(T.shape[3],dtype=T.dtype, device=T.device)[None,:,:]
            inv = torch.linalg.inv(to_inv)
            pwCKME = []
            for p in range(pwS.shape[2]):
                X = T[i,:,p,:]
                y = T[i,:,-1,:]

                CKME = torch.matmul(X,inv[p,:,:])
                Xy = torch.matmul(X.t(),y)
                CKME = torch.matmul(CKME,Xy)
                #clf.fit(pwS[i,:,p,:],pwS[i,:,-1,:])
                #CKME = clf.predict(pwS[i,:,p,:])  # N, D
                pwCKME.append(CKME[:,None,:].cpu().numpy())
            pwCKMEs.append(np.concatenate(pwCKME,axis=1)) # N L D

    
        return np.array(pwCKMEs) # M N L D 




if __name__ == "__main__":
    doctest.testmod()