import itertools
import torch
import numpy as np
from tqdm import tqdm as tqdm
from tqdm import trange as trange
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR, SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from higherOrderKME import sigkernel
from .sklearn_transformers import AddTime, LeadLag, SketchExpectedSignatureTransform, SketchpwCKMETransform

# ===========================================================================================================
# Alternative 1st DR algorithm. Here we use an approximation of the signature kernel based on randomized features 
# 
# ===========================================================================================================

def model_sketch(X, y, depths=[2], ncompos=[20], rbf=True, alphas=[1], ll=None, at=False,  num_trials=1, cv=3, grid={}):
    """Performs a kernel based distribution classification on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series (of possibly unequal lengths)
       We use the RBF embedding throughout. 
       Input:
              X (list): list of lists such that
                        - len(X) = n_samples
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                        - for any j, X[i][j] is an array of shape (length, dim)
              y (np.array): array of shape (n_samples,)
              depths (list of int): signature truncation depth to cross-validate
              ncompos (list of int): number of projection components to cross-validate
              rbf (bool): whether to use the RBF embedding on the state space of the time series 
              alphas (list of floats): RBF lenghtsclaes to cross-validate
              ll (list of ints): dimensions to lag (set to None by default)
              at (bool): if True pre-process the input path with add-time
              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion
              num_trials, cv : parameters for cross-validation
              grid (dict): a dictionary to specify the hyperparameter grid for the gridsearch. Unspecified entries will be set by default
       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (num_trials times) as well results (a dictionary containing the predicted labels and true labels)
    """
    
    use_gpu = torch.cuda.is_available()

    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)
    
    # default grid
    parameters = {'clf__kernel': ['precomputed'],
                    'rbf_mmd__gamma':[1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3],
                    'clf__alpha': [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]}

    # check if the user has not given an irrelevant entry
    assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
        list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

    # merge the user grid with the default one
    parameters.update(grid)

    clf = KernelRidge

    list_kernels = []
    hyperparams = list(itertools.product(depths, ncompos, alphas))
    # Precompute the Gram matrices for the different scaling parameters, to avoid recomputing them for each grid search step
    for (depth, ncompo, alpha) in hyperparams:

        ES = SketchExpectedSignatureTransform(order=depth, ncompo=ncompo, rbf=rbf, lengthscale=alpha).fit_transform(np.array(X))  #(M,D)

        mmd = -2*ES@ES.T
        mmd += np.diag(mmd)[:, None] + np.diag(mmd)[None, :]

        if np.isnan(mmd).any():
            list_kernels.append(np.eye(len(X)))
        else:
            list_kernels.append(mmd)

    scores = np.zeros(num_trials)
    results = {}
    # Loop for each trial
    for i in tqdm(range(num_trials)):

        best_scores_train = np.zeros(len(hyperparams))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the hyperparameters.
   
        MSE_test = np.zeros(len(hyperparams))
        results_tmp = {}
        models = []
        for n, (depth, ncompo, alpha) in enumerate(hyperparams):
            
            ind_train, ind_test, y_train, y_test = train_test_split(np.arange(len(y)), np.array(y), test_size=0.3,
                                                                    random_state=i)
            
            # building the estimator
            pipe = Pipeline([('rbf_mmd', RBFSigMMDKernel(K_full=list_kernels[n])),
                             ('clf', clf())])

            # parameter search
            model = GridSearchCV(pipe, parameters, refit=True, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                 error_score=np.nan)

            model.fit(ind_train, y_train)
            best_scores_train[n] = -model.best_score_
            # print(model_.best_params_)
            y_pred = model.predict(ind_test)
        
            results_tmp[n]={'pred':y_pred, 'true':y_test}
            MSE_test[n] = mean_squared_error(y_pred, y_test)
            models.append(model)
        # pick the model with the best performances on the train set
        best_score = 100000
        index = None
        for n, (depth, ncompo, alpha) in enumerate(hyperparams):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
       
        print('best scaling parameter (cv on the train set): ', hyperparams[index])
        print('best mse score (cv on the train set): ', best_scores_train[index])
    return scores.mean(), scores.std(), results


# ===========================================================================================================
# Alternative 2nd DR algorithm. Here we use an approximation of the signature kernel based on randomized features 
# 
# ===========================================================================================================

def model_higher_order_sketch(X, y, depths1=[2], ncompos1=[20], rbf1=True, alphas1=[1], lambdas_=[10], depths2=[2], ncompos2=[20], rbf2=True, alphas2=[1], ll=None, at=False,  num_trials=1, cv=3, grid={}):
    """Performs a kernel based distribution classification on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series (of possibly unequal lengths)
       We use the RBF embedding throughout. 
       Input:
              X (list): list of lists such that
                        - len(X) = n_samples
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                        - for any j, X[i][j] is an array of shape (length, dim)
              y (np.array): array of shape (n_samples,)
              depths1 (list of int): level 1 signature truncation depth to cross-validate
              ncompos1 (list of int): number of projection components for level 1 to cross-validate
              rbf1 (bool): whether to use the RBF embedding on the state space of the time series for level 1 
              alphas1 (list of floats): RBF lenghtsclaes for level 1 to cross-validate
              lambdas (list of floats): conditional signature mean embedding regularizer to cross-validate
              depths2 (list of int): level 2 signature truncation depth to cross-validate
              ncompos2 (list of int): number of projection components for level 2 to cross-validate
              rbf2 (bool): whether to use the RBF embedding on the state space of the time series for level 2
              alphas2 (list of floats): RBF lenghtsclaes for level 2 to cross-validate
              ll (list of ints): dimensions to lag (set to None by default)
              at (bool): if True pre-process the input path with add-time
              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion
              num_trials, cv : parameters for cross-validation
              grid (dict): a dictionary to specify the hyperparameter grid for the gridsearch. Unspecified entries will be set by default
       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (num_trials times) as well results (a dictionary containing the predicted labels and true labels)
    """
    
    use_gpu = torch.cuda.is_available()

    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)
    
    # default grid
    parameters = {'clf__kernel': ['precomputed'],
                    'rbf_mmd__gamma':[1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3],
                    'clf__alpha': [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]}

    # check if the user has not given an irrelevant entry
    assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
        list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

    # merge the user grid with the default one
    parameters.update(grid)

    clf = KernelRidge

    list_kernels = []
    hyperparams = list(itertools.product(depths1, ncompos1, alphas1, lambdas_, depths2, ncompos2, alphas2))
    # Precompute the Gram matrices for the different scaling parameters, to avoid recomputing them for each grid search step
    for (depth1, ncompo1, alpha1, lambda_, depth2, ncompo2, alpha2) in hyperparams:

        pwCKME = SketchpwCKMETransform(order=depth1, ncompo=ncompo1, rbf=rbf1, lengthscale=alpha1, lambda_=lambda_).fit_transform(X) 
        if at:
            pwCKME = AddTime().fit_transform(pwCKME)
        ES = SketchExpectedSignatureTransform(order=depth2, ncompo=ncompo2, rbf=rbf2, lengthscale=alpha2).fit_transform(np.array(pwCKME))  #(M,D)

        mmd = -2*ES@ES.T
        mmd += np.diag(mmd)[:, None] + np.diag(mmd)[None, :]

        if np.isnan(mmd).any():
            list_kernels.append(np.eye(len(X)))
        else:
            list_kernels.append(mmd)

    scores = np.zeros(num_trials)
    results = {}
    # Loop for each trial
    for i in tqdm(range(num_trials)):

        best_scores_train = np.zeros(len(hyperparams))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the hyperparameters.
   
        MSE_test = np.zeros(len(hyperparams))
        results_tmp = {}
        models = []
        for n, (depth1, ncompo1, alpha1, lambda_, depth2, ncompo2, alpha2) in enumerate(hyperparams):
            
            ind_train, ind_test, y_train, y_test = train_test_split(np.arange(len(y)), np.array(y), test_size=0.3,
                                                                    random_state=i)
            
            # building the estimator
            pipe = Pipeline([('rbf_mmd', RBFSigMMDKernel(K_full=list_kernels[n])),
                             ('clf', clf())])

            # parameter search
            model = GridSearchCV(pipe, parameters, refit=True, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                 error_score=np.nan)

            model.fit(ind_train, y_train)
            best_scores_train[n] = -model.best_score_
            # print(model_.best_params_)
            y_pred = model.predict(ind_test)
        
            results_tmp[n]={'pred':y_pred, 'true':y_test}
            MSE_test[n] = mean_squared_error(y_pred, y_test)
            models.append(model)
        # pick the model with the best performances on the train set
        best_score = 100000
        index = None
        for n, (depth1, ncompo1, alpha1, lambda_, depth2, ncompo2, alpha2) in enumerate(hyperparams):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
       
        print('best scaling parameter (cv on the train set): ', hyperparams[index])
        print('best mse score (cv on the train set): ', best_scores_train[index])
    return scores.mean(), scores.std(), results


class RBFSigMMDKernel(BaseEstimator, TransformerMixin):
    def __init__(self, K_full=None, gamma=1.0):
        super(RBFSigMMDKernel, self).__init__()
        self.gamma = gamma
        self.K_full = K_full

    def transform(self, X):
        K = self.K_full[X][:, self.ind_train].copy()
        return np.exp(-self.gamma*K) 

    def fit(self, X, y=None, **fit_params):
        self.ind_train = X
        return self