# Scientific libraries
import numpy as np
import pandas as pd
from numpy import linalg
from numpy import random
import scipy
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import mode

# cvxopt QP solver
import cvxopt
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False # Verbose quite

from itertools import product


# Scientific libraries
import numpy as np
import pandas as pd
from numpy import linalg
from numpy import random
import scipy
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import mode

# cvxopt QP solver
import cvxopt
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False # Verbose quite

from itertools import product


class SVM2:
    kernels_ = ['linear', 'rbf', 'Cauchy', 'poly', 'TStudent', 'cosine', 'GHI']
    losses_ = ['hinge', 'squared_hinge']
    
    def __init__(self, C=1, kernel='poly', gamma=0.1, mode='OVO', loss='squared_hinge', c=0.159, degree=4, intercept=False,
                 decoding='loss-based'):
        self.C = C
        self.kernel = kernel # kernel_function 'rbf', 'linear'
        self.gamma = gamma # Kernel coefficient gamma for 'rbf'
        self.loss = loss
        self.mode = mode
        self.c = c # Intercept of the polynomial kernel
        self.degree = degree # Degree of the polynomial kernel
        self.alphas_ = [] # coefficients of the estimators
        self.decoding = decoding # Way of decoding the coding matrix 
        self.intercept = False

    def fit(self, X, y):
        
        self.X_train_ = X
        self.y_train_ = y
        self.n_samples_ = y.shape[0] # n_samples
        self.classes_, self.repartition_ = np.unique(y, return_counts=True)
        self.classes_ = self.classes_.astype(int)
        self.n_classes_ = self.classes_.shape[0]
        self.repartition_ = self.repartition_ / self.n_samples_
        self.K_ = self.fit_kernel()
        self.K_test_ = None
        
        
        if self.mode == 'OVA':
            
            for class_ in self.classes_:
                y_copy = y.copy()
                y_copy[y_copy != class_] = -1
                y_copy[y_copy == class_] = 1
                self.fit_dual(y_copy)
                
                # Solving the QP
                
                if self.intercept is True:
                    sol = solvers.qp(matrix(self.K_ + self.Q_, tc='d'), matrix(self.p_, tc='d'),
                                     matrix(self.G_, tc='d'), matrix(self.h_, tc='d'), matrix(self.A_, tc='d'), 
                                     matrix(self.b_, tc='d'))
                # Saving the solution
                else:
                    sol = solvers.qp(matrix(self.K_ + self.Q_, tc='d'), matrix(self.p_, tc='d'), matrix(self.G_), matrix(self.h_, tc='d'))
                    
                # Saving the solution
                self.alphas_.append(np.array(sol['x']).reshape(-1,))
    
        elif self.mode == 'OVO':
            
            masks_samples = [] # empty boolean list, len=45, True if the sample is in the current OVO comparison
            y_copies = [] # classes of OVO configuration are all stored
            for class_plus in range(self.n_classes_):
                for class_minus in range(class_plus+1, 10):
                    y_copy = y.copy()
                    
                    # Mask select the samples which match the classes considered in the current OVO comparison
                    mask = (y_copy == class_plus) | (y_copy == class_minus)
                    masks_samples.append(mask) # saving the mask
                    y_copy = y_copy[mask]
                    y_copy[y_copy == class_minus] = -1
                    y_copy[y_copy == class_plus] = 1
                    y_copies.append(y_copy)
                    
                    # Updating the size of the subproblem for the dual computation
                    # Important otherwise it would be built for n_samples = all the sample
                    self.n_samples_ = y_copy.shape[0]
                        
                    # This_K is the submatrix of K for the current OVO problem
                    this_K = self.K_[mask, :]
                    this_K = this_K[:, mask]
                    self.fit_dual(y_copy)
                    # Solvign the QP
                    if self.intercept is True:
                        sol = solvers.qp(matrix(this_K + self.Q_, tc='d'), matrix(self.p_, tc='d'), matrix(self.G_),
                                         matrix(self.h_), matrix(self.A_, tc='d'), matrix(self.b_, tc='d'))
                    else:
                        sol = solvers.qp(matrix(this_K + self.Q_, tc='d'), matrix(self.p_, tc='d'), matrix(self.G_), matrix(self.h_))
                        
                    # Saving the solution in list attibute alphas_
                    self.alphas_.append(np.array(sol['x']).reshape(-1,))
                    
            # Don't forget to update the n_samples attribute again
            self.n_samples_ = y.shape[0]
            self.y_copies_ = y_copies
            # Saving the masks as attributes
            self.masks_samples_ = np.array(masks_samples)
            
        else:
            raise Exception('mode should be OVA or OVO')
        return self
        
    def predict(self, X_test, alphas=None, fit=True):
        """Fit : boolean. If False, the gram matrix of the test sample is not computed
        The older one in memory is used instead. 
        It prevents from recomputing the matrix when we estimate several times the same X_test
        """
        
        if fit is True or self.K_test_ is None:
            self.K_test_ = self.fit_kernel_test(X_test)
            self.n_test_ = X_test.shape[0] # size of the test sample
        else:
            self.alphas_ = alphas
        
        n = self.n_test_
        
        if self.mode == 'OVA':
            classes_res = np.empty((self.classes_.shape[0], n))

            for class_ in self.classes_:
                alpha = self.alphas_[class_]
                # omega = alpha*self.y_train_ not useful
                classes_res[class_] = np.dot(self.K_test_, alpha) # By the representer theorem
            y_pred = classes_res.argmax(axis=0)
            return y_pred
        
        elif self.mode == 'OVO':
            
            # calling coding_matrix() to built a matrix that would select the correct configuration for OVO
            # rmk : coding_matrix() is always the same, but cheap to compute
            self.coding_matrix_ = coding_matrix()            
            
            estimators_preds = [] # list of length n_estimators which contains the predictions
            # n_estimators = 45 in OVO : n_classes_(n_classes-1)/2
            # each estimators  is of size n_test
            
            #-------CHOOSING THE BEST PREDICTION FOR EACH CLASS------
            for idx, mask in enumerate(self.masks_samples_):
                alpha = self.alphas_[idx]
                # omega = alpha*self.y_copies_[idx] not useful
                estimators_preds.append(np.dot(self.K_test_[:, mask], alpha)) # By the representer theorem
            
            # Converting the result and taking the sign for prediction
            estimators_preds = np.sign(np.array(estimators_preds))
            # estimators_preds is now an array of size n_estimators*n_test
            
            
            #------DECODING PART-----------
            classes_preds = [] # list of length n_classes_, contain the predictions for each class
            for mask in self.coding_matrix_:
                class_pred = estimators_preds.copy()
                
                if self.decoding == 'Hamming':
                    class_pred = (1 - np.sign(class_pred*mask.reshape(-1,1))).sum(0)
                    
                elif self.decoding == 'loss-based':
                    class_pred = ((np.maximum(1 - class_pred*mask.reshape(-1,1), 0))**2).sum(0)
                    
                classes_preds.append(class_pred)
            
            classes_preds = np.array(classes_preds)
            self.classes_preds = classes_preds
            y_pred = classes_preds.argmin(0) # Not argmax, because it's loss
            return y_pred
        
    def fit_kernel(self):
        
        X = self.X_train_
        
        if self.kernel == 'rbf':
            print('rbf choosen')
            pairwise_dists = squareform(pdist(X, 'euclidean'))
            K = scipy.exp(-self.gamma*pairwise_dists ** 2)
        
        elif self.kernel == 'linear':
            K = np.dot(X, X.transpose())
        
        elif self.kernel == 'poly':
            K = (self.c + np.dot(X, X.transpose())) ** self.degree
        
        elif self.kernel == 'TStudent':
            pairwise_dists = squareform(pdist(X, 'euclidean'))
            K = 1 / (1 + pairwise_dists ** self.degree)
        elif self.kernel == 'log':
            # Not a PD kernel, raised error solving the QP
            # Rank(A) < p or Rank([P; A; G]) < n
            pairwise_dists = squareform(pdist(X, 'euclidean'))
            K = - scipy.log(pairwise_dists ** self.degree + 1)
        
        elif self.kernel == 'power':
            # ONLY FOR POSITIVE VALUES
            # Error raised solving the QP
            # Error QP : Rank(A) < p or Rank([P; A; G]) < n

            pairwise_dists = squareform(pdist(X))
            K = -(pairwise_dists ** self.degree)
        
        elif self.kernel == 'Cauchy':
            pairwise_dists = squareform(pdist(X, 'euclidean'))
            K = 1 / (1 + (pairwise_dists ** 2) / self.gamma**2)
        
        elif self.kernel == 'cosine':
            # Doesn't perform well ACC 20% on n=4000
            pairwise_dists = squareform(pdist(X, 'cosine'))
            K = -pairwise_dists + 1
        
        elif self.kernel == 'sigmoid':
            # Also known as the tangent hyperbolic
            #There are two adjustable parameters in the sigmoid kernel
            #the slope alpha and the intercept constant c. 
            #A common value for gamma is 1/N, where N is the data dimension.
            pairwise_dists = squareform(pdist(X, 'cosine'))
            K = scipy.tanh(self.gamma*np.dot(X, X.transpose()) + self.c)
        
        elif isinstance(self.kernel, list):
            
            pairwise_dists = squareform(pdist(X, 'euclidean'))
            K = 0.1*scipy.exp(-self.gamma*pairwise_dists ** 2)
            K += 0.9*(self.c + np.dot(X, X.transpose())) ** self.degree
        
        elif self.kernel == 'chi2':
            data_1 = X
            data_2 = X
            if np.any(data_1 < 0) or np.any(data_2 < 0):
                warnings.warn('Chi^2 kernel requires data to be strictly positive!')

            K = np.zeros((data_1.shape[0], data_2.shape[0]))

            for d in range(data_1.shape[1]):
                column_1 = data_1[:, d].reshape(-1, 1)
                column_2 = data_2[:, d].reshape(-1, 1)
            K += (column_1 - column_2.T)**2 / (column_1 + column_2.T)
            
        elif self.kernel == 'min':
            data_1 = X
            data_2 = X
            if np.any(data_1 < 0) or np.any(data_2 < 0):
                warnings.warn('Min kernel requires data to be strictly positive!')

            K = np.zeros((data_1.shape[0], data_2.shape[0]))

            for d in range(data_1.shape[1]):
                column_1 = data_1[:, d].reshape(-1, 1)
                column_2 = data_2[:, d].reshape(-1, 1)
                K += np.minimum(column_1, column_2.T)
        
        elif self.kernel == 'GHI':
            data_1 = np.abs(X)**self.gamma
            data_2 = np.abs(X)**self.gamma
            if np.any(data_1 < 0) or np.any(data_2 < 0):
                warnings.warn('Min kernel requires data to be strictly positive!')

            K = np.zeros((data_1.shape[0], data_2.shape[0]))

            for d in range(data_1.shape[1]):
                column_1 = data_1[:, d].reshape(-1, 1)
                column_2 = data_2[:, d].reshape(-1, 1)
                K += np.minimum(column_1, column_2.T)
            
            K += np.eye(K.shape[0]) * 0.0000001
        
        else:         
            raise Exception('the kernel must either be rbf or linear')
                       
        return K
                       
    def fit_kernel_test(self, X_test):
        # Compute the kernel gram matrix for the TEST set
            
        if self.kernel == 'rbf':
            pairwise_dists = cdist(X_test, self.X_train_)
            K = scipy.exp(-self.gamma*pairwise_dists ** 2)
        
        elif self.kernel == 'linear':
            K = np.dot(X_test, self.X_train_.transpose() + self.c)
                       
        elif self.kernel == 'poly':
            K = (self.c + np.dot(X_test, self.X_train_.transpose())) ** self.degree
            
        elif self.kernel == 'TStudent':
            pairwise_dists = (cdist(X_test, self.X_train_))
            K = 1 / (1 + pairwise_dists ** self.degree)
            
        elif self.kernel == 'log':
            # ONLY FOR POSITIVE VALUES
            # Error raised solving the QP

            pairwise_dists = cdist(X_test, self.X_train_)
            K = - scipy.log(pairwise_dists ** self.degree + 1)
        
        elif self.kernel == 'power':
            # ONLY FOR POSITIVE VALUES
            # Error raised solving the QP

            pairwise_dists = cdist(X_test, self.X_train_)
            K = -(pairwise_dists ** self.degree)
        
        elif self.kernel == 'Cauchy':
            pairwise_dists = cdist(X_test, self.X_train_)
            K = 1 / (1 + (pairwise_dists ** 2) / self.gamma**2)
            
        elif self.kernel == 'cosine':
            pairwise_dists = cdist(X_test, self.X_train_, 'cosine')
            K = -pairwise_dists + 1
            
        elif self.kernel == 'sigmoid':
            # Also known as the tangent hyperbolic
            K = scipy.tanh(self.gamma*np.dot(X_test, self.X_train_.transpose()) + self.c)
        
        elif isinstance(self.kernel, list):
            
            pairwise_dists = cdist(X_test, self.X_train_)
            K = 0.1*scipy.exp(-self.gamma*pairwise_dists ** 2)
            K += 0.9*(self.c + np.dot(X_test, self.X_train_.transpose())) ** self.degree
        
        elif self.kernel == 'chi2':
            data_1 = X_test
            data_2 = self.X_train_
            if np.any(data_1 < 0) or np.any(data_2 < 0):
                warnings.warn('Chi^2 kernel requires data to be strictly positive!')

            K = np.zeros((data_1.shape[0], data_2.shape[0]))

            for d in range(data_1.shape[1]):
                column_1 = data_1[:, d].reshape(-1, 1)
                column_2 = data_2[:, d].reshape(-1, 1)
            K += (column_1 - column_2.T)**2 / (column_1 + column_2.T)
        
        elif self.kernel == 'min':
            data_1 = X_test
            data_2 = self.X_train_
            if np.any(data_1 < 0) or np.any(data_2 < 0):
                warnings.warn('Min kernel requires data to be strictly positive!')

            K = np.zeros((data_1.shape[0], data_2.shape[0]))

            for d in range(data_1.shape[1]):
                column_1 = data_1[:, d].reshape(-1, 1)
                column_2 = data_2[:, d].reshape(-1, 1)
                K += np.minimum(column_1, column_2.T)
        
        elif self.kernel == 'GHI':
            data_1 = np.abs(X_test)**self.gamma
            data_2 = np.abs(self.X_train_)**self.gamma
            if np.any(data_1 < 0) or np.any(data_2 < 0):
                warnings.warn('Min kernel requires data to be strictly positive!')

            K = np.zeros((data_1.shape[0], data_2.shape[0]))

            for d in range(data_1.shape[1]):
                column_1 = data_1[:, d].reshape(-1, 1)
                column_2 = data_2[:, d].reshape(-1, 1)
                K += np.minimum(column_1, column_2.T)


                

        else:
            raise Exception('the kernel must either be rbf or linear')
        
        return K
    
    def fit_dual(self, y):
        
        n = y.shape[0]
        # We don't use self.n_samples_ because it's not up to date for the OVO
        # since the subproblem size has been modified in between (and we don't use setters)
        y = y.astype(float)
        
        if self.loss == 'hinge':
            diag_y = np.diag(y)
            self.p_ = (-y)
            self.Q_ = np.zeros((n,n), dtype= float)# Quadratic matrix
            self.G_ = np.r_[diag_y, -diag_y] # Constraint matrix of size(2*n, n)
            self.h_ = np.r_[self.C*np.ones(n, dtype=float), np.zeros(n, dtype=float)]
            if self.intercept is True:
                self.A_ = np.zeros(n, dtype=float)
                self.b_ = 0.0
            else :
                self.A_ = None #np.eye(n, dtype=float)
                self.b_ = None #np.zeros(n, dtype=float)           
        
        elif self.loss == 'squared_hinge':
            
            diag_y = np.diag(y)
            self.p_ = (-y)
            self.Q_ = self.C*n*np.eye(n, dtype=float) # Quadratic matrix, need + K, added in fit()
            self.G_ = -diag_y # Constraint matrix of size(2*n, n)
            self.h_ = np.zeros(n, dtype=float)
            if self.intercept is True:
                self.A_ = np.zeros(n, dtype=float)
                self.b_ = 0.0
            else :
                self.A_ = None #np.eye(n, dtype=float)
                self.b_ = None #np.zeros(n, dtype=float)
            
        else:
            raise Exception('loss should be hinge loss or squared_hinge_loss')
        
        return self
                       
    def describe(self):
        # Describe the train set
        print(" The Data contain ", self.n_classes_, " classes, and the repartion is :  ", self.repartition_)

def coding_matrix():
    """Computation of the coding matrix for the OVO as referred to in Schapire e al. paper
    http://www.jmlr.org/papers/volume1/allwein00a/allwein00a.pdf"""
    masks = []
    for n in range(10):    
        mask = []
        for k in range(10):
            for j in range(k+1, 10):
                if (k == n) | (j==n):
                    if j==n:
                        mask.append(-1)
                    else:
                        mask.append(1)
                else:
                    mask.append(0)
        mask = np.array(mask)
        masks.append(mask)
    masks = np.array(masks)
    return masks

def tune_parameters(X, y, X_val, y_val, param_grid, n_train, X_test = None, verbose = True):
    """X : array which would be split in train and val set according to n_train
    n_train : number of train samples. Integer or percentage
    param_grid : dict containing list of parameters to be tested
    IMPORTANT : param_grid values have to be a list. ex : not 'hinge' but ['hinge']
    IMPORTANT 2 : We should pass X_val as an argument, because an image x is differentiate 4 times. Therefore if only 2 of them are in the train set, the 2 others can be in the validation one, and hence the score increase 
    """
    
    n_total = X.shape[0]
    if n_total != y.shape[0]:
        raise Exception('X and y have different size')
    
    
    # Storing results
    scores = {}
    preds = {}
    preds_test = {}
    estimators = {}
    param_grid = [param_grid]
    for param in param_grid:
        # sort the keys of a dictionary, for reproducibility
        items = sorted(param.items())
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            # Parameters are ready for fitting the model            
            svm = SVM2(**params)
            
            # Checking if n_train is percentage or integer
            if n_train <= 1:
                idx_train = random.choice(np.arange(n_total), int(n_train*n_total), replace=False)
            else :
                idx_train = random.choice(np.arange(n_total), n_train, replace=False)
               
            #idx_val = list(set(np.arange(n_total)) - set(idx_train))
            # n_val max is set to 2000
            """if len(idx_val) > 2000:
                idx_val = idx_val[:2000]
            """
            # Fitting and storing results
            svm.fit(X[idx_train], y[idx_train])
            pred = svm.predict(X_val)
            estimators[str(params)] = svm.alphas_
            preds[str(params)] = pred
            score = np.mean(pred == y_val)                
            scores[str(params)]= score
            
            if X_test is not None:
                    pred_test = svm.predict(X_test)
                    preds_test[str(params)] = pred_test
            
            if verbose is True:
                print(params)
                print('SCORE : ', score)
    
    return {'scores' : scores, 'preds' : preds, 'estimators' : estimators, 'preds_test' : preds_test}


def bagging(X, y, params, X_val, y_val, n_iter=100, ratio=0.4):

    """Bagging implementation.
    
    UPDATE : estimators disabled for the moment. Too computationally expensive, memory errors. 
    Therefore passing X_val and y_val argument is necessary
    DESCRIPTION :
    The prediction can be done after computation calling the svm.predict for svm in the estimators list
    and then taking the argmax over the labels, i.e the scipy.stats.mode(preds, axis=0) 
    Or it can be done in parallel parsing, X_val, y_val, and then the array preds is returned.
    
    ---Inputs---
    X : A set of train data
    y : labels 
    params : dict of parameters for the SVM
    X_val, y_val : optional, for inside evaluation of our estimators
    
    ---Outputs---
    estimators = list of svm objects
    preds : n_estimators*n_val array, if X_val is 
    score : intermediate score for each prediction, if y_val provided
    
    """
    
    n_train, p = X.shape
    
    # To save the results
    #estimators = []  Disabled for the moment because it lead to ''the kernel has died'' after 50 iterations
    preds = []
    scores = []
    
    for kk in range(n_iter):
        print('---------- ', 'iter : ', kk,' ------')
        
        # Bootstrap phase
        X_bootstrap = []
        y_bootstrap = []
        n_samples = round(n_train * ratio)
        random_indexes = random.choice(n_samples, n_samples)
        for idx in random_indexes:
            X_bootstrap.append(X[idx])
            y_bootstrap.append(y[idx])

        # Fitting model
        svm = SVM2(**params)
        svm.fit(np.asarray(X_bootstrap), np.asarray(y_bootstrap))
        
        if X_val is not None:
            # prediction and score
            pred = svm.predict(X_val)
            preds.append(pred)
            if y_val is not None:
                score = np.mean(pred == y_val)
                scores.append(score)
                print(" Score iter ", score)
                print(" Global score ", np.mean(scipy.stats.mode(preds, 0)[0] == y_val))
        
        # Saving the estimator
        #estimators.append(svm)
        # Disabled for the moment because it lead to ''the kernel has died'' after 50 iterations
    return { 'preds' : preds, 'scores' : scores}

def submit_solution(y_pred):
    df = pd.DataFrame()
    df['Id'] = np.arange(1, y_pred.shape[0]+1)
    df['Prediction'] = y_pred.astype(int)
    df.to_csv('OVA_SVM_1_poly_0.01_2_squared_it14', index=False)

def mode(a, axis=0, weights=None):
    """It onlt works if the accuracy is > 50 % 
    Otherwise we make much mistake than good prediction, so taign the mode leads to a worse score
    """
    if isinstance(a, np.ndarray):
        pass
    else:
        a = np.asarray(a)
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    if weights is None:
        weights = np.ones(a.shape[0])
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape, dtype=a.dtype)
    oldcounts = np.zeros(testshape, dtype=int)
    for score in scores:
        template = (a == score).astype(int)
        counts = np.expand_dims(np.dot(weights, template), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis

def bagging_corrected(l_bag, y_val):
    score_max = np.max(l_bag['scores'])
    exp_score =  np.exp(l_bag['scores'] - score_max)
    w = exp_score / np.sum(exp_score)
    pred_corrected = mode(l_bag['preds'], 0, weights=w)
    score_corrected = np.mean(y_val == pred_corrected)
    print(score_corrected)
    return score_corrected

    
def tune_parameters_random(X, y, X_val, y_val, param_grid, n_train, n_iter=1, X_test = None, verbose = True):
    """X : array which would be split in train and val set according to n_train
    n_train : number of train samples. Integer or percentage
    param_grid : dict containing list of parameters to be tested
    IMPORTANT : param_grid values have to be a list. ex : not 'hinge' but ['hinge']
    IMPORTANT 2 : We should pass X_val as an argument, because an image x is differentiate 4 times. Therefore if only 2 of them are in the train set, the 2 others can be in the validation one, and hence the score increase 
    """
    
    n_total = X.shape[0]
    if n_total != y.shape[0]:
        raise Exception('X and y have different size')
        
    if n_train <= 1:
        # Checking if n_train is percentage or integer
        n_train = round(n_train*n_total)
    
    # Storing results
    scores = {}
    preds = {}
    preds_test = {}
    estimators = {}
    param_grid = [param_grid]
    for param in param_grid:
        # sort the keys of a dictionary, for reproducibility
        items = sorted(param.items())
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            # Parameters are ready for fitting the model            
            
            for kk in range(n_iter):
                print('---------- ', 'iter : ', kk,' ------')
                # Fitting the model here is necessary, it leads to an error if before loop otherwise
                svm = SVM2(**params)

        
                # Bootstrap phase
                X_bootstrap = []
                y_bootstrap = []
                random_indexes = random.choice(n_total, n_train, replace=True)
                print('rnd ', random_indexes.shape[0])
                for idx in random_indexes:
                    X_bootstrap.append(X[idx])
                    y_bootstrap.append(y[idx])

                #idx_val = list(set(np.arange(n_total)) - set(idx_train))
                # n_val max is set to 2000
                """if len(idx_val) > 2000:
                    idx_val = idx_val[:2000]
                """
                # Fitting and storing results
                svm.fit(np.asarray(X_bootstrap), np.asarray(y_bootstrap))
                """idx_val = np.random.choice(X_val.shape[0], 2000, replace=False)
                print('idx_val : ', idx_val.shape[0])"""
                pred = svm.predict(X_val)
                estimators[str(params)] = svm.alphas_
                preds[str(params)] = pred
                score = np.mean(pred == y_val)                
                scores[str(params)]= score

                if X_test is not None:
                        pred_test = svm.predict(X_test)
                        preds_test[str(params)] = pred_test

                if verbose is True:
                    print(params)
                    print('SCORE : ', score)
    
    return {'scores' : scores, 'preds' : preds, 'estimators' : estimators, 'preds_test' : preds_test}
