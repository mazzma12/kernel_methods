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
    
    def __init__(self, C=1, kernel='rbf', gamma=0.1, mode='OVA', loss='squared_hinge', c=0, degree=2):
        self.C = C
        self.kernel = kernel # kernel_function 'rbf', 'linear'
        self.gamma = gamma # Kernel coefficient gamma for 'rbf'
        self.loss = loss
        self.mode = mode
        self.c = c # Intercept of the polynomial kernel
        self.degree = degree # Degree of the polynomial kernel
        self.alphas_ = [] # coefficients of the estimators

    def fit(self, X, y, ):
        
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
                sol = solvers.qp(matrix(self.K_), matrix(self.p_), matrix(self.G_), matrix(self.h_))
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
                    sol = solvers.qp(matrix(this_K + self.Q_), matrix(self.p_), matrix(self.G_), matrix(self.h_))
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
            
            # calling OVO_estimators_masks() to built masks_estimators
            # Used in masks_estimators*K_test later
            # rmk : OVO_estimators_masks() is always the same, but cheap to compute
            self.masks_estimators_ = OVO_estimators_masks()            
            
            estimators_preds = [] # list of length n_estimators which contains the predictions
            # n_estimators = 45 in OVO : n_classes_(n_classes-1)/2
            # each estimators  is of size n_test
            
            for idx, mask in enumerate(self.masks_samples_):
                alpha = self.alphas_[idx]
                # omega = alpha*self.y_copies_[idx] not useful
                estimators_preds.append(np.dot(self.K_test_[:, mask], alpha)) # By the representer theorem
            
            # Converting the result and taking the sign for prediction
            estimators_preds = np.sign(np.array(estimators_preds))
            # estimators_preds is now an array of size n_estimators*n_test
            
            class_preds = np.dot(self.masks_estimators_, estimators_preds)
            y_pred = class_preds.argmax(0)
            
            """# Choosing which class is predicted using the mask_matrix built with OVO_idx_class()
            classes_preds = [] # list of length n_classes_, contain the predictions for each class
            for mask in self.masks_estimators_:
                class_pred = estimators_preds.copy()
                class_pred = class_pred*mask.reshape(-1,1)
                class_pred[class_pred < 0] = 0
                class_pred = class_pred.sum(0)
                classes_preds.append(class_pred)
            
            classes_preds = np.array(classes_preds)
            
            # Argmax give the index of the row with the highest score
            # rows are ordered so the index corresponds to the class
            y_pred = classes_preds.argmax(0)"""
            
            return y_pred

    def fit_kernel(self):
        
        X = self.X_train_
        
        if self.kernel == 'rbf':
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

            pairwise_dists = squareform(pdist(X_test))
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
            #A common value for alpha is 1/N, where N is the data dimension.
            pairwise_dists = squareform(pdist(X, 'cosine'))
            K = scipy.tanh(self.gamma*np.dot(X, X.transpose()) + self.c)
        elif self.kernel == 'min':
            K = squareform(pdist(X, lambda u, v : np.sum(scipy.minimum(u, v))))
        
        elif self.kernel == 'GHI':
            # Error QP : Rank(A) < p or Rank([P; A; G]) < n
            K = squareform(pdist(X, lambda u, v : np.sum(scipy.minimum(np.abs(u)** self.gamma, np.abs(v)** self.gamma))))
        
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
        elif self.kernel == 'min':
            K = cdist(X_test, self.X_train_, lambda u, v : np.sum(scipy.minimum(u, v)))
        elif self.kernel == 'GHI':
            K = cdist(X_test, self.X_train_, 
                      lambda u, v : np.sum(scipy.minimum(np.abs(u)** self.gamma, np.abs(v)** self.gamma)))
                          
        else:
            raise Exception('the kernel must either be rbf or linear')
        
        return K
    
    def fit_dual(self, y):
        
        n = y.shape[0]
        # We don't use self.n_samples_ because it's not up to date for the OVO
        # since the subproblem size has been modified in between (and we don't use setters)
        
        if self.loss == 'hinge':
            diag_y = np.diag(y)
            self.p_ = (-y)
            self.Q_ = self.K_ # Quadratic matrix
            self.G_ = np.r_[diag_y, -diag_y] # Constraint matrix of size(2*n, n)
            self.h_ = np.r_[self.C*np.ones(n), np.zeros(n)]
                       
        elif self.loss == 'squared_hinge':
            
            diag_y = np.diag(y)
            self.p_ = (-y)
            self.Q_ = self.C*n*np.eye(n) # Quadratic matrix
            self.G_ = -diag_y # Constraint matrix of size(2*n, n)
            self.h_ = np.zeros(n)
            
        else:
            raise Exception('loss should be hinge loss or squared_hinge_loss')
        
        return self
                       
    def describe(self):
        # Describe the train set
        print(" The Data contain ", self.n_classes_, " classes, and the repartion is :  ", self.repartition_)

def OVO_estimators_masks():
    # DIFFICULT
    # Build a matrix with 1 and 0 for the OVO prediction
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

def tune_parameters(X, y, param_grid, n_train, X_test = None, verbose = True):
    """X : array which would be split in train and val set according to n_train
    n_train : number of train samples. Integer or percentage
    param_grid : dict containing list of parameters to be tested
    IMPORTANT : param_grid values have to be a list. ex : not 'hinge' but ['hinge']
    """
    
    n_total = X.shape[0]
    if n_total != y.shape[0]:
        raise Exception('X and y have different size')
    
    
    # Storing results
    scores = {}
    preds = []
    preds_test = []
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
            if n_train < 1:
                idx_train = random.choice(np.arange(n_total), int(n_train*n_total), replace=False)
            else :
                idx_train = random.choice(np.arange(n_total), n_train, replace=False)
               
            idx_val = list(set(np.arange(n_total)) - set(idx_train))
            # n_val max is set to 2000
            if len(idx_val) > 2000:
                idx_val = idx_val[:2000]
            
            # Fitting and storing results
            svm.fit(X[idx_train], y[idx_train])
            pred = svm.predict(X[idx_val])
            estimators[str(params)] = svm.alphas_
            preds.append(pred)
            score = np.mean(pred == y[idx_val])                
            scores[str(params)]= score
            
            if X_test is not None:
                    pred_test = svm.predict(X_test)
                    preds_test.append(pred_test)
            
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