import numpy as np
from numpy import linalg

# cvxopt QP solver
import cvxopt
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False # Verbose quite

import scipy
from scipy.spatial.distance import pdist, cdist, squareform

class SVM:
    kernels_ = ['linear', 'rbf', 'Cauchy', 'poly', 'TStudent', 'cosine', 'GHI']
    losses_ = ['hinge', 'squared_hinge']
    
    def __init__(self, C=1, kernel='rbf', gamma=0.1, mode='OVO', loss='hinge', c=0, degree=2):
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

                    this_K = self.K_[mask, :]
                    this_K = this_K[:, mask]
                    self.fit_dual(y_copy)
                    # Solvign the QP
                    sol = solvers.qp(matrix(this_K), matrix(self.p_), matrix(self.G_), matrix(self.h_))
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
            semf.alphas_ = alphas
        
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
            # n_estimators = 45 in OVO : n_classes_(n_classes+1)/2
            
            for idx, mask in enumerate(self.masks_samples_):
                alpha = self.alphas_[idx]
                # omega = alpha*self.y_copies_[idx] not useful
                print(omega)
                estimators_preds.append(np.dot(self.K_test_[:, mask], alpha)) # By the representer theorem
            
            # Converting the result and taking the sign for prediction
            estimators_preds = np.sign(np.array(estimators_preds))
            
            # Choosing which class is predicted using the mask_matrix built with OVO_idx_class()
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
            y_pred = classes_preds.argmax(0)
            
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
            print(X_test.shape, self.X_train_.shape)
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
        
        if self.loss == 'loss':
            n = self.n_samples_
            diag_y = np.diag(y)
            self.p_ = (-y)
            self.Q_ = self.K_ # Quadratic matrix
            self.G_ = np.r_[diag_y, -diag_y] # Constraint matrix of size(2*n, n)
            self.h_ = np.r_[self.C*np.ones(n), np.zeros(n)]
                       
        elif self.loss == 'squared_hinge':
            
            n = self.n_samples_
            diag_y = np.diag(y)
            self.p_ = (-y)
            self.Q_ = self.K_ + self.C*n*np.eye(n) # Quadratic matrix
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
