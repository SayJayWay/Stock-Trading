# =============================================================================
# Unsupervised Learning
# =============================================================================

# Data creation
import numpy as np
import pandas as pd
import datetime as ddt
from pylab import mpl, plt
from sklearn.datasets.samples_generator import make_blobs

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
np.random.seed(1000)
np.set_printoptions(suppress = True, precision = 4)

# Create sample data set for clustering w/ 250 samples and 4 centers
X, y = make_blobs(n_samples = 250, centers = 4, random_state = 500, cluster_std = 1.25)

plt.figure(figsize = (10, 6))
plt.scatter(X[:, 0], X[:, 1], s = 50)
plt.title('Sample data for application of clustering algorithms')

#%% k-means clustering
from sklearn.cluster import KMeans

# Instantiating model object, given certain parametrs, knowledge about the 
# sample data is used to inform the instantiation
model = KMeans(n_clusters = 4, random_state = 0)


model.fit(X)

# Fit model object to raw data
KMeans(algorithm = 'auto', copy_x = True, init = 'k-means++', max_iter = 300,
       n_clusters = 4, n_init = 10, n_jobs = None, precompute_distances = 'auto',
       random_state = 0, tol = 0.0001, verbose = 0)

# Predicts the cluster (number) given the raw data
y_kmeans = model.predict(X)

plt.figure(figsize = (10, 6))
plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, cmap = 'coolwarm')
plt.title('Sample data and identified clusters')

#%% Gaussian mixture
# Alternative clustering method, similar to k-means, with slightly different 
# implementation

from sklearn.mixture import GaussianMixture

model = GaussianMixture(n_components = 4, random_state = 0)

model.fit(X)
GaussianMixture(covariance_type = 'full', init_params = 'kmeans', max_iter = 100,
                means_init = None, n_components = 4, n_init = 1, precisions_init = None,
                random_state = 0, reg_covar = 1e-06, tol = 0.001, verbose = 0,
                verbose_interval = 10, warm_start = False, weights_init = None)

y_gm = model.predict(X)

#(y_gm == y_kmeans).all() # Output: True -> Results from k-means and Gaussian are same

# =============================================================================
# Supervised Learning
# =============================================================================
# Data creation must have two real-valued, informative features and a single binary
# label (characterized by two different classes only, 0 and 1).

from sklearn.datasets import make_classification

n_samples = 100

X, y = make_classification(n_samples = n_samples, n_features = 2, n_informative = 2,
                            n_redundant = 0, n_repeated = 0, random_state = 250)

X.shape # Output: (100,2) -> two informtive, real-valued features
y.shape # Output: (100,) -> single binary label

plt.figure(figsize = (10, 6))
plt.hist(X)

plt.figure(figsize = (10, 6))
plt.scatter(x = X[:, 0], y = X[:, 1], c=y, cmap = 'coolwarm')
plt.title('Sample data for the application of classification algorithms')

#%% Gaussian Naive Bayes
# Good baseline algorithm for multitide of different classification problems

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

model = GaussianNB()

model.fit(X, y)
GaussianNB(priors = None, var_smoothing = 1e-09)

model.predict_proba(X).round(4) # Shows probabilities that algorithm assigns to
                                # each class after fitting
                                
pred = model.predict(X) # Based on probabilities, predicts binary classes for data set

pred == y # Compares predicted classes with real ones

accuracy_score(y, pred) # Calculates accuracy score -> 0.87

Xc = X[y == pred] # Selects "correct" predictions and plots them
Xf = X[y != pred] # Selects "false" predictions and plots them

plt.figure(figsize = (10, 6))

# Selects "correct" predictions and plots them
plt.scatter(x = Xc[:, 0], y = Xc[:, 1], c = y[y == pred], marker = 'o', cmap = 'coolwarm')

# Selects "false" predictions and plots them
plt.scatter(x = Xf[:, 0], y = Xf[:, 1], c = y[y != pred], marker = 'x', cmap = 'coolwarm')
plt.title('Correct (dots) and false predictions (crosses) from GNB')

#%% Logistic Regression (LR)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C = 1, solver = 'lbfgs')

model.fit(X, y)
LogisticRegression(C = 1, class_weight = None, dual = False, fit_intercept = True,
                   intercept_scaling = 1, max_iter = 100, multi_class = 'warn',
                   n_jobs = None, penalty = 'l2', random_state = None, solver = 'lbfgs',
                   tol = 0.0001, verbose = 0, warm_start = False)

model.predict_proba(X).round(4)

pred = model.predict(X)
accuracy_score(y, pred)

Xc = X[y == pred]
Xf = X[y != pred]

plt.figure(figsize = (10, 6))
plt.scatter(x = Xc[:, 0], y = Xc[:, 1], c = y[y == pred], marker = 'o', cmap = 'coolwarm')
plt.scatter(x = Xf[:, 0], y = Xf[:, 1], c = y[y != pred], marker = 'x', cmap = 'coolwarm')
# In this case, LR performs a bit better than GaussianNB

#%% Decision trees (DTs) scale quite well.  With depth of 1, the algorithm already
# performs slightly better than both GNB and LR

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth = 1)

model.fit(X, y)
DecisionTreeClassifier(class_weight = None, criterion = 'gini', max_depth = 1,
                       max_features = None, max_leaf_nodes = None,
                       min_impurity_decrease = 0.0, min_impurity_split = None,
                       min_samples_leaf = 1, min_samples_split = 2,
                       min_weight_fraction_leaf = 0.0, presort = False, random_state = None,
                       splitter = 'best')

model.predict_proba(X).round(4)
pred = model.predict(X)

accuracy_score(y, pred)

Xc = X[y == pred]
Xf = X[y != pred]

plt.figure(figsize = (10, 6))
plt.scatter(Xc[:, 0], Xc[:, 1], marker = 'o', c = y[y == pred], cmap = 'coolwarm')
plt.scatter(Xf[:, 0], Xf[:, 1], marker = 'x', c = y[y != pred], cmap = 'coolwarm')

# Increasing the maximum depth parameter allows one to reach a perfect result

print('{:>8s} | {:8s}'.format('depth', 'accuracy'))
print(20 * '-')
for depth in range(1, 7):
    model = DecisionTreeClassifier(max_depth = depth)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    print('{:8d} | {:8.2f}'.format(depth, acc))

#%%    
# =============================================================================
# Deep neural networks (DNNs) are among the most powerful but also computationally
# # demanding alogrithms for both estimation and classification
# =============================================================================
    
#%% DNNs with scikit-learn
# scikit-learn provides same API for its MLPClassifier algorithm class, which is 
# a DNN model.  With just two 'hidden-layers', it reaches a perfect result on
# the test data
    
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, 
                      hidden_layer_sizes = 2 * [75], random_state = 10)

model.fit(X, y)# %time 179ms

MLPClassifier(activation = 'relu', alpha = 1e-05, batch_size = 'auto', beta_1 = 0.9,
              beta_2 = 0.999, early_stopping = False, epsilon = 1e-08,
              hidden_layer_sizes = [75, 75], learning_rate = 'constant',
              learning_rate_init = 0.001, max_iter = 200, momentum = 0.9,
              n_iter_no_change = 10, nesterovs_momentum = True, power_t = 0.5,
              random_state = 10, shuffle = True, solver = 'lbfgs', tol = 0.001,
              validation_fraction = 0.1, verbose = False, warm_start = False)

pred = model.predict(X)

accuracy_score(y, pred)

#%% DNNs with tensorflow
# Tensorflow set-up is slightly different

import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR) # Sets verbosity for TensorFlow logging

fc = [tf.contrib.layers.real_valued_column('features')] # Defines real-valued features abstractly

# Instantiates model object
model = tf.contrib.learn.DNNClassifier(hidden_units = 5 * [250],
                                       n_classes = 2,
                                       feature_columns = fc)

# Features and label data are delivered by a function
def input_fn():
    fc = {'features' : tf.constant(X)}
    la = tf.constant(y)
    return fc, la

# Fit and evaluate model
model.fit(input_fn = input_fn, steps = 100) # %time 5.33s
model.evaluate(input_fn = input_fn, steps = 1)

# Predicts label values based on feature values
pred = np.array(list(model.predict(input_fn = input_fn))) # Predicts label values based on feature values

# Retrains model based on learning more steps (previous results are taken as starting point)
model.fit(input_fn = input_fn, steps = 750)

model.evaluate(input_fn = input_fn, steps = 1) # Accuracy increases after retraining

#%% Feature transforms
# Typical transformations that are useful to know

from sklearn import preprocessing

X[:5] # Original matrix

# Feature data -> Standard normally distributed data w/ 0 mean and unit variance
Xs = preprocessing.StandardScaler().fit_transform(X)
Xs[:5]

# Feature data -> Given range for every feature as defined by min/max values per feature
Xm = preprocessing.MinMaxScaler().fit_transform(X)
Xm[:5]

# Scales feature data individually accoridng to unit norm (L1 or L2)
Xn1 = preprocessing.Normalizer(norm = 'l1').transform(X)
Xn1[:5]
Xn2 = preprocessing.Normalizer(norm = 'l2').transform(X)
Xn2[:5]

plt.figure(figsize = (10, 6))
markers = ['o', '.', 'x', '^', 'v']
data_sets = [X, Xs, Xm, Xn1, Xn2]
labels = ['raw', 'standard', 'minmax', 'norm(1)', 'norm(2)']
for x, m, l in zip(data_sets, markers, labels):
    plt.scatter(x = x[:, 0], y = x[:, 1], c = y, marker = m, cmap = 'coolwarm',
                label = l)
plt.legend()

# For pattern recognition tasks, a transformation to categorical features is
# often helpful or sometimes required to achieve acceptable results.  To this
# end, the real values of the features are mapped to a limited, fixed number
# of possible integer values (categories, classes):

X[:5]

# Transforms the features to binary features -> 2 ** 2 number of possible feature
# value combinations for two binary features
Xb = preprocessing.Binarizer().fit_transform(X)
Xb[:5]

# Transforms features to categorical features based on a list of values used
# for binning -> 4 ** 2 number of possible feature value combinations with 3 
# values used for binning for two features
Xd = np.digitize(X, bins = [-1, 0, 1]) 
Xd[:5]

#%% Train-test splits: Support vector machines
# Always better to use train/test datasets (as opposed to what we did prior).
# scikit-learn has function to do this effectively (train_test_split().  The 
# following code uses another classification algorithm : the "support vector
# machine (SVM)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.33, random_state = 0)

model = SVC(C = 1, kernel = 'linear')

model.fit(train_x, train_y) # Fits model based on training data
SVC(C = 1, cache_size = 200, class_weight = None, coef0 = 0.0,
    decision_function_shape = 'ovr', degree = 3, gamma = 'auto_deprecated',
    kernel = 'linear', max_iter = -1, probability = False, random_state = None,
    shrinking = True, tol = 0.001, verbose = False)

pred_train = model.predict(train_x) # Predicts training data label values

accuracy_score(train_y, pred_train) # Accuracy of training data prediction ("in-sample")

# Predicts testing data label values based on test data
pred_test = model.predict(test_x)

# Evaluates accuracy of fitted model for test data ("out-of-sample")
test_y == pred_test

accuracy_score(test_y, pred_test)

test_c = test_x[test_y == pred_test]
test_f = test_x[test_y != pred_test]

plt.figure(figsize = (10, 6))
plt.scatter(x = test_c[:, 0], y = test_c[:, 1], c = test_y[test_y == pred_test],
            marker = 'o', cmap = 'coolwarm')

plt.scatter(x = test_f[:, 0], y = test_f[:, 1], c = test_y[test_y != pred_test],
            marker = 'x', cmap = 'coolwarm')
plt.title('Correct (dots) and false predictions (crosses) from SVM for test data')

# SVM algorithm has number of options for kernel to be used.  Dpeending on kernel
# used, might lead to quite different results (i.e. accuracy scores)

bins = np.linspace(-4.5, 4.5, 50)
Xd = np.digitize(X, bins = bins)

train_x, test_x, train_y, test_y = train_test_split(Xd, y, test_size = 0.33, random_state = 0)

print('{:>8s} | {:8s}'.format('kernel', 'accuracy'))
print(20 * '-')
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    model = SVC(C = 1, kernel = kernel, gamma = 'auto')
    model.fit(train_x, train_y)
    acc = accuracy_score(test_y, model.predict(test_x))
    print('{:>8s} | {:8.3f}'.format(kernel, acc))