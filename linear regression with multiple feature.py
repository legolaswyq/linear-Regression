import numpy as np
import util as ut

# house price prediction
# size of the house (in square feet),
# number of bedrooms,
# price

filename = "ex1data2.txt"
data = np.loadtxt(filename, delimiter=",")

# extract feature and label
X = data[:, :2]
y = data[:, 2].reshape(-1,1)


# feature normalize
def feature_normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return np.true_divide((X - mean), std)


X_normalize = feature_normalize(X)
# add bias term
ones = np.ones([X.shape[0],1])
X = np.hstack([ones, X_normalize])

alpha = 1
num_iters = 400
theta = np.zeros(shape=(3,1))


J_history, theta = ut.gradient_descend(X,y,theta,alpha,num_iters)

print(theta)


