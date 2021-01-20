import matplotlib.pyplot as plt
import numpy as np
import util as ut

filename = "ex1data1.txt"
data = np.loadtxt(filename, delimiter=',')

# transfer X, y into vector
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# plot data set
plt.scatter(X, y)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()

# to take into account of the intercept term theta_0
# add ones to X
ones = np.ones([X.shape[0], 1])
X = np.hstack([ones, X])

theta = np.zeros(shape=(2, 1))
iterations = 1500
alpha = 0.01
theta2 = np.asarray([[-1], [2]])

J_history, theta = ut.gradient_descend(X, y, theta, alpha, iterations)


def predict(arr, theta):
    return np.dot(arr, theta)


print(predict(np.asarray([1, 3.5]), theta))

theta_0 = np.linspace(-10, 10, 100)
theta_1 = np.linspace(-1, 4, 100)

J_vals = np.zeros([len(theta_0), len(theta_1)])

for i in range(len(theta_0)):
    for j in range(len(theta_1)):
        t = [theta_0[i], theta_1[j]]
        J_vals[i][j] = ut.compute_cost(X, y, t)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(theta_0, theta_1, J_vals, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()

# plot contour
