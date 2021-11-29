import numpy as np
from matplotlib import pyplot as plt
import matplotlib

#Data set
# x_neg = np.array([[3,4],[1,4],[2,3]])
x_neg = np.array([  # y = 0
            [-2, 2],
            [-4, 2],
            [-4, 4],
            [-5, 3],
            [-6, 7],
            [-7, 1],
            [-7, 5],
            [-8, 8],
            [-9, 7],
])
# y_neg = np.array([-1,-1,-1])
y_neg = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1])  # y = 0 or -1
# x_pos = np.array([[6,-1],[7,-1],[5,-3]])
x_pos = np.array([  # y = 1
            [-2, 9],
            [1, 5],
            [2, 8],
            [4, 6],
            [5, 9],
            [7, 2],
            [8, 7],
            [9, 4],
            [6, 4],
])
# y_pos = np.array([1,1,1])
y_pos = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])  # y = 1

x1 = np.linspace(-10,10)
x = np.vstack((np.linspace(-10,10),np.linspace(-10,10)))

#Data for the next section
X = np.vstack((x_pos, x_neg))
y = np.concatenate((y_pos,y_neg))

#Parameters guessed by inspection
w = np.array([1,-1]).reshape(-1,1)
b = -3

#Plot
fig = plt.figure(figsize = (10,10))
plt.scatter(x_neg[:,0], x_neg[:,1], marker = 'x', color = 'r', label = 'Zeros 0')
plt.scatter(x_pos[:,0], x_pos[:,1], marker = 'o', color = 'b',label = 'Ones +1')
plt.xlim(-10, 10)
plt.ylim(0, 10)
plt.xticks(np.arange(-10, 10, step=1))
plt.yticks(np.arange(0, 10, step=1))

#Labels and show
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc = 'lower right')
plt.show()

#Importing with custom names to avoid issues with numpy / sympy matrix
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

#Initializing values and computing H. Note the 1. to force to float type
m,n = X.shape
y = y.reshape(-1,1) * 1.
X_dash = y * X
H = np.dot(X_dash , X_dash.T) * 1.

#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Setting solver parameters (change default to decrease tolerance)
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

#w parameter in vectorized form
w = ((y * alphas).T @ X).reshape(-1,1)

#Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

#Computing b
b = y[S] - np.dot(X[S], w)

#Display results
print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])

x_min = -10
x_max = 10
y_min = 0
y_max = 2

xx = np.linspace(x_min, x_max)
a = -w[0]/w[1]
yy = a*xx - (b)/w[1]

margin = 1 / np.sqrt(np.sum(w**2))
yy_neg = yy - np.sqrt(1 + a**2) * margin
yy_pos = yy + np.sqrt(1 + a**2) * margin

plt.figure(1, figsize=(4, 3))
plt.clf()

plt.plot(xx, yy[0, :], "b-")
plt.plot(xx, yy_neg[1, :], "m--")
plt.plot(xx, yy_pos[2, :], "m--")

colors = ["steelblue", "orange"]

plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), alpha=0.5, cmap=matplotlib.colors.ListedColormap(colors), edgecolors="black")

plt.show()
