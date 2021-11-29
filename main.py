import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cvxopt import matrix, solvers
from sklearn.datasets import load_iris
import pandas as pd

def main():
    iris = load_iris()
    iris_df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
    # Retain only 2 linearly separable classesiris_df = iris_df[iris_df["target"].isin([0,1])]iris_df["target"] = iris_df[["target"]].replace(0,-1)
    # Select only 2 attributesiris_df = iris_df[["petal length (cm)", "petal width (cm)", "target"]]
    iris_df.head()

    x = iris_df[["petal length (cm)", "petal width (cm)"]].to_numpy()
    y = iris_df[["target"]].to_numpy()

    x = np.array([
        # y = 1
        [-2, 9],
        [1, 5],
        [2, 8],
        [4, 6],
        [5, 9],
        [7, 2],
        [8, 7],
        [9, 4],
        [6, 4],

        # y = 0
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

    y = np.array([
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],

        # y = 0
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
    ])

    n = x.shape[0]
    H = np.dot(y * x, (y * x).T)
    q = np.repeat([-1.0], n)[..., None]
    A = y.reshape(1, -1)
    b = 0.0
    G = np.negative(np.eye(n))
    h = np.zeros(n)

    # Q_ex = 2 * matrix([[2, .5], [.5, 1]])
    # p_ex = matrix([1.0, 1.0])
    # G_ex = matrix([[-1.0, 0.0], [0.0, -1.0]])
    # h_ex = matrix([0.0, 0.0])
    # A_ex = matrix([1.0, 1.0], (1, 2))
    # b_ex = matrix(1.0)
    # print(Q_ex.size)
    # print(p_ex.size)
    # print(G_ex.size)
    # print(h_ex.size)
    # print(A_ex.size)
    # print(b_ex.size)

    P = matrix(H)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    print(P.size)
    print(q.size)
    print(G.size)
    print(h.size)
    print(A.size)
    print(b.size)

    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution["x"])

    w = np.dot((y * alphas).T, x)[0]
    S = (alphas > 1e-5).flatten()
    b = np.mean(y[S] - np.dot(x[S], w.reshape(-1, 1)))

    print('W:', w)
    print('b:', b)

    x_min = 0
    x_max = 5.5
    y_min = 0
    y_max = 2

    xx = np.linspace(x_min, x_max)
    a = -w[0] / w[1]
    yy = a * xx - (b) / w[1]

    margin = 1 / np.sqrt(np.sum(w ** 2))
    yy_neg = yy - np.sqrt(1 + a ** 2) * margin
    yy_pos = yy + np.sqrt(1 + a ** 2) * margin

    plt.figure(figsize=(8, 8))
    plt.plot(xx, yy, "b-")
    plt.plot(xx, yy_neg, "m--")
    plt.plot(xx, yy_pos, "m--")
    colors = ["steelblue", "orange"]

    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), alpha=0.5, cmap=matplotlib.colors.ListedColormap(colors),
                edgecolors="black")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.show()

    # w_coefficients = np.random.rand((9, 2))
    # w0_arbitrary_constant = np.random.rand((2, 2))
    # w_norm = np.linalg.norm(w_coefficients)
    # lagrange_multipliers
    # distance

    # plotting the original datasets
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(y1_data[:, 0], y1_data[:, 1], c='b', label="y = 1")
    # ax.scatter(y0_data[:, 0], y0_data[:, 1], c='r', label="y = 0")
    # plt.legend()
    # plt.show()




if __name__ == "__main__":
    main()