# Q2.3

import numpy as np
import matplotlib.pyplot as plt

n=3
d=2
lamb = 1
T = 100

X0 = np.random.normal(size=(n, d))

X = [0] * (T+1)

X[0] = X0
for t in range(T):
    St = np.exp(np.matmul(X[t], np.transpose(X[t])) / lamb)
    Pt = np.matmul(np.linalg.inv(np.diag(np.matmul(np.ones(n), St))), St)
    # print(Pt)
    X[t+1] = np.matmul(Pt, X[t])

colors = ['rebeccapurple', 'springgreen', 'darkturquoise']
for i in range(n):
    points = [Xt[i, :] for Xt in X]
    x, y = zip(*points)
    plt.plot(x, y, color=colors[i])
    for j in range(n):
        plt.scatter([X0[j, 0]], [X0[j, 1]], color=colors[j], zorder=5)
    plt.title(f'Row {i+1} Convergence')
    plt.show()
