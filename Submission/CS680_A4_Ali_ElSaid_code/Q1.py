# All the code for Question 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Q1.1
class GMM():
    # GMM Class
    def __init__(self, lamb, mu, sigma):
        self.lamb = lamb
        self.mu = mu
        self.sigma = sigma

def GMMsample(gmm, n=1000, b=50):
    # Sample from GMM
    y = np.random.uniform(size=n)
    x1 = np.random.normal(gmm.mu[0], gmm.sigma[0], n)
    x2 = np.random.normal(gmm.mu[1], gmm.sigma[1], n)
    x = np.vectorize(lambda y, x1, x2: x1 if y < gmm.lamb else x2)(y, x1, x2)
    count, bins, ignored = plt.hist(x, b, density=True, facecolor='rebeccapurple')
    plt.title("Q1.1 Histogram of X")
    # plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()
    return x 

gmm = GMM(0.5, [1, -1], [0.5, 0.5])
n=1000
b=50
x = GMMsample(gmm, n, b)

# Q1.2
def F_X(x, gmm):
    # CDF of X
    l1, m1, s1 =   gmm.lamb, gmm.mu[0], gmm.sigma[0]
    l2, m2, s2 = 1-gmm.lamb, gmm.mu[1], gmm.sigma[1]
    F = l1 * norm.cdf(x,loc=m1,scale=s1) + l2 * norm.cdf(x,loc=m2,scale=s2)
    return F

def GMMinv(x, gmm, b=50, title=''):
    # Inverting the GMM to N(0,1)
    u = norm.ppf(F_X(x, gmm))
    count, bins, ignored = plt.hist(u, b, density=True, facecolor='rebeccapurple')
    plt.title(title)
    plt.show()
    return u

u = GMMinv(x, gmm, b, "Q1.2 Histogram of U")

# Q1.3
def BinarySearch(F, u, lo=-100, hi=100, maxiter=100, eps=1e-10):
    # Binary search for solving a monotonic nonlinear equation F(x) = u, x = F_inv(u)
    while(F(lo) > u):
        hi = lo
        lo *= 2
    while(F(hi) < u):
        lo = hi
        hi *= 2
    for i in range(maxiter):
        mid = lo + (hi-lo)/2
        v = F(mid)
        if v < u:
            lo = mid
        else:
            hi = mid
        if abs(v - u) <= eps:
            break
    return mid

def T(z, gmm):
    # Push forward function from N(0,1) to 2 var GMM
    u = norm.cdf(z)
    return BinarySearch(lambda x: F_X(x, gmm=gmm), u)

r = np.arange(-5, 5 + 0.1, 0.1)
q1 = [T(v, gmm) for v in r]
plt.plot(r, q1, color='rebeccapurple')
plt.title("Q1.3 Plot of T")
plt.show()

# Q1.4
def PushForward(z, gmm):
    x = [T(v, gmm) for v in z]
    count, bins, ignored = plt.hist(x, b, density=True, facecolor='rebeccapurple')
    plt.title("Q1.4 Histogram of X tilda")
    plt.show()
    return x

z = np.random.normal(size=n)
x2 = PushForward(z, gmm)

# Q1.5
u2 = GMMinv(x2, gmm, b, "Q1.5 Histogram of U tilda")
