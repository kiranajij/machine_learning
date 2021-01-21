import numpy as np
import matplotlib.pyplot as plt


def cost(x, y, theta):
    """
    calculates the cost of the linear model given x, y and parameters theta

    cost is defined by

    cost  = 1/2m sum( theta.T*x(i) - y )^2 )
    """

    m = len(theta)
    raw_cost = sum(
        (x.dot(theta) - y)**2
    ) 
    cost = raw_cost / (2*m)
    return cost[0]

def 

def main():
    # x data
    x = 10*np.random.rand(1000, 2)
    x[:, 0] = np.ones(1000)

    # y data
    y = x[:, 0] * .4 + x[:, 1]*2 + 1*np.random.rand(1000)
    y = y.reshape((1000, 1))

    # theta
    theta = np.zeros((2, 1))



if __name__ == '__main__':
    main()