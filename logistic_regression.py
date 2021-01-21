import numpy as np
import matplotlib.pyplot as plt
import logging
import tqdm
from termcolor import colored


# logger = logging


class Classifier:
    def __init__(self):

        """
        Assume data set already has 1's at it's first column 
        """

        self.nfeatures = None
        self.X = None
        self.y = None

    def fit(self, X, y, alpha=0.1, iters=100, regul=100):
        """
        Fit the optimizer to the given dataset
        """

        # Set up trainig environment
        self.nfeatures = X.shape[1]
        assert (X.shape == (y.size, self.nfeatures))
        theta = np.zeros((self.nfeatures,), dtype='float')
        self.X = X
        self.y = y

        for i in tqdm.tqdm(range(iters)):
            cost = self._cost(theta, regul=regul)
            grad = self._grad(theta, regul=regul)
            theta -= grad*alpha

        print(f" Iter {colored(iters, 'yellow')}\t cost={colored(cost, 'green')}")
        self.cost = cost
        
        self.theta = theta
    
    def predict(self, X):
        hypo = self.sigmoid(np.matmul(X, self.theta))
        labels = np.zeros(hypo.shape)
        labels[hypo>=0.5] = 1

        return labels

    def probability(self, X):
        return self.sigmoid(np.matmul(X, self.theta))

    def score(self, X, y):
        labels = self.predict(X)
        correct = np.count_nonzero(labels == y)

        size = len(y)
        acc = correct / float(size)
        # print(f"\nscore {correct}/{size}: {acc:0.5f}")
        return acc
    
    @staticmethod
    def sigmoid(z):
        return 1 /(1+np.exp(-z))

    def plot_decision_boundary(self, ax0, ax1, axis=None):
        
        if axis is None:
            ax = plt.gca()
        
        x1 = self.X[:, ax0]
        x2 = self.X[:, ax1]

        # boundary
        xm0, xM0 = min(self.X[:, ax0]) - 1, max(self.X[:, ax0]) + 1
        xm1, xM1 = min(self.X[:, ax1])-1, max(self.X[:, ax1])+1

        YY, XX = np.mgrid[xm1:xM1:100j, xm0:xM0:100j]
        labels = self.predict(
                np.c_[np.ones(XX.size),XX.ravel(), YY.ravel()]
        )
        
        labels = labels.reshape(XX.shape)
        # print(labels)
        out = ax.contourf(XX, YY, labels,
                cmap=plt.cm.coolwarm, alpha=0.5
        )
        ax.scatter(x1, x2, c=self.y, cmap='coolwarm', edgecolor='k')


    def _cost(self, theta, regul=0):
        """
        X is of shape (m, n)
            m : size of the dataset
            n : number of features (including 1's)
        
        y is fo shape (n, ) -> y is only the labels

        """
        X = self.X
        y = self.y
        # theta = theta.reshape((self.nfeatures, 1))

        H = ( 1 / (1+np.exp(-np.matmul(X, theta)) ))# [:, 0] # Our hypothesis
        # print("H = ", H, "shape=", H.shape)
        assert H.shape == y.shape
        m = len(y)

        cost = sum (
            -y*np.log(H)-(1-y)*np.log(1-H)        
        )
        cost = cost / m  # finally divide the entire thing with number of dataset
        return cost

    def _grad(self, theta, regul=0):
        """
        returns the gradient of the cost function
        """

        X = self.X
        y = self.y
        # theta = theta.reshape((self.nfeatures, 1))

        H = ( 1 / (1+np.exp(-np.matmul(X, theta)) )) # [:, 0] # Our hypothesis
        m = len(y)  # Length of our training set
        n = X.shape[1] # Number of training features

        dtheta = np.zeros((n,), dtype='float')
        
        for i in range(m):
            for j in range(n):
                dtheta[j] += (H[i]-y[i])*X[i, j]

        dtheta /= m
        return dtheta

    def trim(self):
        self.X = None
        self.y = None

class MulticlassClassifier(Classifier):
    def fit(self, X, y, **kwargs):
        self.nclasses = len(np.unique(y))

        print("\nStarted trainig")
        print(f"Number of unique classes detected {colored(self.nclasses, 'blue')}")
        print("=="*20)

        self.classifiers = {}

        for i in range(self.nclasses):
            ith_label = np.copy(y)
            ith_label[y==i] = 1
            ith_label[y!=i] = 0

            clf = Classifier()

            print(f" Training classifier {colored(i, 'blue')}")
            print("-"*40)
            clf.fit(X, ith_label, **kwargs)
            clf.trim()
            self.classifiers[i] = clf
            print(f" COST ={colored(clf.cost, 'blue')}\n")

    def predict(self, X):
        probabs = np.zeros((X.shape[0], self.nclasses))
        for i in range(self.nclasses):
            probabs[:, i] = self.classifiers[i].probability(X)     

        return np.argmax(probabs, axis=1)


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # Generating dataset

    # DX, Dy = generate_dataset(size=15000, nclasses=3, nfeatures=4)
    # train_X, test_X, train_y, test_y = train_test_split(
    #     DX, Dy, test_size=0.2
    # )

    # # Plotting the dataset
    # # plot_dataset(X, y, alpha=0.4) 

    # clf = Classifier(nfeatures=3)
    # clf.fit(train_X,train_y)

    # clf.score(test_X, test_y)

    # DX, Dy = make_classification(
    #     n_samples=15000,
    #     n_redundant=0,
    #     n_informative=3,
    #     n_features=3,
    #     n_classes=4,
    #     class_sep=10
    # )

    DX, Dy = load_iris(return_X_y=True)
    DX = np.concatenate(
        (np.ones((DX.shape[0], 1)) , DX),
        axis=1
    )
    # Dycp=Dy[:]
    # Dy[Dycp==1] = 1
    # Dy[Dycp!=1] = 0
    # print(DX)
    train_X, test_X, train_y, test_y = train_test_split(
            DX, Dy, test_size=0.3
    )
    # print(train_y, train_y==2)

    clf = MulticlassClassifier()
    clf.fit(train_X, train_y, iters=500)
    preds = clf.score(test_X, test_y)
    self_score = clf.score(train_X, train_y)
    print("Score is :", preds)
    print("Score on training data:", self_score)


