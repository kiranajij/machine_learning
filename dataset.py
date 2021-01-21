import numpy as np
# import matplotlib.pyplot as plt

CLASSIFICATION = 0
REGRESSION     = 1

def generate_dataset(size=1000, nfeatures=2,
        nclasses=2, task=CLASSIFICATION):

    np.random.seed(420)
    X = np.random.randn(size, nfeatures)

    X = np.concatenate(
        (
            np.ones((size, 1)), X
        ),
        axis=1
    )

    thetas = np.random.randn(nfeatures+1, 1)

    labels = X.dot(thetas)[:, 0]*10

    if task == REGRESSION:
        return X, labels

    # Otherwise return the classification splitted at the mean
    if nclasses == 2:
        splitter = labels.mean()

        labels[labels >= splitter] = 1
        labels[labels < splitter]  = 0


        return X, labels

    if nclasses == 3:
        m, sigma = labels.mean(), 3
        splitter1, splitter2 = m-sigma, m+sigma

        labels[labels<splitter1] = 0
        for i in range(len(labels)):
            if labels[i] >=splitter1 and labels[i]<splitter2:
                labels[i] = 1
        labels[labels>splitter2] = 2

        print(np.unique(labels))
        return X, labels

def plot_dataset(X, y, **options):
    import matplotlib.pyplot as plt

    plt.style.use('ggplot')
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap='plasma', edgecolor='k', **options)
    plt.show()

if __name__=='__main__':
    import matplotlib.pyplot as plt

    X, labels =  generate_dataset(size=100, nfeatures=2, task=CLASSIFICATION)
    plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='plasma')
    plt.show()
