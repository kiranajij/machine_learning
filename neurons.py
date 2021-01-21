import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))
def softmax(z):
    pass

class LayerBase:
    def __init__(self, input_shape, output_shape):
        """
        Initialize a Layer of neuron 
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Weight matrix of the layer
        self.W = np.random.randn(output_shape, input_shape)
        self.b = np.random.randn( output_shape )

    def set_coeffs(self, W, b):
        self.W = W
        self.b = b

class SigmoidLayer(LayerBase):
    def feed_forward(self, input_weights):
        output = np.matmul(self.W, input_weights) + self.b
        return sigmoid(output)


class Model:
    def __init__(self, layers = None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_layers(self, layers):
        self.layers.extend(layers)

    def predict(self, X):

        for layer in self.layers:
            X = layer.feed_forward(X)

        return X

if __name__ == '__main__':
    model = Model()

    l1 = SigmoidLayer(2, 1)
    W  = np.array([20, 20], dtype='float')
    b  = np.array([-30], dtype='float')
    l1.set_coeffs(W, b)
    model.add_layer(l1)

    for i in range(2):
        for j in range(2):
            print(f"{i=}, {j=}")
            X = np.array([i, j], dtype='float')
            pred = model.predict(X)
            print(pred)
