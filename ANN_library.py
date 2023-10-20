import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score


class ANN:

    def __init__(self, hidden_layer):

        self.parameters = None
        self.hidden_layer = hidden_layer
        self.dimension = None

    def initialisation(self, dimensions):
        parameters = {}
        c = len(dimensions)

        # Initialisation of the parameters as random
        for i in range(1, c):
            parameters['W' + str(i)] = np.random.randn(dimensions[i], dimensions[i - 1])
            parameters['b' + str(i)] = np.random.randn(dimensions[i], 1)

        return parameters

    def foward_propagation(self, parameters, x):

        activations = {'A0': x}
        # len(parameters)//2 represents the length of the neural network
        c = len(parameters) // 2

        for i in range(1, c + 1):
            z = parameters['W' + str(i)].dot(activations['A' + str(i - 1)]) + parameters['b' + str(i)]
            activations['A' + str(i)] = 1 / (1 + np.exp(-z))

        return activations

    def back_propagation(self, y, activations, parameters):

        c = len(parameters) // 2
        m = y.shape[1]

        # the first dZ gradient is the activation of the last layer - y
        dz = activations['A' + str(c)] - y
        gradients = {}

        for i in reversed(range(1, c + 1)):
            gradients['dW' + str(i)] = 1 / m * np.dot(dz, activations['A' + str(i - 1)].T)
            gradients['db' + str(i)] = 1 / m * np.sum(dz, axis=1, keepdims=True)

            # don't need to calculate c0 because it doesn't exist, and we will not use it in a further calcul
            if c > 1:
                dz = np.dot(parameters['W' + str(i)].T, dz) * activations['A' + str(i - 1)] * (
                            1 - activations['A' + str(i - 1)])

        return gradients

    def predict(self, X):

        activation = self.foward_propagation(self.parameters, X)
        c = len(self.parameters) // 2
        Af = activation['A' + str(c)]
        return Af >= 0.5

    def update(self, gradients, parameters, learning_rate):

        c = len(parameters) // 2

        for i in range(1, c + 1):
            parameters['W' + str(i)] = parameters['W' + str(i)] - learning_rate * gradients['dW' + str(i)]
            parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * gradients['db' + str(i)]

        return parameters

    def train(self, X, y, learning_rate, n_itter, training_recap=False):

        self.dimension = list(self.hidden_layer)
        self.dimension.insert(0,X.shape[0])
        self.dimension.append(y.shape[0])

        self.parameters = self.initialisation(self.dimension)

        c = len(self.parameters) //2

        training_history = np.zeros((int(n_itter), 2))

        for i in tqdm(range(n_itter)):

            activations = self.foward_propagation(self.parameters, X)
            gradients = self.back_propagation(y, activations, self.parameters)
            self.parameters = self.update(gradients, self.parameters, learning_rate)
            Af = activations['A' + str(c)]

            if i % 10 == 0 and training_recap:
                training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
                y_pred = self.predict(X)
                training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))

        if training_recap:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(training_history[:, 0], label='train loss')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(training_history[:, 1], label='train acc')
            plt.legend()
            plt.show()


    def accuracy_test(self, parameters, x_test, y_test):
        y_pred = self.predict(x_test)
        print(accuracy_score(y_test, y_pred))

    def normalization_image_classification_2(self, x_train, x_test):
        x_train_normalize = x_train / x_train.max()
        x_test_normalize = x_test / x_train.max()

        return x_train_normalize, x_test_normalize

    def predict_value(self, X):
        activation = self.foward_propagation(self.parameters, X)
        c = len(self.parameters) // 2
        Af = activation['A' + str(c)].mean(axis=1)
        return Af

    def get_para(self):
        return self.parameters





