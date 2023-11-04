import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Inicializa os pesos e o viés aleatoriamente
        self.weights = np.random.rand(X.shape[1])
        self.bias = np.random.rand()

        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                # Calcula a saída da rede
                net_input = np.dot(X[i], self.weights) + self.bias
                output = self.activation(net_input)

                # Calcule o erro
                error = y[i] - output

                # Atualiza os pesos e o viés
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            net_input = np.dot(X[i], self.weights) + self.bias
            output = self.activation(net_input)
            predictions.append(output)
        return np.array(predictions)


X_train = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y_train = np.array([0, 1, 1, 1, 1, 1, 1, 1])  # Saída OR correspondente


adaline = Adaline(learning_rate=0.1, epochs=1000)
adaline.fit(X_train, y_train)

X_test = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
predictions = adaline.predict(X_test)

for i, prediction in enumerate(predictions):
    print(f'Entrada: {X_test[i]}, Saída Adaline: {prediction}')
