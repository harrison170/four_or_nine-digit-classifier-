from xml.parsers.expat import model
import numpy as np
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

mask_train = (y_train == 4) | (y_train == 9)
mask_test = (y_test == 4) | (y_test == 9)

X_train = X_train[mask_train]
y_train = y_train[mask_train]

X_test = X_test[mask_test]
y_test = y_test[mask_test]
y_train = (y_train == 9).astype(int)
y_test = (y_test == 9).astype(int)
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(X_train.shape) 
print(y_train.shape) 

print(X_test.shape) 
print(y_test.shape)  



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros(output_size)

        self.lr = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]

        output_delta = y - output

        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        self.W2 += self.lr * (self.a1.T.dot(output_delta) / m)
        self.b2 += self.lr * (np.sum(output_delta, axis=0) / m)

        self.W1 += self.lr * (X.T.dot(hidden_delta) / m)
        self.b1 += self.lr * (np.sum(hidden_delta, axis=0) / m)

        if np.random.random() < 0.0001:
            print(np.mean(np.abs(output_delta)))
            print(np.mean(np.abs(hidden_delta)))

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 1000 == 0:
                epsilon = 1e-15
                output_clipped = np.clip(output, epsilon, 1 - epsilon)

                loss = -np.mean(
                    y * np.log(output_clipped) +
                    (1 - y) * np.log(1 - output_clipped)
                )
                print(f"Epoch {epoch}: {loss:.6f}")



nn = NeuralNetwork(
    input_size=784,
    hidden_size=32,
    output_size=1
)

nn.train(X_train, y_train, epochs=10000)

np.savez(
    "mnist_4or9_model.npz",
    W1=nn.W1,
    b1=nn.b1,
    W2=nn.W2,
    b2=nn.b2
)

print("Model saved!")

output = nn.forward(X_test)

print(output.min())
print(output.max())
print(output.mean())

predictions = (output > 0.5).astype(int)

accuracy = np.mean(predictions == y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")

print(output[:20].flatten())
