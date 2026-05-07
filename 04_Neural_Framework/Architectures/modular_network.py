import numpy as np
from xray_utils import show_detailed_math

class Layer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        
    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.biases
        self.output = 1 / (1 + np.exp(-self.z)) # Sigmoid
        return self.output

    def backward(self, output_error, learning_rate):
        # Calculate sensitivity (z' part of chain rule)
        sigmoid_grad = self.output * (1 - self.output)
        d_z = output_error * sigmoid_grad
        
        # Calculate gradients for weights and biases
        d_weights = np.dot(self.input.T, d_z)
        d_biases = np.sum(d_z, axis=0, keepdims=True)
        
        # Calculate error for the PREVIOUS layer (to keep the chain moving)
        input_error = np.dot(d_z, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return input_error

def main():
    # XOR DATA
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_true = np.array([[0], [1], [1], [0]])

    # 🧱 BUILDING THE NETWORK MODULARLY
    # 2 inputs -> 4 neurons -> 1 output
    layer1 = Layer(2, 4)
    layer2 = Layer(4, 1)

    epochs = 100000
    lr = 0.5

    print("--- 🚀 TRAINING MODULAR NETWORK ---")
    for epoch in range(epochs):
        # 1. FORWARD (The signal flows through the stack)
        out1 = layer1.forward(X)
        predictions = layer2.forward(out1)

        # 2. BACKWARD (The error flows back through the stack)
        error = 2 * (predictions - y_true)
        error = layer2.backward(error, lr)
        error = layer1.backward(error, lr)

        if epoch % 2000 == 0:
            loss = np.mean((predictions - y_true)**2)
            print(f"Epoch {epoch} | Loss: {loss:.6f}")

    print("\n--- 🏁 MODULAR XOR RESULTS ---")
    print(np.round(predictions, 2))

if __name__ == "__main__":
    main()
