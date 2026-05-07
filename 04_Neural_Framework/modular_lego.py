import numpy as np

# --- 🧱 PRO DENSE LAYER (Supports Batching) ---
class Dense:
    def __init__(self, input_size, output_size):
        # Xavier/Glorot Initialization for better starting math
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        # This now handles [Batch_Size x Input_Size] @ [Input_Size x Output_Size]
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_error, learning_rate):
        # Math for the whole batch at once
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Update weights and biases (averaging over the batch happens here)
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return input_error

# --- ⚡ THE ReLU SWITCH ---
class ReLU:
    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        # If input was > 0, pass the error. If <= 0, kill the error.
        return output_error * (self.input > 0)

# --- 🎯 THE SIGMOID (Final Layer) ---
class Sigmoid:
    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, output_error, learning_rate):
        # derivative: output * (1 - output)
        return output_error * (self.output * (1 - self.output))

# --- 🏗️ THE MODERN STACK ---
X = np.array([[0,0], [0,1], [1,0], [1,1]]) # No more triple brackets!
y = np.array([[0], [1], [1], [0]])

network = [
    Dense(2, 8),
    ReLU(),      # Modern activation
    Dense(8, 1),
    Sigmoid()    # Output probability
]

# Simple Trainer Loop (BATCH MODE)
epochs = 5000
lr = 0.05

print("🚀 TRAINING PRO BATCH-MODE NETWORK (ReLU + Xavier)...")
for epoch in range(epochs):
    # 1. FORWARD (The whole matrix X goes in at once!)
    output = X
    for layer in network:
        output = layer.forward(output)

    # 2. Compute Loss (MSE)
    loss = np.mean((y - output)**2)

    # 3. BACKWARD (The whole matrix of error flows back!)
    error = 2 * (output - y) / y.size
    for layer in reversed(network):
        error = layer.backward(error, lr)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

print("\n🏁 FINAL BATCH RESULTS:")
print(np.round(output, 4))
